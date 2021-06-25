import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchpruner as pruner
import torchpruner.model_tools as model_tools
import torchslim
import torchslim.slim_solver as slim_solver
from torchslim.modules.rep_modules import merge_conv_bn

from collections import defaultdict, OrderedDict
import numpy as np
from sklearn.cluster import KMeans

import copy


def cluster_by_k_mean(model, prune_module_names, cluster_percent):
    cluster_dict = {}
    for prune_module_name in prune_module_names:
        conv = model_tools.get_object(model, prune_module_name)
        if conv is None or not isinstance(conv, nn.Conv2d):
            raise RuntimeError(
                "Can not find the module name %s or the module is not the instance of nn.Conv2d"
                % (prune_module_name)
            )
        weight = conv.weight.detach().numpy()
        if conv.bias is not None:
            bias = conv.bias.detach().cpu().numpy()
            features = np.concatenate(
                (weight.reshape(weight.shape[0], -1), bias.reshape(weight.shape[0], 1)),
                axis=1,
            )
        else:
            features = weight.reshape(weight.shape[0], -1)
        feature_numbers = features.shape[0]
        num_cluster = int(cluster_percent * features.shape[0])

        km = KMeans(n_clusters=num_cluster)
        km.fit(features)
        result = []
        for j in range(num_cluster):
            result.append([])
        for i, c in enumerate(km.labels_):
            result[c].append(i)
        for r in result:
            r.sort()
            assert len(r) > 0
        cluster_dict[prune_module_name] = result
    return cluster_dict


def generate_relative_cluser(model, inputs, cluster_dict):
    graph = pruner.ONNXGraph(model)
    graph.build_graph(inputs)
    relative_cluster_dict = {}
    for key in cluster_dict.keys():
        module = graph.modules[key]
        cluster_list = cluster_dict[key]

        analysis_result = module.cut_analysis("weight", index=[0], dim=0)
        analysis_result = analysis_result["terminal"]
        for key in analysis_result.keys():
            if key in relative_cluster_dict:
                raise RuntimeError("wrong module name setting")
            if len(analysis_result[key][0]) != 0:
                relative_cluster_dict[key] = []
        for index in cluster_list:
            analysis_result = module.cut_analysis("weight", index=index, dim=0)
            analysis_result = analysis_result["terminal"]
            for key in analysis_result.keys():
                if len(analysis_result[key][0]) != 0:
                    relative_cluster_dict[key].append(analysis_result[key][0])

    for key in relative_cluster_dict.keys():
        feature_size = graph.modules[key].terminal_node.size(0)
        relative_cluster_dict[key] = (relative_cluster_dict[key], feature_size)

    return relative_cluster_dict


def generate_merge_matrix(cluster_dict, device):
    merge_matrix_dict = {}
    for key in cluster_dict.keys():
        cluster_list, size = cluster_dict[key]
        merge_matrix = np.zeros((size, size))
        for cluster in cluster_list:
            for i in cluster:
                for j in cluster:
                    merge_matrix[i][j] = 1 / len(cluster)
        merge_matrix_dict[key] = torch.Tensor(merge_matrix).cuda(device)
    return merge_matrix_dict


def generate_decay_matrix(
    cluster_dict, device, weight_decay, weight_decay_bias, centri_strength, model
):
    decay_matrix_dict = {}
    for key in cluster_dict.keys():
        cluster_list, size = cluster_dict[key]
        obj = model_tools.get_object(model, key)
        decay_matrix = np.zeros((size, size))
        if len(obj.size()) == 1:
            strength_gamma = 0.1
            apply_decay = weight_decay_bias
        else:
            strength_gamma = 1.0
            apply_decay = weight_decay
        for cluster in cluster_list:
            for i in cluster:
                decay_matrix[i][i] = apply_decay + centri_strength * strength_gamma
                for j in cluster:
                    decay_matrix[i][j] -= (
                        centri_strength * strength_gamma / len(cluster)
                    )
        decay_matrix_dict[key] = torch.Tensor(decay_matrix).cuda(device)
    return decay_matrix_dict


def deploy_model(model, optimizer, inputs, cluster_dict):
    print("to deploy model....")
    for key in cluster_dict.keys():
        cut_index = []
        cluster = cluster_dict[key]
        graph = pruner.ONNXGraph(model)
        graph.build_graph(inputs)
        module = graph.modules[key]
        for c_list in cluster:
            if len(c_list) == 1:
                continue
            analysis_result = module.cut_analysis("weight", c_list, 0)
            analysis_result = analysis_result["terminal"]
            for module_key in analysis_result.keys():
                conv_module_key = ".".join(module_key.split(".")[:-1])
                obj = model_tools.get_object(model, conv_module_key)
                if not isinstance(obj, nn.Conv2d):
                    continue
                if len(analysis_result[module_key]) != 4:
                    continue
                if len(analysis_result[module_key][1]) == 0:
                    continue
                merge_index = analysis_result[module_key][1]
                obj.weight.data[:, merge_index[0], :, :] = torch.sum(
                    obj.weight.data[:, merge_index], dim=1, keepdim=False
                )
            cut_index.extend(c_list[1:])
        if len(cut_index) != 0:
            analysis_result = module.cut_analysis("weight", cut_index, 0)
            model, _ = pruner.set_cut(model, analysis_result)
            if optimizer is not None:
                optimizer, _ = pruner.set_cut_optimizer(
                    model, optimizer, analysis_result
                )
    print(model)
    return model


# just get the linear structure
def get_linear_conv_names(graph):
    modules = graph.modules
    conv_names = []
    for _, module in modules.items():
        if isinstance(module.nn_object, (nn.Conv2d)):
            cut_dict = module.cut_analysis("weight", [0], 0)
            terminal_dict = cut_dict["terminal"]
            key_length = len(terminal_dict.keys())
            if key_length > 7:
                continue
            conv_names.append(module.name)
    return conv_names


def get_all_conv_names(graph):
    modules = graph.modules
    conv_names = []
    for _, module in modules.items():
        if isinstance(module.nn_object, (nn.Conv2d)):
            conv_names.append(module.name)
    return conv_names


strategy_mapping = {
    "linear_conv": get_linear_conv_names,
    "all_conv": get_all_conv_names,
}


def get_target_module_names(model, graph_inputs, strategy="linear"):
    # the first step compress the RepModule to the single conv
    model = copy.deepcopy(model)
    # create the graph
    graph = pruner.ONNXGraph(model)
    graph.build_graph(graph_inputs)
    return strategy_mapping[strategy](graph)


def flops(model, graph_inputs):
    graph = pruner.ONNXGraph(model)
    graph.build_graph(graph_inputs)
    return graph.flops()


# the init hook
# insert the compactor into the model and get the init flops
def init_hook(self):
    # prepare the sample input
    if self.config["input_shapes"] is None:
        input_shapes = self.infer_input_shapes()
    else:
        input_shapes = self.config["input_shapes"]
    graph_inputs = []
    for input_shape in input_shapes:
        graph_inputs.append(torch.zeros(1, *input_shape))
    graph_inputs = tuple(graph_inputs)

    if self.config["prune_module_names"] is not None:
        target_module_names = self.config["prune_module_names"]
    else:
        target_module_names = get_target_module_names(
            self.model, graph_inputs, self.config["auto_find_module_strategy"]
        )

    print("The pruning module is:")
    print(target_module_names)

    base_cluster_dict = cluster_by_k_mean(
        self.model, target_module_names, self.config["cluster_percent"]
    )
    model = copy.deepcopy(self.model)
    model.cpu()
    init_flops = flops(model, graph_inputs)
    print("The init flops is: %.4f" % (init_flops))
    model = deploy_model(model, None, graph_inputs, base_cluster_dict)
    target_flops = flops(model, graph_inputs)
    print("The target flops is: %.4f" % (target_flops))

    relative_cluster_dict = generate_relative_cluser(
        self.model, graph_inputs, base_cluster_dict
    )
    merge_matrix = generate_merge_matrix(
        relative_cluster_dict, self.variable_dict["base_device"]
    )
    decay_matrix = generate_decay_matrix(
        relative_cluster_dict,
        self.variable_dict["base_device"],
        self.config["weight_decay"],
        0.0,
        self.config["centri_strength"],
        self.model,
    )

    merge_param_dict = {}
    for key in merge_matrix.keys():
        obj = model_tools.get_object(self.model, key)
        merge_param_dict[obj] = merge_matrix[key]
    decay_param_dict = {}
    for key in decay_matrix.keys():
        obj = model_tools.get_object(self.model, key)
        decay_param_dict[obj] = decay_matrix[key]

    # save the variables
    self.variable_dict["graph_inputs"] = graph_inputs
    self.variable_dict["target_module_names"] = target_module_names
    self.variable_dict["base_cluster_dict"] = base_cluster_dict
    self.variable_dict["relative_cluster_dict"] = relative_cluster_dict
    self.variable_dict["merge_matrix"] = merge_param_dict
    self.variable_dict["decay_matrix"] = decay_param_dict

    self.variable_dict["allow_save"] = False


def epoch_begin_hook(self):
    if self.variable_dict["epoch"] == self.config["prune_epoch"]:
        self.variable_dict["allow_save"] = True
        self.variable_dict["apply_weight_decay"] = self.config["weight_decay"]
        print("Pruning the model...")
        self.model = deploy_model(
            self.model,
            self.optimizer,
            self.variable_dict["graph_inputs"],
            self.variable_dict["base_cluster_dict"],
        )


def optimizer_generator(params, config):
    return torch.optim.SGD(params, lr=0.0)


def scheduler_generator(optimizer, config):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epoch"])


class CSGDSolver(slim_solver.CommonSlimSolver):
    def __init__(self, model, config):
        super(CSGDSolver, self).__init__(model, config)
        self.regist_init_hook(init_hook)
        self.regist_epoch_begin_hook(epoch_begin_hook)
        self.variable_dict["apply_weight_decay"] = 0

    __config_setting__ = [
        ("task_name", str, "default", False, "The task name"),
        ("lr", float, 0.01, False, "The learning rate of the optimizer"),
        ("epoch", int, 360, False, "The total epoch to train the model"),
        ("prune_epoch", int, 100, False, "The epoch to prune the model"),
        ("batch_size", int, 128, False, "The batch size per step"),
        ("test_batch_size", int, 128, False, "The evaluation batch size per step"),
        ("momentum", float, 0.9, False, "The momentum for the optimizer"),
        ("weight_decay", float, 1e-4, False, "The wegith decay for the parameters"),
        ("centri_strength", float, 3e-3, False, "The lasso decay for the compactor"),
        (
            "input_shapes",
            list,
            None,
            True,
            "A 2 dim list, representing the input shapes of the data, if None, \
        the first item size of the dataset will be used as the input size",
        ),
        (
            "prune_module_names",
            list,
            None,
            True,
            "The module names to be pruned, just support BatchNorm2d and Conv2d",
        ),
        (
            "auto_find_module_strategy",
            str,
            "linear_bn",
            False,
            "The strategy to determine the name of module layers to be cut if the prune_module_names is None, \
        support linear and all",
        ),
        ("cluster_percent", float, None, False, "The cluster percent of the model"),
        ("num_workers", int, 0, False, "The number of workers to read data"),
        ("save_keyword", str, "acc", False, "The keyword for save"),
        ("save_dir", str, "checkpoints", False, "The model save dir"),
        ("devices", list, None, False, "The device to be used in training"),
        ("log_interval", int, 20, False, "The interval to report the log"),
        # generate the optimizer
        (
            "optimizer_generator",
            "function",
            optimizer_generator,
            False,
            "The optimizer generator (params,config)->optimizer",
        ),
        # generate the scheduler
        (
            "scheduler_generator",
            "function",
            scheduler_generator,
            True,
            "the scheduler generator for the task (optmizer,config)->scheduler",
        ),
        # predict the result
        (
            "predict_function",
            "function",
            None,
            False,
            "get the prediction of the data (model,batch_data)->predict",
        ),
        # calculate the loss for one iteration
        (
            "calculate_loss_function",
            "function",
            None,
            False,
            "(predict,batch_data)->loss",
        ),
        # get the evaluate result for one iteration
        (
            "evaluate_function",
            "function",
            None,
            True,
            "(predict,batch_data)->evaluate_dict",
        ),
        # get the dataset
        (
            "dataset_generator",
            "function",
            None,
            True,
            "()->dataset_train,dataset_validation",
        ),
    ]

    # infer the input sizes
    def infer_input_shapes(self):
        samples = iter(self.trainloader).__next__()
        input_size = samples[0].size()[1:]
        return [list(input_size)]

    # overwrite the generate_params setting
    def generate_params_setting(self):
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        # weight_decay = self.variable_dict["apply_weight_decay"]
        momentum = self.config["momentum"]
        base_lr = self.config["lr"]
        weight_decay = self.config["weight_decay"]

        params = []
        for key, value in model.named_parameters():
            apply_weight_decay = weight_decay
            apply_momentum = momentum
            apply_lr = base_lr
            if not value.requires_grad:
                continue
            parent_key = ["self"] + key.split(".")[:-1]
            parent_key = ".".join(parent_key)
            parent_object = model_tools.get_object(model, parent_key)
            if isinstance(parent_object, nn.BatchNorm2d):
                apply_weight_decay = 0.0
            if (
                isinstance(parent_object, (nn.Conv2d, nn.ConvTranspose2d))
                and parent_object.groups == parent_object.in_channels
            ):
                apply_weight_decay = 0.0
            item_list = key.split(".")
            if len(item_list) <= 3:
                continue
            grand_parent_key = ["self"] + key.split(".")[:-2]
            grand_parent_key = ".".join(grand_parent_key)
            grand_parent = model_tools.get_object(model, grand_parent_key)
            if "bias" in key:
                apply_lr = 2 * base_lr
                apply_weight_decay = 0.0
            else:
                apply_lr = base_lr
            params += [
                {
                    "params": [value],
                    "lr": apply_lr,
                    "weight_decay": weight_decay,
                    "momentum": apply_momentum,
                }
            ]
        return params

    # overwrite the after_loss_backward
    def after_loss_backward(self):
        if self.variable_dict["allow_save"]:
            return
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        for key, param in model.named_parameters():
            if param in self.variable_dict["merge_matrix"]:
                p_dim = param.dim()
                p_size = param.size()
                if p_dim == 4:
                    param_mat = param.reshape(p_size[0], -1)
                    g_mat = param.grad.reshape(p_size[0], -1)
                elif p_dim == 1:
                    param_mat = param.reshape(p_size[0], 1)
                    g_mat = param.grad.reshape(p_size[0], 1)
                else:
                    assert p_dim == 2
                    param_mat = param
                    g_mat = param.grad
                csgd_gradient = self.variable_dict["merge_matrix"][param].matmul(
                    g_mat
                ) + self.variable_dict["decay_matrix"][param].matmul(param_mat)
                param.grad.copy_(csgd_gradient.reshape(p_size))

    def save_model(self):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        deploy_model = copy.deepcopy(model)
        deploy_model = deploy_model.cpu()
        torch.save(
            {
                self.config["save_keyword"]: self.variable_dict["save_target"],
                "net": deploy_model,
            },
            self.variable_dict["save_path"],
        )
