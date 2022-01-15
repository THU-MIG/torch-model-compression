import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchpruner as pruner
import torchpruner.model_tools as model_tools
import torchslim
import torchslim.slim_solver as slim_solver
from torchslim.modules.rep_modules import (
    merge_conv_bn,
    merge_conv_compactor,
    Compactor,
    ModuleCompactor,
    RepModule,
)

from collections import defaultdict, OrderedDict

import copy


def neg_index(index, size):
    n_index = np.arange(0, size)
    mask = np.ones(size) > 0
    mask[index] = False
    n_index = n_index[mask]
    return list(n_index)


# just get the linear structure
# conv->bn->relu->conv
def get_linear_bn_names(graph):
    modules = graph.modules
    bn_names = []
    for _, module in modules.items():
        if isinstance(module.nn_object, nn.BatchNorm2d):
            cut_dict = module.cut_analysis("weight", [0], 0)
            terminal_dict = cut_dict["terminal"]
            key_length = len(list(terminal_dict.keys()))
            if key_length > 7:
                continue
            bn_names.append(module.name)
    return bn_names


def get_linear_conv_names(graph):
    modules = graph.modules
    conv_names = []
    for _, module in modules.items():
        if isinstance(module.nn_object, (nn.Conv2d, nn.ConvTranspose2d)):
            cut_dict = module.cut_analysis("weight", [0], 0)
            terminal_dict = cut_dict["terminal"]
            key_length = len(terminal_dict.keys())
            if key_length > 7:
                continue
            conv_names.append(module.name)
    return conv_names


# get all the bn names
def get_all_bn_names(graph):
    modules = graph.modules
    bn_names = []
    for _, module in modules.items():
        if isinstance(module.nn_object, nn.BatchNorm2d):
            bn_names.append(module.name)
    return bn_names


def get_all_conv_names(graph):
    modules = graph.modules
    conv_names = []
    for _, module in modules.items():
        if isinstance(module.nn_object, (nn.Conv2d, nn.ConvTranspose2d)):
            conv_names.append(module.name)
    return conv_names


strategy_mapping = {
    "linear_bn": get_linear_bn_names,
    "all_bn": get_all_bn_names,
    "linear_conv": get_linear_conv_names,
    "all_conv": get_all_conv_names,
}


def RepModule_convert_hook(name, origin_object):
    return origin_object.convert()


def get_target_module_names(model, graph_inputs, strategy):
    # the first step compress the RepModule to the single conv
    model = copy.deepcopy(model)
    module_names = model_tools.get_names_by_class(model, RepModule)
    model = model_tools.replace_object_by_names(
        model, module_names, RepModule_convert_hook
    )
    # create the graph
    graph = pruner.ONNXGraph(model)
    graph.build_graph(graph_inputs)
    return strategy_mapping[strategy](graph)


def module_compactor_replace_function(name, origin_object):
    module_compactor = ModuleCompactor(origin_object).to(
        origin_object.weight.data.device
    )
    return module_compactor


# Conv BN and Compactor
def merge_conv_bn_compactor_hook(names, object_groups):
    conv, bn, compactor = object_groups
    conv = merge_conv_bn(conv, bn)
    return merge_conv_compactor(conv, compactor), nn.Identity(), nn.Identity()


# Conv N
def merge_conv_compactor_hook(names, object_groups):
    conv, compactor = object_groups
    return merge_conv_compactor(conv, compactor), nn.Identity()


def get_bn_channels(model, names):
    return_dict = OrderedDict()
    for name in names:
        nn_object = pruner.model_tools.get_object(model, name)
        return_dict[name] = nn_object.compactor.conv.out_channels
    return return_dict


def deploy_convert(model, graph_inputs):
    model = copy.deepcopy(model)
    model = model.cpu()
    # rep module
    model = model_tools.replace_object_by_class(
        model, RepModule, RepModule_convert_hook
    )
    current_graph = pruner.ONNXGraph(model)
    current_graph.build_graph(graph_inputs)
    # conv and compactor
    name_groups = pruner.model_tools.get_name_groups_by_classes(
        current_graph, [(nn.Conv2d, nn.ConvTranspose2d), Compactor]
    )
    model = model_tools.replace_object_by_name_groups(
        model, name_groups, merge_conv_compactor_hook
    )
    # conv bn and compactor
    name_groups = pruner.model_tools.get_name_groups_by_classes(
        current_graph, [(nn.Conv2d, nn.ConvTranspose2d), nn.BatchNorm2d, Compactor]
    )
    model = model_tools.replace_object_by_name_groups(
        model, name_groups, merge_conv_bn_compactor_hook
    )
    return model


def prune_model(graph, model, optimizer, prune_groups=1, group_size=8, min_channels=8):
    module_names = []
    module_lasso_value = []
    for name, module in graph.modules.items():
        nn_object = module.nn_object
        if isinstance(nn_object, Compactor):
            weight = nn_object.conv.weight.data.cpu().numpy()
            lasso_value = np.sum(weight * weight, axis=(1, 2, 3))
            module_names.append(name)
            module_lasso_value.append(lasso_value)
    for c_group in range(0, prune_groups):
        min_module_name = None
        min_lasso_value = 1e100
        name_index = -1
        min_index = None
        for i in range(0, len(module_names)):
            module_name, lasso_value = module_names[i], module_lasso_value[i]
            remain_channels = len(lasso_value)
            if remain_channels <= group_size or remain_channels <= min_channels:
                continue
            index = np.argsort(lasso_value)
            index_group = index[:group_size]
            lasso_sum_value = np.sum(lasso_value[index_group])
            if lasso_sum_value < min_lasso_value:
                min_lasso_value = lasso_sum_value
                min_module_name = module_name
                min_index = index_group
                name_index = i
        if min_module_name is None:
            raise RuntimeError("The model can not be pruned to target flops")
        print("Cutting layer is: " + min_module_name)
        module_lasso_value[name_index] = module_lasso_value[name_index][
            neg_index(min_index, len(module_lasso_value[name_index]))
        ]
        analysis_result = graph.modules[min_module_name].cut_analysis(
            "conv.weight", index=min_index, dim=0
        )
        model, _ = pruner.set_cut(model, analysis_result)
        optimizer, _ = pruner.set_cut_optimizer(model, optimizer, analysis_result)
    return model, optimizer


def flops(model, graph_inputs):
    model = deploy_convert(model, graph_inputs)
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

    # remove the bn layer without conv or with depthwise conv
    model = copy.deepcopy(self.model)
    module_names = model_tools.get_names_by_class(model, RepModule)
    model = model_tools.replace_object_by_names(
        model, module_names, RepModule_convert_hook
    )
    graph = pruner.ONNXGraph(model)
    graph.build_graph(graph_inputs)
    name_groups = model_tools.get_name_groups_by_classes(
        graph, [(nn.Conv2d, nn.ConvTranspose2d), nn.BatchNorm2d]
    )
    filtered_module_names = []
    for conv_name, bn_name in name_groups:
        if (
            model_tools.get_object(model, conv_name).groups == 1
            and bn_name in target_module_names
        ):
            filtered_module_names.append(bn_name)
    names = model_tools.get_names_by_class(model, (nn.Conv2d, nn.ConvTranspose2d), True)
    for name in names:
        if (
            name in target_module_names
            and model_tools.get_object(model, name).groups == 1
        ):
            filtered_module_names.append(name)
    # sort the names:
    sorted_filtered_module_names = []
    for name in target_module_names:
        if name in filtered_module_names:
            sorted_filtered_module_names.append(name)
    target_module_names = sorted_filtered_module_names
    print("The pruning module is:")
    print(target_module_names)

    # insert the compactor
    self.model = model_tools.replace_object_by_names(
        self.model, target_module_names, module_compactor_replace_function
    )
    current_flops = flops(self.model, graph_inputs)

    # save the variables
    self.variable_dict["graph_inputs"] = graph_inputs
    self.variable_dict["target_module_names"] = target_module_names
    self.variable_dict["init_flops"] = current_flops
    self.variable_dict["current_flops"] = current_flops
    print("The init flops is: %.4f" % (current_flops))
    print("The target flops is: %.4f" % ((1 - self.config["prune_rate"]) * current_flops))

    # set the allow save to be false
    self.variable_dict["allow_save"] = False


# before iteration hook
def before_iteration_hook(self):
    if self.variable_dict["epoch"] >= self.config["warmup_epoch"]:
        self.variable_dict["prune_iteration"] += 1


# the iteration hook
def after_iteration_hook(self):
    current_flops = self.variable_dict["current_flops"]
    init_flops = self.variable_dict["init_flops"]
    graph_inputs = self.variable_dict["graph_inputs"]
    target_module_names = self.variable_dict["target_module_names"]

    if self.variable_dict["prune_iteration"] % self.config["prune_interval"] == 0:
        if current_flops < (1 - self.config["prune_rate"]) * init_flops:
            print("reach the target flops no need to prune")
            self.variable_dict["allow_save"] = True
        else:
            print(">>>>>>>>>>>>>>>>>>>>Pruning the model >>>>>>>>>>>>>>>>>>>>")
            current_graph = pruner.ONNXGraph(self.model)
            current_graph.build_graph(graph_inputs)
            self.model, self.optimizer = prune_model(
                current_graph,
                self.model,
                self.optimizer,
                self.config["prune_groups"],
                self.config["group_size"],
                self.config["min_channels"],
            )
            current_flops = flops(self.model, graph_inputs)
            bn_channels = get_bn_channels(self.model, target_module_names)
            print("The cutting bn channel is:")
            for key in bn_channels.keys():
                print(key + ": " + str(bn_channels[key]))
            print("The new flops is %.4f (%.1f%% of %.4f)" % (current_flops,
                current_flops / init_flops * 100, init_flops))
            self.variable_dict["current_flops"] = current_flops


def optimizer_generator(params, config):
    return torch.optim.SGD(params, lr=0.0)


def scheduler_generator(optimizer, config):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epoch"])


class ResRepSolver(slim_solver.CommonSlimSolver):
    def __init__(self, model, config):
        super(ResRepSolver, self).__init__(model, config)
        self.variable_dict["prune_iteration"] = 1
        self.regist_init_hook(init_hook)
        self.regist_iteration_begin_hook(before_iteration_hook)
        self.regist_iteration_end_hook(after_iteration_hook)

    __config_setting__ = [
        ("task_name", str, "default", False, "The task name"),
        (
            "save_deploy_format",
            bool,
            True,
            False,
            "convert the ACNet to conv when saving the model",
        ),
        ("lr", float, 0.01, False, "The learning rate of the optimizer"),
        ("epoch", int, 360, False, "The total epoch to train the model"),
        ("batch_size", int, 128, False, "The batch size per step"),
        ("test_batch_size", int, 128, False, "The evaluation batch size per step"),
        ("warmup_epoch", int, 5, False, "The total train epoch before pruning"),
        ("momentum", float, 0.9, False, "The momentum for the optimizer"),
        ("compactor_momentum", float, 0.99, False, "The momentum value for compactor"),
        ("weight_decay", float, 1e-4, False, "The wegith decay for the parameters"),
        ("lasso_decay", float, 1e-4, False, "The lasso decay for the compactor"),
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
        ("prune_rate", float, None, False, "The prune rate of the model"),
        ("prune_interval", int, 200, False, "The prune iteration per pruning"),
        ("prune_groups", int, 1, False, "The prune groups prune pruning"),
        ("group_size", int, 8, False, "The channels to be pruned per group"),
        ("min_channels", int, 8, False, "The min channels that layer remain"),
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
        weight_decay = self.config["weight_decay"]
        momentum = self.config["momentum"]
        base_lr = self.config["lr"]

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
            if isinstance(grand_parent, Compactor):
                apply_weight_decay = 0.0
                apply_momentum = 0.99
            if "bias" in key:
                apply_lr = 2 * base_lr
                apply_weight_decay = 0.0
            else:
                apply_lr = base_lr
            params += [
                {
                    "params": [value],
                    "lr": apply_lr,
                    "weight_decay": apply_weight_decay,
                    "momentum": apply_momentum,
                }
            ]
        return params

    # overwrite the after_loss_backward
    def after_loss_backward(self):
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        for key, value in model.named_parameters():
            split_key = key.split(".")
            if len(split_key) <= 3:
                continue
            split_key = ["self"] + split_key[:-2]
            grand_parent_key = ".".join(split_key)
            nn_object = model_tools.get_object(model, grand_parent_key)
            if isinstance(nn_object, Compactor):
                lasso_grad = value.data * (
                    (value.data ** 2).sum(dim=(1, 2, 3), keepdim=True) ** (-0.5)
                )
                value.grad.data.add_(self.config["lasso_decay"], lasso_grad)

    def save_model(self):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        deploy_model = copy.deepcopy(model)
        deploy_model = deploy_model.cpu()
        if self.config["save_deploy_format"]:
            graph_inputs = self.variable_dict["graph_inputs"]
            deploy_model = deploy_convert(model, graph_inputs)
        torch.save(
            {
                self.config["save_keyword"]: self.variable_dict["save_target"],
                "net": deploy_model,
            },
            self.variable_dict["save_path"],
        )
