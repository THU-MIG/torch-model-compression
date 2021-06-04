import sys

import torch
import copy
import torchslim
import torchslim.slim_solver as slim_solver
import torch.nn as nn
import numpy as np

import torchpruner as pruner
import torchpruner.model_tools as model_tools

import onnx

from . import qat_tools


def optimizer_generator(params, config):
    return torch.optim.SGD(params, lr=0.0)


def scheduler_generator(optimizer, config):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epoch"])


def init_hook_function(self):
    sample_inputs, _ = iter(self.trainloader).__next__()
    input_list = []
    if not isinstance(sample_inputs, tuple):
        input_list.append(sample_inputs[:1])
    else:
        for memeber in sample_inputs:
            input_list.append(memeber[:1])
    graph_inputs = tuple(input_list)
    self.variable_dict["graph_inputs"] = graph_inputs
    pruner.activate_function_module()
    self.model = qat_tools.prepare_qat(
        self.model, graph_inputs, qat_tools.tensorrt_qconfig
    )


def end_hook_function(self):
    pruner.deactivate_function_module()


class QATSolver(slim_solver.CommonSlimSolver):
    __config_setting__ = [
        ("task_name", str, "defualt", False, "The task name"),
        ("lr", float, 0.001, False, "The learning rate of the optimizer"),
        ("epoch", int, 360, False, "The total epoch to train the model"),
        ("batch_size", int, 128, False, "The batch size per step"),
        ("test_batch_size", int, 128, False, "The evaluation batch size per step"),
        ("momentum", float, 0.9, False, "The momentum for the optimizer"),
        ("weight_decay", float, 1e-4, False, "The wegith decay for the parameters"),
        ("save_keyword", str, "acc", False, "The keyword for save"),
        ("save_dir", str, "checkpoints", False, "The model save dir"),
        ("num_workers", int, 0, False, "The number of workers to read data"),
        ("devices", list, None, False, "The device to be used in training"),
        ("log_interval", int, 200, False, "The interval to report the log"),
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

    def __init__(self, model, config):
        super(QATSolver, self).__init__(model, config)
        self.regist_init_hook(init_hook_function)
        self.regist_end_hook(end_hook_function)

    def generate_params_setting(self):
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        base_lr = self.config["lr"]
        weight_decay = self.config["weight_decay"]
        momentum = self.config["momentum"]
        params = []
        for key, value in model.named_parameters():
            apply_weight_decay = weight_decay
            apply_momentum = momentum
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

    def save_model(self):
        save_path = os.path.join(self.config["save_dir"], self.config["task_name"])
        save_path = os.path.join(save_path, "model.trt")
        model = copy.deepcopy(self.model)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = model.cpu()
        model = qat_tools.merge_convbn2d(model)
        qat_tools.export_onnx(model, self.variable_dict["graph_inputs"], "tmp.onnx")
        trt_engin = qat_tools.export_trt("tmp.onnx")
        with open(save_path, "wb") as file:
            file.write(trt_engin.serialize())
