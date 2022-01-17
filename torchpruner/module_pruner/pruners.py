from collections import OrderedDict
import torch
import torch.nn as nn

import numpy as np

from typing import Dict, List
from .prune_function import *


class BasePruner(object):
    def __init__(self, name):
        self.name = name

    # set the value to be zeros and return the context
    def set_zero(self, nn_module, cut_dict):
        raise NotImplementedError("The set_zero is not implemented")

    # recovery from zeros
    def recovery_zero(self, nn_module, cut_dict, context):
        raise NotImplementedError("The recovery_zero is not implemented")

    # cut the value from zeros and return the context
    def set_cut(self, nn_module, cut_dict):
        raise NotImplementedError("The set_cut is not implemented")

    # reconvery the model from the context
    def recovery_cut(self, nn_module, cut_dict, context):
        raise NotImplementedError("The recovery_cut is not implemented")


class TensorPruner(BasePruner):
    def __init__(self, name):
        super(TensorPruner, self).__init__(name)

    def set_zero(self, data, cut_dict):
        if self.name not in cut_dict["terminal"]:
            return data, {}
        param_context = {}
        cut_dims = cut_dict["terminal"][self.name]
        data, param_list = set_zero_tensor(data, cut_dims)
        param_context[self.name] = param_list
        return data, param_context

    def recovery_zero(self, data, cut_dict, param_context):
        if self.name not in cut_dict["terminal"]:
            return data
        cut_dims = cut_dict["terminal"][self.name]
        if self.name in param_context.keys():
            param_list = param_context[self.name]
        else:
            param_list = None
        return recovery_zero_tensor(data, cut_dims, param_list)

    def set_cut(self, data, cut_dict):
        if self.name not in cut_dict["terminal"]:
            return data, {}
        param_context = {}
        cut_dims = cut_dict["terminal"][self.name]
        data, param_list = set_cut_tensor(data, cut_dims)
        param_context[self.name] = param_list
        return data, param_context

    def recovery_cut(self, data, cut_dict, param_context):
        if self.name not in cut_dict["terminal"]:
            return data
        if self.name in param_context.keys():
            param_list = param_context[self.name]
        else:
            param_list = None
        cut_dims = cut_dict["terminal"][self.name]
        return recovery_cut_tensor(data, cut_dims, param_list)


class ParameterPruner(TensorPruner):
    def __init__(self, name):
        super(ParameterPruner, self).__init__(name)


class ConvPruner(BasePruner):
    def __init__(self, name):
        super(ConvPruner, self).__init__(name)
        self.weight_pruner = ParameterPruner(name + ".weight")
        self.bias_pruner = ParameterPruner(name + ".bias")

    # set the value to be zeros and return the context
    def set_zero(self, nn_module, cut_dict):
        nn_module.weight, weight_context = self.weight_pruner.set_zero(
            nn_module.weight, cut_dict
        )
        nn_module.bias, bias_context = self.bias_pruner.set_zero(
            nn_module.bias, cut_dict
        )
        return nn_module, {**weight_context, **bias_context}

    # recovery from zeros
    def recovery_zero(self, nn_module, cut_dict, param_context):
        nn_module.weight = self.weight_pruner.recovery_zero(
            nn_module.weight, cut_dict, param_context
        )
        nn_module.bias = self.bias_pruner.recovery_zero(
            nn_module.bias, cut_dict, param_context
        )
        return nn_module

    # cut the value from zeros and return the context
    def set_cut(self, nn_module, cut_dict):
        nn_module.weight, weight_context = self.weight_pruner.set_cut(
            nn_module.weight, cut_dict
        )
        nn_module.bias, bias_context = self.bias_pruner.set_cut(
            nn_module.bias, cut_dict
        )
        onnx_name = self.name + ".Conv"
        if onnx_name in cut_dict["operator"]:
            ONNX_params = cut_dict["operator"][onnx_name]
            nn_module.groups -= ONNX_params["group"]
        in_dim = 1 if isinstance(nn_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) else 0
        nn_module.in_channels = nn_module.weight.data.size(in_dim) * nn_module.groups
        nn_module.out_channels = nn_module.weight.data.size(1 - in_dim)
        return nn_module, {**weight_context, **bias_context}

    # reconvery the model from the context
    def recovery_cut(self, nn_module, cut_dict, param_context):
        nn_module.weight = self.weight_pruner.recovery_cut(
            nn_module.weight, cut_dict, param_context
        )
        nn_module.bias = self.bias_pruner.recovery_cut(
            nn_module.bias, cut_dict, param_context
        )
        onnx_name = self.name + ".CONV"
        if onnx_name in cut_dict["operator"]:
            ONNX_params = cut_dict["operator"][onnx_name]
            nn_module.groups += ONNX_params["group"]
        in_dim = 1 if isinstance(nn_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) else 0
        nn_module.in_channels = nn_module.weight.data.size(in_dim) * nn_module.groups
        nn_module.out_channels = nn_module.weight.data.size(1 - in_dim)
        return nn_module


class BNPruner(BasePruner):
    def __init__(self, name):
        super(BNPruner, self).__init__(name)
        self.weight_pruner = ParameterPruner(name + ".weight")
        self.bias_pruner = ParameterPruner(name + ".bias")
        self.running_mean_pruner = TensorPruner(name + ".running_mean")
        self.running_var_pruner = TensorPruner(name + ".running_var")
        self.num_batches_tracked_pruner = TensorPruner(name + ".num_batches_tracked")

    # set the value to be zeros and return the context
    def set_zero(self, nn_module, cut_dict):
        nn_module.weight, weight_context = self.weight_pruner.set_zero(
            nn_module.weight, cut_dict
        )
        nn_module.bias, bias_context = self.bias_pruner.set_zero(
            nn_module.bias, cut_dict
        )
        (
            nn_module.running_mean,
            running_mean_context,
        ) = self.running_mean_pruner.set_zero(nn_module.running_mean, cut_dict)
        nn_module.running_var, running_var_context = self.running_var_pruner.set_zero(
            nn_module.running_var, cut_dict
        )
        return nn_module, {
            **weight_context,
            **bias_context,
            **running_mean_context,
            **running_var_context,
        }

    # recovery from zeros
    def recovery_zero(self, nn_module, cut_dict, context):
        nn_module.weight = self.weight_pruner.recovery_zero(
            nn_module.weight, cut_dict, context
        )
        nn_module.bias = self.bias_pruner.recovery_zero(
            nn_module.bias, cut_dict, context
        )
        nn_module.running_mean = self.running_mean_pruner.recovery_zero(
            nn_module.running_mean, cut_dict, context
        )
        nn_module.running_var = self.running_var_pruner.recovery_zero(
            nn_module.running_var, cut_dict, context
        )
        return nn_module

    # cut the value from zeros and return the context
    def set_cut(self, nn_module, cut_dict):
        nn_module.weight, weight_context = self.weight_pruner.set_cut(
            nn_module.weight, cut_dict
        )
        nn_module.bias, bias_context = self.bias_pruner.set_cut(
            nn_module.bias, cut_dict
        )
        nn_module.running_mean, running_mean_context = self.running_mean_pruner.set_cut(
            nn_module.running_mean, cut_dict
        )
        nn_module.running_var, running_var_context = self.running_var_pruner.set_cut(
            nn_module.running_var, cut_dict
        )
        nn_module.num_features = nn_module.bias.size(0)
        return nn_module, {
            **weight_context,
            **bias_context,
            **running_mean_context,
            **running_var_context,
        }

    # reconvery the model from the context
    def recovery_cut(self, nn_module, cut_dict, context):
        nn_module.weight = self.weight_pruner.recovery_cut(
            nn_module.weight, cut_dict, context
        )
        nn_module.bias = self.bias_pruner.recovery_cut(
            nn_module.bias, cut_dict, context
        )
        nn_module.running_mean = self.running_mean_pruner.recovery_cut(
            nn_module.running_mean, cut_dict, context
        )
        nn_module.running_var = self.running_var_pruner.recovery_cut(
            nn_module.running_var, cut_dict, context
        )
        nn_module.num_features = nn_module.bias.size(0)
        return nn_module


class LinearPruner(BasePruner):
    def __init__(self, name):
        super(LinearPruner, self).__init__(name)
        self.weight_pruner = ParameterPruner(name + ".weight")
        self.bias_pruner = ParameterPruner(name + ".bias")

    # set the value to be zeros and return the context
    def set_zero(self, nn_module, cut_dict):
        nn_module.weight, weight_context = self.weight_pruner.set_zero(
            nn_module.weight, cut_dict
        )
        nn_module.bias, bias_context = self.bias_pruner.set_zero(
            nn_module.bias, cut_dict
        )
        return nn_module, {**weight_context, **bias_context}

    # recovery from zeros
    def recovery_zero(self, nn_module, cut_dict, param_context):
        nn_module.weight = self.weight_pruner.recovery_zero(
            nn_module.weight, cut_dict, param_context
        )
        nn_module.bias = self.bias_pruner.recovery_zero(
            nn_module.bias, cut_dict, param_context
        )
        return nn_module

    # cut the value from zeros and return the context
    def set_cut(self, nn_module, cut_dict):
        nn_module.weight, weight_context = self.weight_pruner.set_cut(
            nn_module.weight, cut_dict
        )
        nn_module.bias, bias_context = self.bias_pruner.set_cut(
            nn_module.bias, cut_dict
        )
        nn_module.in_channels = nn_module.weight.data.size(1)
        nn_module.out_channels = nn_module.weight.data.size(0)
        return nn_module, {**weight_context, **bias_context}

    # reconvery the model from the context
    def recovery_cut(self, nn_module, cut_dict, param_context):
        nn_module.weight = self.weight_pruner.recovery_cut(
            nn_module.weight, cut_dict, param_context
        )
        nn_module.bias = self.bias_pruner.recovery_cut(
            nn_module.bias, cut_dict, param_context
        )
        nn_module.in_channels = nn_module.weight.data.size(1)
        nn_module.out_channels = nn_module.weight.data.size(0)
        return nn_module
