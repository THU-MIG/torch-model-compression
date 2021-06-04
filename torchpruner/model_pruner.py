# set the location to 0 given the cut_dict and return the context
from torchpruner.model_tools import *
from torchpruner.register import module_pruner_reg
import copy


def merge_cut_dict(cut_dict_list):
    new_cut_dict = {}
    new_cut_dict["terminal"] = {}
    new_cut_dict["iner"] = {}
    new_cut_dict["operator"] = {}
    scopes = ["terminal", "iner", "operator"]
    for cut_dict in cut_dict_list:
        for scope in scopes:
            for key in cut_dict[scope]:
                if key not in new_cut_dict[scope].keys():
                    new_cut_dict[scope][key] = cut_dict[scope][key]
                else:
                    if new_cut_dict[scope][key] != cut_dict[scope][key]:
                        raise RuntimeError("merge confiction on " + key)
    return new_cut_dict


def set_zero(model, cut_dict):
    keys = cut_dict["terminal"].keys()
    visited = []
    param_context = {}
    for key in keys:
        names = key.split(".")
        prefix = ""
        for name in names:
            if prefix == "":
                prefix = name
            else:
                prefix = prefix + "." + name
            if prefix in visited:
                break
            nn_module = get_object(model, prefix)
            pruner_class = module_pruner_reg.get(type(nn_module))
            if pruner_class is None:
                continue
            module_pruner = pruner_class(prefix)
            nn_module, context = module_pruner.set_zero(nn_module, cut_dict)
            param_context.update(context)
            model = set_object(model, prefix, nn_module)
            break
    return model, param_context


def set_cut(model, cut_dict):
    keys = cut_dict["terminal"].keys()
    visited = []
    param_context = {}
    for key in keys:
        names = key.split(".")
        prefix = ""

        for name in names:
            if prefix == "":
                prefix = name
            else:
                prefix = prefix + "." + name

            if prefix in visited:
                break
            nn_module = get_object(model, prefix)
            pruner_class = module_pruner_reg.get(type(nn_module))
            if pruner_class is None:
                continue
            module_pruner = pruner_class(prefix)
            nn_module, context = module_pruner.set_cut(nn_module, cut_dict)
            param_context.update(context)
            model = set_object(model, prefix, nn_module)
            visited.append(prefix)
            break
    return model, param_context


def recovery_zero(model, cut_dict, param_context):
    keys = cut_dict["terminal"].keys()
    visited = []
    for key in keys:
        names = key.split(".")
        prefix = ""
        for name in names:
            if prefix == "":
                prefix = name
            else:
                prefix = prefix + "." + name
            if prefix in visited:
                break
            nn_module = get_object(model, prefix)
            pruner_class = module_pruner_reg.get(type(nn_module))
            if pruner_class is None:
                continue
            module_pruner = pruner_class(prefix)
            nn_module = module_pruner.recovery_zero(nn_module, cut_dict, param_context)
            model = set_object(model, prefix, nn_module)
            break
    return model


def recovery_cut(model, cut_dict, param_context):
    keys = cut_dict["terminal"].keys()
    visited = []
    for key in keys:
        names = key.split(".")
        prefix = ""
        for name in names:
            if prefix == "":
                prefix = name
            else:
                prefix = prefix + "." + name
            if prefix in visited:
                break
            nn_module = get_object(model, prefix)
            pruner_class = module_pruner_reg.get(type(nn_module))
            if pruner_class is None:
                continue
            module_pruner = pruner_class(prefix)
            nn_module = module_pruner.recovery_cut(nn_module, cut_dict, param_context)
            model = set_object(model, prefix, nn_module)
            break
    return model


def set_cut_optimizer(model, optimizer, cut_dict):
    keys = cut_dict["terminal"].keys()
    param_context = {}
    buffer_keyward = [
        "momentum_buffer",
        "square_avg",
        "acc_delta",
        "exp_avg",
        "exp_avg_sq",
        "exp_inf" "max_exp_avg_sq",
        "ax",
        "square_avg" "grad_avg",
        "sum",
    ]

    for key in keys:
        param = get_object(model, key)
        find_key = False
        tensor_class = module_pruner_reg.get(torch.Tensor)
        tensor_pruner = tensor_class(key)
        for group in optimizer.param_groups:
            if find_key:
                break
            for p in group["params"]:
                if id(p) == id(param):
                    find_key = True
                    param_state = optimizer.state[p]
                    for buf_key in buffer_keyward:
                        if buf_key in param_state:
                            param_state[buf_key], context = tensor_pruner.set_cut(
                                param_state[buf_key], cut_dict
                            )
                            param_context.update(context)
                    break
    return optimizer, param_context


def recovery_optimizer(model, optimizer, cut_dict, param_context):
    keys = cut_dict["terminal"].keys()
    buffer_keyward = [
        "momentum_buffer",
        "square_avg",
        "acc_delta",
        "exp_avg",
        "exp_avg_sq",
        "exp_inf" "max_exp_avg_sq",
        "ax",
        "square_avg" "grad_avg",
        "sum",
    ]

    for key in keys:
        param = get_object(model, key)
        find_key = False
        tensor_class = module_pruner_reg.get(torch.Tensor)
        tensor_pruner = tensor_class(key)
        for group in optimizer.param_groups:
            if find_key:
                break
            for p in group["params"]:
                if id(p) == id(param):
                    find_key = True
                    if p.grad is None:
                        break
                    param_state = optimizer.state[p]
                    for buf_key in buffer_keyward:
                        if buf_key in param_state:
                            param_state[buf_key] = tensor_pruner.recovery_cut(
                                param_state[buf_key], cut_dict, param_context
                            )
                    break
    return optimizer
