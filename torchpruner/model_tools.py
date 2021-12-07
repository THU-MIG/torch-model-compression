import collections
import torch
import copy
import torch.nn as nn


def intable(value):
    try:
        int(value)
        return True
    except:
        return False


# change with _module
def get_object(model, name):
    if isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    ):
        model = model.module
    if name == "self":
        return model
    attributes = name.split(".")
    if len(attributes) == 1:
        raise RuntimeError("Can not find the " + name + " in model")
    current = model
    for i in range(1, len(attributes)):
        name = attributes[i]
        try:
            if intable(name):
                current = current[int(name)]
            else:
                current = getattr(current, name)
        except Exception as e:
            raise RuntimeError("Can not find the " + name + " in model")
    return current


def is_type(object, object_class):
    if isinstance(object_class, list):
        if type(object) in object_class:
            return True
    if type(object) == object_class:
        return True
    return False


def set_object(model, name, nn_module):
    base_ptr = model
    if isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    ):
        model = model.module
    if name == "self":
        return nn_module
    attributes = name.split(".")
    prefix = ".".join(attributes[:-1])
    prefix_object = get_object(model, prefix)
    setattr(prefix_object, attributes[-1], nn_module)
    return base_ptr


def get_exclusion_names(graph, object_class, attribute_name, attribute_dim):
    modules = graph.get_modules()
    exclusion_dict = collections.OrderedDict()
    for _, module in modules.items():
        if isinstance(module.nn_object, object_class):
            cut_dict = module.cut_analysis(attribute_name, [0], attribute_dim)
            terminal_names = cut_dict["terminal"]
            terminal_names = list(terminal_names)
            terminal_names.sort()
            exclusion_key = "_".join(terminal_names)
            exclusion_dict[exclusion_key] = module.name
    exclusion_list = []
    for key in exclusion_dict.keys():
        exclusion_list.append(exclusion_dict[key])
    return exclusion_list


# change setting with _modules
def get_names_by_class(model, object_class, include_super_class=True):
    name_list = []
    stack = []
    stack.append([model, "self"])
    while len(stack) != 0:
        pop_object, object_name = stack.pop()
        if include_super_class and isinstance(pop_object, object_class):
            name_list.append(object_name)
            continue
        if is_type(pop_object, object_class):
            name_list.append(object_name)
            continue
        # if the nn.Module is a Sequential or ModuleList
        if isinstance(pop_object, (torch.nn.Sequential, torch.nn.ModuleList)):
            for index in range(0, len(pop_object)):
                if isinstance(pop_object[index], torch.nn.Module):
                    stack.append([pop_object[index], object_name + "." + str(index)])
        if isinstance(pop_object, torch.nn.ModuleDict):
            for key in pop_object.keys():
                if isinstance(pop_object[key], torch.nn.Module):
                    stack.append([pop_object[key], object_name + "." + str(key)])
        attributes = dir(pop_object)
        for attribute in attributes:
            sub_object = getattr(pop_object, attribute)
            if isinstance(sub_object, torch.nn.Module):
                stack.append([sub_object, object_name + "." + attribute])
    return name_list


def _find_match_module(
    modules,
    prev_node,
    object_class_list,
    current_name_list,
    group_name_list,
    include_super_class,
):
    if len(current_name_list) == len(object_class_list):
        group_name_list.append(copy.deepcopy(current_name_list))
        return
    for _, module in modules.items():
        if (
            include_super_class
            and isinstance(module.nn_object, object_class_list[len(current_name_list)])
        ) or is_type(module.nn_object, object_class_list[len(current_name_list)]):
            if prev_node is None or prev_node in module.in_data:
                if len(module.out_data) != 0:
                    current_name_list.append(module.name)
                    _find_match_module(
                        modules,
                        module.out_data[0],
                        object_class_list,
                        current_name_list,
                        group_name_list,
                        include_super_class,
                    )
                    current_name_list.pop()


def get_name_groups_by_classes(graph, object_class_list, include_super_class=True):
    modules = graph.modules
    group_name_list = []
    current_name_list = []
    _find_match_module(
        modules,
        None,
        object_class_list,
        current_name_list,
        group_name_list,
        include_super_class,
    )
    return group_name_list


def replace_object_by_name_groups(model, group_name_list, replace_function):
    for group_name in group_name_list:
        objs = []
        for name in group_name:
            obj = get_object(model, name)
            objs.append(obj)
        new_objs = replace_function(group_name, objs)
        for name, obj in zip(group_name, new_objs):
            model = set_object(model, name, obj)
    return model


def replace_object_by_names(model, name_list, replace_function):
    for name in name_list:
        obj = get_object(model, name)
        model = set_object(model, name, replace_function(name, obj))
    return model


def replace_object_by_class(
    model, object_class, replace_function, include_super_class=True
):
    name_list = get_names_by_class(model, object_class, include_super_class)
    return replace_object_by_names(model, name_list, replace_function)


def normalize_onnx_parameters(**kwargs):
    torch_version = torch.__version__.split(".")
    if torch_version[0] > "2" or len(torch_version) > 1 and torch_version[1] >= "10":
        kwargs.pop("_retain_param_name", None)
    return kwargs
