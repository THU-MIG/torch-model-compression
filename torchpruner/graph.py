# -*- coding: utf-8 -*-
import torch
import torch.onnx
import torch.onnx.symbolic_helper
import torch.onnx.utils
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from . import register
from . import mask_utils
from . import function_module

from typing import Dict, List, Set

from torchpruner.operator import operator as operator_module
from torchpruner.model_tools import *

import copy


def create_operator(node):
    op_kind = node.kind().split("::")
    type_name = op_kind[1]
    operator_class = register.operator_reg.get(type_name)
    if operator_class is None:
        raise RuntimeError("Can not find operator " + str(type_name))
    return operator_class(node)


max_search_time = 100000


def _operator_params_combine(params_list):
    params_dict = {}
    for params in params_list:
        for key in params:
            if not isinstance(params[key], int):
                raise RuntimeError("The change params must be integer")
            if key not in params_dict:
                params_dict[key] = params[key]
            else:
                params_dict[key] += params[key]
    return params_dict


def _get_all_modules(module_dict, prefix, names):
    module_list = []
    for name in names:
        prefix = prefix + "." + name
        module_list.append(module_dict[prefix])
    return module_list


def _get_common(scope1, scope2):
    prefix = ""
    i = 0
    while i < len(scope1) and i < len(scope2):
        if scope1[i] == scope2[i]:
            if prefix == "":
                prefix += scope1[i]
            else:
                prefix = prefix + "." + scope1[i]
            i += 1
        else:
            break
    list1 = []
    list2 = []
    for j in range(i, len(scope1)):
        list1.append(scope1[j])
    for j in range(i, len(scope2)):
        list2.append(scope2[j])
    return prefix, list1, list2


def _cat_names(names):
    name = names[0]
    for i in range(1, len(names)):
        name = name + "." + names[i]
    return name


def _find_module_list(module_list, target):
    keys = None
    if isinstance(module_list, nn.ModuleList):
        keys = range(0, len(module_list))
    if isinstance(module_list, nn.ModuleDict):
        keys = module_list.keys()
    for key in keys:
        if module_list[key] is target:
            return [target._get_name(), str(key)]
        if isinstance(module_list[key], (nn.ModuleList, nn.ModuleDict)):
            return [module_list[key]._get_name(), str(key)] + _find_module_list(
                module_list[key], target
            )
    return None


def _get_object_to_name_dict(model):
    to_name_dict = {}
    stack = []
    stack.append([model, "self"])
    while len(stack) != 0:
        obj, name = stack.pop()
        if isinstance(obj, nn.Module):
            to_name_dict[id(obj)] = name
        for key in obj._modules.keys():
            stack.append([obj._modules[key], name + "." + key])
    return to_name_dict


class scope_name_workaround(object):
    def __init__(self, model):
        self.backup = None
        self.to_name_dict = _get_object_to_name_dict(model)
        self.scope_stack = []

    def __enter__(self):
        def _tracing_name(self_, tracing_state):
            if not tracing_state._traced_module_stack:
                return None
            module = tracing_state._traced_module_stack[-1]
            for name, child in module.named_children():
                if child is self_:
                    return name
                if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
                    search_result = _find_module_list(child, self_)
                    if search_result is not None:
                        search_result = [child._get_name(), name] + search_result
                        return search_result
            return None

        def _slow_forward(self_, *input, **kwargs):
            tracing_state = torch._C._get_tracing_state()
            if not tracing_state or isinstance(self_.forward, torch._C.ScriptMethod):
                return self_.forward(*input, **kwargs)
            if tracing_state.current_scope() != "":
                self.scope_stack.append(tracing_state.current_scope())
                tracing_state.pop_scope()
            if id(self_) in self.to_name_dict:
                tracing_state.push_scope(self.to_name_dict[id(self_)])
            try:
                result = self_.forward(*input, **kwargs)
            finally:
                if tracing_state.current_scope() != "":
                    tracing_state.pop_scope()
                if len(self.scope_stack) != 0:
                    tracing_state.push_scope(self.scope_stack[-1])
                    self.scope_stack.pop()
            return result

        self.backup = torch.nn.Module._slow_forward
        setattr(torch.nn.Module, "_slow_forward", _slow_forward)

    def __exit__(self, type, value, tb):
        setattr(torch.nn.Module, "_slow_forward", self.backup)


# DataNode
class DataNode(object):
    def __init__(self, node):
        # basic info
        self.name = "self." + node.debugName()
        self._type = None
        self._size = None
        self.kind = str(node.type().kind())
        if self.kind == "TensorType" or self.kind == "CompleteTensorType":
            sizes = node.type().sizes()
            if sizes is not None:
                self._size = list(sizes)
                self._type = str(node.type().scalarType())
        self._is_terminal = False
        self._is_input = False
        self._is_output = False
        # operator related
        self.in_operator: operator_module.OperatorNode = None
        self.out_operators: List[operator_module.OperatorNode] = []
        # data add with the hook
        self.data = None
        # add a key value changeable
        self._changeable = True
        # set the graph
        self.graph: ONNXGraph = None

    def get(self, indexs, dim):
        dict_tuple = []
        for _ in range(0, dim):
            dict_tuple.append(slice(None, None, None))
        dict_tuple.append(indexs)
        return self.data[tuple(dict_tuple)]

    def __str__(self):
        return_str = "%" + self.name + ": "
        if self._type is None:
            return return_str + "Unknown()"
        return_str += self._type
        return_str += "("
        if len(self._size) == 0:
            return_str += ")"
            return return_str
        for s in self._size:
            return_str += str(s)
            return_str += ", "
        return_str = return_str[:-2]
        return_str += ")"
        return return_str

    def __repr__(self):
        return self.__str__()

    def is_terminal(self):
        return self._is_terminal

    def is_input(self):
        return self._is_input

    def is_output(self):
        return self._is_output

    def is_changeable(self):
        return self._changeable

    def size(self, dim=None):
        if dim is None:
            return self._size
        if dim >= len(self._size):
            raise RuntimeError("the dim out of index")
        return self._size[dim]

    def type(self):
        return self._type

    def __len__(self):
        if self._size is None:
            return 0
        else:
            if len(self._size) == 0:
                return 0
            return self._size[0]

    def cut_analysis(self, index, dim):
        mask = mask_utils.Mask(self._size)
        if not isinstance(index, (list, np.ndarray)):
            raise RuntimeError("The index must be a list or a ndarray")
        mask.set_mask([index], [dim])
        return self.cut_analysis_with_mask(mask)

    def cut_analysis_with_mask(self, mask):
        times = 0
        mask_dict = OrderedDict()
        mask_dict[self.name] = mask
        operator_dict = OrderedDict()
        stack = []
        stack.append((self, mask, None))
        while len(stack) != 0:
            node, mask, push_operator = stack.pop()
            operators = node.out_operators
            operators = operators[:]
            if node.in_operator is not None:
                operators.append(node.in_operator)
            # remove the push_opeartion
            if push_operator is not None:
                for i in range(0, len(operators)):
                    if id(operators[i]) == id(push_operator):
                        del operators[i]
                        break
            # run analysis for operator
            for operator in operators:
                return_masks, operator_params = operator.analysis(node, mask)
                # handle operator_dict
                if operator_params is not None:
                    if operator.name not in operator_dict:
                        operator_dict[operator.name] = [operator_params]
                    else:
                        operator_dict[operator.name].append(operator_params)
                # handle return_dict
                for name in return_masks.keys():
                    return_node = self.graph.nodes[name]
                    if name in mask_dict.keys():
                        if mask_dict[name].include(return_masks[name]):
                            continue
                        mask_dict[name] = mask_utils.combine_mask(
                            [mask_dict[name], return_masks[name]]
                        )
                    else:
                        mask_dict[name] = return_masks[name]
                    # push stack
                    stack.append((return_node, return_masks[name].copy(), operator))
                times += 1
                if times >= max_search_time:
                    raise RuntimeError("max search time exceed")

        conbine_dict = {}
        conbine_dict["terminal"] = {}
        conbine_dict["iner"] = {}
        conbine_dict["operator"] = {}

        for key in mask_dict.keys():
            node = self.graph.nodes[key]
            result = mask_dict[key].indexs()
            if not node.is_terminal():
                conbine_dict["iner"][key] = result
            else:
                conbine_dict["terminal"][key] = result
        for key in operator_dict.keys():
            conbine_dict["operator"][key] = _operator_params_combine(operator_dict[key])
        return conbine_dict


# the module class
class Module(object):
    def __init__(self):
        self.name = ""
        self.sub_modules: Dict[str, Module] = OrderedDict()  # store the sub modules
        self.in_data: List[DataNode] = []  # the data may be used different times
        self.out_data: List[DataNode] = []  # the data may produced different times
        self.operators: List[operator_module.OperatorNode] = []  # save the opeartor in current module
        self.nn_object: nn.Module = None  # bounding the actual object
        self.terminal_node: DataNode = None

    def cut_analysis(self, attribute_name, index, dim):
        attrs = attribute_name.split(".")
        current_module = self
        for attr in attrs:
            if attr in current_module.sub_modules:
                current_module = current_module.sub_modules[attr]
            else:
                raise RuntimeError("Can not find attribute " + str(attribute_name))
        if current_module.terminal_node is None:
            raise RuntimeError("The target attribute is not cuttable")
        return current_module.terminal_node.cut_analysis(index, dim)

    def __str__(self):
        return_string = ""
        class_string = str(self.nn_object.__class__)[8:-2]
        return_string += class_string
        return_string += "["
        if self.terminal_node is not None:
            return_string += "self."
            terminal_string = str(getattr(self, "terminal_node"))[1:]
            split_string = terminal_string.split(":")
            return_string += split_string[0]
            return_string += "]:"
            return_string += split_string[1][1:]
        else:
            return_string += self.name
            return_string += "]"
        return return_string

    def __repr__(self):
        return self.__str__()


class ONNXGraph(object):
    def __init__(self, model, onnx_device="CPU"):
        if isinstance(
            model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
        ):
            print(
                "WARNING: The model is warped with the DataParallel, the Graph object just binding the model.module part"
            )
            self._model: nn.Module = model.module
        else:
            self._model: nn.Module = model
        self.modules: Dict[str, Module] = OrderedDict()
        self.inputs: Dict[str, DataNode] = OrderedDict()
        self.nodes: Dict[str, DataNode] = OrderedDict()
        self.outputs: Dict[str, DataNode] = OrderedDict()
        self.operators: Dict[str, operator_module.OperatorNode] = OrderedDict()
        self._device: str = onnx_device

    def __str__(self):
        out_string = ""
        for node in self.nodes.keys():
            out_string += str(self.nodes[node])
            out_string += "\n"
        for operator in self.operators.keys():
            out_string += str(self.operators[operator])
            out_string += "\n"
        return out_string

    def forward(self, inputs):
        # change the inputs
        if len(self.inputs.keys()) != len(inputs):
            raise RuntimeError(
                "The inputs numbers is wrong expected "
                + str(self.inputs.keys())
                + " but got "
                + str(len(inputs))
            )
        input_keys = list(self.inputs.keys())
        for i in range(0, len(inputs)):
            if list(inputs[i].size()) != list(self.inputs[input_keys[i]].size()):
                raise RuntimeError(
                    "The inputs must as the same size as the origin input"
                )
            self.inputs[input_keys[i]].data = inputs[i].numpy()
        for operator in self.operators:
            self.operators[operator].fill_shape()
            self.operators[operator].fill_value()

    def set_device(self, device):
        self._device = device
        for operator in self.operators:
            self.operators[operator].set_device(device)

    def get_device(self):
        return self._device

    def flops(self):
        total_flops = 0
        for operator in self.operators:
            total_flops += self.operators[operator].flops()
        return total_flops / 1000000

    def get_module_by_object(self, obj):
        for module_name in self.modules:
            c_module = self.modules[module_name]
            if id(c_module.nn_object) == id(obj):
                return c_module
        return None

    def build_graph(self, inputs, fill_value=True, training=False):
        # prepare the data structure
        data_node_dict: Dict[str, DataNode] = OrderedDict()
        # input node dict
        input_node_dict: Dict[str, DataNode] = OrderedDict()
        # terminal_node_dict
        terminal_node_dict: Dict[str, DataNode] = OrderedDict()
        # output node dict
        output_node_dict: Dict[str, DataNode] = OrderedDict()
        # operator list
        operator_dict: Dict[str, operator_module.OperatorNode] = OrderedDict()
        # module dict
        module_dict: Dict[str, Module] = OrderedDict()

        # check the function_module
        if function_module.function_module_activate() and not hasattr(
            self._model, "_init_function_module_ok"
        ):
            raise RuntimeError("Call the init_function_module in function_module mode")

        # deep copy the model
        model = copy.deepcopy(self._model)
        model = model.cpu()

        # preprocess the quantization node
        for module in model.modules():
            if isinstance(module, torch.quantization.FakeQuantize):
                module.calculate_qparams()
        model.apply(torch.quantization.disable_observer)

        with scope_name_workaround(model):
            torch.onnx.symbolic_helper._set_opset_version(11)
            graph, params_dict, torch_out = torch.onnx.utils._model_to_graph(
                model,
                inputs,
                **normalize_onnx_parameters(
                    _retain_param_name=True,
                    do_constant_folding=False,
                    training=training,
                )
            )
            torch.onnx.symbolic_helper._set_opset_version(9)
        # create the inputs and the terminals
        inputs_number = len(inputs)
        input_nodes = list(graph.inputs())
        total_number = len(input_nodes)

        for i in range(0, total_number):
            data_node = DataNode(input_nodes[i])
            if i < inputs_number:
                data_node._is_input = True
                data_node._changeable = False
                data_node.data = inputs[i].numpy()
                input_node_dict[data_node.name] = data_node
            else:
                data_node._is_terminal = True
                data_node.data = params_dict[
                    ".".join(data_node.name.split(".")[1:])
                ].numpy()
                terminal_node_dict[data_node.name] = data_node
            data_node_dict[data_node.name] = data_node

        # create the iner node and the operator node
        body_nodes = list(graph.nodes())
        for i in range(0, len(body_nodes)):
            # create the operator node
            node = body_nodes[i]
            operator_node = create_operator(node)
            operator_node.set_device(self._device)
            # create the outputs node
            outputs = list(node.outputs())
            for out_node in outputs:
                data_node = DataNode(out_node)
                data_node.in_operator = operator_node
                data_node_dict[data_node.name] = data_node
                operator_node.out_data.append(data_node)
            # link the inputs node
            inputs = list(node.inputs())
            for in_node in inputs:
                in_node_name = "self." + in_node.debugName()
                data_node = data_node_dict[in_node_name]
                operator_node.in_data.append(data_node)
                data_node.out_operators.append(operator_node)
            operator_dict[str(i)] = operator_node

        # if the data node is the output, set the changeable to be false, set the is output to be true
        outputs = list(node.outputs())
        for out_node in outputs:
            out_node_name = "self." + out_node.debugName()
            data_node = data_node_dict[out_node_name]
            data_node._changeable = False
            data_node._is_output = True
            output_node_dict[out_node_name] = data_node

        # binding the graph to node
        for key in data_node_dict.keys():
            data_node_dict[key].graph = self

        # create the module
        for key in operator_dict:
            operator = operator_dict[key]
            obj_list = operator.obj_list
            current = ""
            parent = None
            for i in range(0, len(obj_list)):
                name = obj_list[i]
                if current == "":
                    current = name
                else:
                    current = current + "." + name
                actual_obj = get_object(self._model, current)
                if current not in module_dict.keys():
                    module_dict[current] = Module()
                    module_dict[current].name = current
                    module_dict[current].graph = graph
                    module_dict[current].nn_object = actual_obj
                if parent is not None:
                    parent.sub_modules[name] = module_dict[current]
                parent = module_dict[current]
                if i == len(obj_list) - 1:
                    module_dict[current].operators.append(operator)

        # add the terminal node
        for node_name in terminal_node_dict.keys():
            node = terminal_node_dict[node_name]
            obj_names = node_name.split(".")
            if len(obj_names) == 2 and intable(obj_names[1]):
                continue
            current = "self"
            parent = None
            for i in range(1, len(obj_names)):
                name = obj_names[i]
                current = current + "." + name
                actual_obj = get_object(self._model, current)
                if current not in module_dict.keys():
                    if i == len(obj_names) - 1:
                        if not isinstance(actual_obj, (nn.Parameter, torch.Tensor)):
                            raise RuntimeError(
                                "The terminal node must be the nn.Parameter or torch.Tensor"
                            )
                    module_dict[current] = Module()
                    module_dict[current].terminal_node = node
                    module_dict[current].name = current
                    module_dict[current].graph = graph
                    module_dict[current].nn_object = actual_obj
                    module_dict[current].nn_type = type(actual_obj)
                if parent is not None:
                    parent.sub_modules[name] = module_dict[current]
                parent = module_dict[current]

        # bind the in_data and out_data for modules
        for node_name in data_node_dict.keys():
            node = data_node_dict[node_name]
            if node.is_terminal() and not node.is_input():
                continue
            if node.is_input():
                out_operators = node.out_operators
                for operator in out_operators:
                    obj_names = operator.obj_list[1:]
                    prefix = "self"
                    modules_list = _get_all_modules(module_dict, prefix, obj_names)
                    for module in modules_list:
                        if node not in module.in_data:
                            module.in_data.append(node)
                continue
            in_operator = node.in_operator
            in_scope = in_operator.obj_list
            out_operators = node.out_operators[:]
            if not node.is_output() and len(out_operators) == 0:
                module_name = _cat_names(in_operator.obj_list)
                module_dict[module_name].out_data.append(node)
                continue
            output_scope_list = []
            for out_operator in out_operators:
                output_scope_list.append(out_operator.obj_list)
            if node.is_output:
                output_scope_list.append(["self"])
            for scope in output_scope_list:
                prefix, in_scope_names, out_scope_names = _get_common(in_scope, scope)
                in_modules_list = _get_all_modules(module_dict, prefix, in_scope_names)
                for module in in_modules_list:
                    if node not in module.out_data:
                        module.out_data.append(node)
                out_modules_list = _get_all_modules(
                    module_dict, prefix, out_scope_names
                )
                for module in out_modules_list:
                    if node not in module.in_data:
                        module.in_data.append(node)
        self.nodes = data_node_dict
        self.inputs = input_node_dict
        self.outputs = output_node_dict
        self.modules = module_dict
        self.operators = operator_dict
        # fille the data and value
        if fill_value:
            visited: Set[operator_module.OperatorNode] = set()
            to_check_tensors: Set[DataNode] = set()
            for operator in operator_dict.values():
                operator.fill_shape()
                operator.fill_value()
                visited.add(operator)
                for to_check in tuple(to_check_tensors):
                    if not (set(i for i in to_check.out_operators) - visited):
                        to_check_tensors.remove(to_check)
                        to_check.data = None
                to_check_tensors.update(operator.out_data)
            for to_check in to_check_tensors:
                to_check.data = None
        else:
            for operator in operator_dict.values():
                operator.fill_shape()
        import gc; gc.collect()
