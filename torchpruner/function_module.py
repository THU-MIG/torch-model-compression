import torch
import torch.nn as nn
import torch.nn.functional as F


class TraceState(object):
    def __init__(self):
        self._traced_module_stack = []
        self._traced_module_number = []


_trace_state = TraceState()

_call_impl_backup = torch.nn.Module._call_impl
_relu_backup = F.relu
_add_backup = torch.add
_mul_backup = torch.mul
_cat_backup = torch.cat
_activate = False

# The function nn Module
class FunctionModule(nn.Module):
    def __init__(self):
        super(FunctionModule, self).__init__()


class Relu(FunctionModule):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, *args, **kwargs):
        return _relu_backup(*args, **kwargs)


class Add(FunctionModule):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, *args, **kwargs):
        return _add_backup(*args, **kwargs)


class Cat(FunctionModule):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, *args, **kwargs):
        return _cat_backup(*args, **kwargs)


class Mul(FunctionModule):
    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, *args, **kwargs):
        return _mul_backup(*args, **kwargs)


_type_module_mapping = {"Add": Add, "Relu": Relu, "Cat": Cat, "Mul": Mul}

_type_back_up_mapping = {
    "Add": _add_backup,
    "Relu": _relu_backup,
    "Cat": _cat_backup,
    "Mul": _mul_backup,
}


def _call_impl(self_, *input, **kwargs):
    global _trace_state
    _trace_state._traced_module_stack.append(self_)
    _trace_state._traced_module_number.append(0)
    try:
        result = _call_impl_backup(self_, *input, **kwargs)
    finally:
        _trace_state._traced_module_stack.pop()
        _trace_state._traced_module_number.pop()
    return result


def _get_function_module(module_type):
    global _trace_state
    if len(_trace_state._traced_module_stack) == 0:
        return _type_back_up_mapping[module_type]
    traced_module = _trace_state._traced_module_stack[-1]
    if not hasattr(traced_module, "_function_module_list"):
        traced_module.add_module("_function_module_list", torch.nn.ModuleList())
    module_number = _trace_state._traced_module_number[-1]
    function_module_list = traced_module._function_module_list
    if module_number >= len(function_module_list):
        function_module_list.append(_type_module_mapping[module_type]())
    function_module = function_module_list[module_number]
    _trace_state._traced_module_number[-1] += 1
    return function_module


def relu_function(*args, **kwargs):
    function_module = _get_function_module("Relu")
    return function_module(*args, **kwargs)


def add_function(*args, **kwargs):
    function_module = _get_function_module("Add")
    return function_module(*args, **kwargs)


def cat_function(*args, **kwargs):
    function_module = _get_function_module("Cat")
    return function_module(*args, **kwargs)


def mul_function(*args, **kwargs):
    function_module = _get_function_module("Mul")
    return function_module(*args, **kwargs)


def function_module_activate():
    global _activate
    return _activate


def activate_function_module():
    global _activate
    F.relu = relu_function
    torch.add = add_function
    torch.cat = cat_function
    torch.mul = mul_function
    setattr(torch.nn.Module, "__call__", _call_impl)
    _activate = True


def deactivate_function_module():
    global _activate
    F.relu = _relu_backup
    torch.add = _add_backup
    torch.cat = _cat_backup
    torch.mul = _mul_backup
    setattr(torch.nn.Module, "__call__", _call_impl_backup)
    _activate = False


def init_function_module(model, inputs):
    _ = model(*inputs)
    setattr(model, "_init_function_module_ok", True)
    return model
