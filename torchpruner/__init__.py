import torchpruner.register
import torchpruner.mask_utils
import torchpruner.graph
import torchpruner.model_pruner
import torchpruner.init
import torchpruner.function_module

ONNXGraph = torchpruner.graph.ONNXGraph
set_zero = torchpruner.model_pruner.set_zero
set_cut = torchpruner.model_pruner.set_cut
recovery_zero = torchpruner.model_pruner.recovery_zero
recovery_cut = torchpruner.model_pruner.recovery_cut
merge_cut_dict = torchpruner.model_pruner.merge_cut_dict
create_mask = torchpruner.mask_utils.create_mask
set_cut_optimizer = torchpruner.model_pruner.set_cut_optimizer
recovery_optimizer = torchpruner.model_pruner.recovery_optimizer

module_pruner_reg = torchpruner.register.module_pruner_reg
operator_reg = torchpruner.register.operator_reg

# the function module tools
activate_function_module = torchpruner.function_module.activate_function_module
deactivate_function_module = torchpruner.function_module.deactivate_function_module
function_module_activate = torchpruner.function_module.function_module_activate
init_function_module = torchpruner.function_module.init_function_module

__all__ = [
    "ONNXGraph",
    "set_zero",
    "set_cut",
    "recovery_zero",
    "recovery_cut",
    "merge_cut_dict",
    "create_mask",
]
__all__ += ["cut_optimizer", "recovery_optimizer"]
__all__ += ["module_pruner_reg", "operator_reg"]
__all__ += [
    "activate_function_module",
    "deactivate_function_module",
    "function_module_activate",
    "init_function_module",
]
