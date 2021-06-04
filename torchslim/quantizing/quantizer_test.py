import torch
import torchvision

import torch.quantization as q

from torch.onnx import OperatorExportTypes


def pre_hook(self, input):
    return self.input_observer(*input)


resnet = torchvision.models.resnet18()

resnet.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")

resnet_prepare = torch.quantization.prepare_qat(resnet)

resnet_prepare.add_module("input_observer", resnet_prepare.qconfig.activation())
resnet_prepare.register_forward_pre_hook(pre_hook)

resnet_quantized = torch.quantization.convert(resnet_prepare)

print(resnet_prepare)
print(resnet_quantized)

resnet_prepare.apply(torch.quantization.enable_observer)
resnet_prepare.apply(torch.quantization.enable_fake_quant)

for module in resnet_prepare.modules():
    if isinstance(module, torch.quantization.FakeQuantize):
        module.calculate_qparams()

resnet_prepare.apply(torch.quantization.disable_observer)

# resnet_int8=torch.quantization.convert(resnet_prepare)

# print(resnet_int8)

torch.onnx.symbolic_helper._set_opset_version(10)

# # torch.onnx.symbolic_opset10
# with torch.onnx.select_model_mode_for_export(resnet_prepare, None):
#     graph = torch.onnx.utils._trace(resnet_prepare,(torch.zeros(1,3,224,224),), OperatorExportTypes.ONNX)

graph, params_dict, torch_out = torch.onnx.utils._model_to_graph(
    resnet_prepare, (torch.zeros(1, 3, 224, 224),), _retain_param_name=True
)

print(graph)

torch.quantization.fuse_modules
