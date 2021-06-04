import sys

sys.path.append("../..")
import torchpruner
import torchpruner.model_tools as model_tools
import torchvision

import torch

import torch.nn as nn

# model creator,  the input shape, prune_modules
models = {
    "fcn_resnet50": (
        torchvision.models.segmentation.fcn_resnet50,
        (1, 3, 224, 224),
        [
            ("self.backbone.layer2.3.bn3", "weight"),
            ("self.backbone.layer4.0.downsample.0", "weight"),
            ("self.classifier.0", "weight"),
        ],
    ),
    "deeplabv3_resnet50": (
        torchvision.models.segmentation.deeplabv3_resnet50,
        (2, 3, 224, 224),
        [
            ("self.backbone.layer4.0.downsample.0", "weight"),
            ("self.classifier.0.convs.0.0", "weight"),
            ("self.classifier.0.convs.4.1", "weight"),
            ("self.classifier.0.project.1", "bias"),
        ],
    ),
}


for model_key in models.keys():
    print("Testing " + model_key + " ....")
    model_function, input_size, prune_modules = models[model_key]
    input_tensor = (torch.zeros(input_size),)
    for prune_module, attribute in prune_modules:
        print("The prune module is:" + str(prune_module))
        model = model_function()
        graph = torchpruner.ONNXGraph(model)
        graph.build_graph(input_tensor)
        module = graph.modules[prune_module]
        result = module.cut_analysis(attribute, index=[0, 1, 2, 3], dim=0)
    print("End Testing....")
