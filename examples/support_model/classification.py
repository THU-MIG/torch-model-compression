import sys

sys.path.append("../..")
import torchpruner
import torchpruner.model_tools as model_tools
import torchvision

import torch

import torch.nn as nn

# model creator,  the input shape, prune_modules
models = {
    "alexnet": (
        torchvision.models.alexnet,
        (1, 3, 224, 224),
        [
            ("self.features.0", "weight"),
            ("self.features.6", "bias"),
            ("self.classifier.1", "weight"),
        ],
    ),
    "vgg13": (
        torchvision.models.vgg13_bn,
        (1, 3, 224, 224),
        [
            ("self.features.7", "weight"),
            ("self.features.32", "bias"),
            ("self.classifier.3", "weight"),
        ],
    ),
    "resnet18": (
        torchvision.models.resnet18,
        (1, 3, 224, 224),
        [("self.layer1.0.conv1", "weight"), ("self.layer3.1.bn2", "bias")],
    ),
    "resnext50": (
        torchvision.models.resnext50_32x4d,
        (1, 3, 224, 224),
        [
            ("self.layer1.0.downsample.0", "weight"),
            ("self.layer2.2.bn2", "weight"),
            ("self.fc", "weight"),
        ],
    ),
    "resnet50": (
        torchvision.models.resnet50,
        (1, 3, 224, 224),
        [("self.layer1.0.conv2", "weight"), ("self.layer3.1.bn2", "bias")],
    ),
    "mobilenet": (
        torchvision.models.mobilenet_v2,
        (1, 3, 224, 224),
        [
            ("self.features.0.1", "weight"),
            ("self.features.1.conv.0.0", "weight"),
            ("self.features.5.conv.3", "weight"),
        ],
    ),
    "shufflenet": (
        torchvision.models.shufflenet_v2_x1_0,
        (1, 3, 224, 224),
        [
            ("self.conv1.0", "weight"),
            ("self.stage2.0.branch1.1", "weight"),
            ("self.stage3.0.branch2.3", "weight"),
            ("self.conv5.0", "weight"),
        ],
    ),
    "inception": (
        torchvision.models.inception_v3,
        (2, 3, 299, 299),
        [
            ("self.Conv2d_2a_3x3.conv", "weight"),
            ("self.Mixed_5b.branch5x5_1.bn", "weight"),
            ("self.Mixed_6d.branch7x7_2.conv", "weight"),
        ],
    ),
    "mnasnet": (
        torchvision.models.mnasnet1_0,
        (1, 3, 224, 224),
        [
            ("self.layers.4", "bias"),
            ("self.layers.8.1.layers.6", "weight"),
            ("self.layers.12.0.layers.3", "weight"),
        ],
    ),
    "densenet": (
        torchvision.models.densenet121,
        (1, 3, 224, 224),
        [
            ("self.features.denseblock1.denselayer1.conv2", "weight"),
            ("self.features.transition2.conv", "weight"),
        ],
    ),
    # "googlenet":(torchvision.models.googlenet,(1,3,224,224),
    # [('self.conv1.conv','weight'),('self.inception3a.branch2.1.bn','weight'),('self.inception4d.branch1.bn','weight')]), #torchvision googlenet unsupport for onnx export error
    "squeezenet": (
        torchvision.models.squeezenet1_0,
        (1, 3, 224, 224),
        [
            ("self.features.0", "weight"),
            ("self.features.3.expand3x3", "weight"),
            ("self.features.12.squeeze", "weight"),
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
