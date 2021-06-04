import sys
import torchpruner
import torchpruner.model_tools as model_tools
import torchvision

import torch

import torch.nn as nn

# model creator,  the input shape, prune_modules
models = {
    #     "faster_rcnn": (torchvision.models.detection.fasterrcnn_resnet50_fpn,(1,3,300,400),
    #     [('self.backbone.body.layer2.3.bn3','weight'),('self.backbone.body.layer4.0.downsample.0','weight'),('self.backbone.fpn.layer_block.0','weight'),
    #     ('self.rpn.head.conv','weight'),('self.roi_heads.box_head.fc6')]),
    "mask_rcnn": (
        torchvision.models.detection.maskrcnn_resnet50_fpn,
        (1, 3, 500, 400),
        [
            ("self.backbone.fpn.inner_blocks.1", "weight"),
            ("self.roi_heads.mask_head.mask_fcn1", "weight"),
            ("self.roi_heads.mask_predictor.conv5_mask", "weight"),
        ],
    )
}


for model_key in models.keys():
    print("Testing " + model_key + " ....")
    model_function, input_size, prune_modules = models[model_key]
    input_tensor = (torch.zeros(input_size),)
    for prune_module, attribute in prune_modules:
        print("The prune module is:" + str(prune_module))
        model = model_function()
        model = model.eval()
        graph = torchpruner.ONNXGraph(model)
        graph.build_graph(input_tensor)
        module = graph.modules[prune_module]
        result = module.cut_analysis(attribute, index=[0, 1, 2, 3], dim=0)
    print("End Testing....")
