import sys

sys.path.append("..")
import torch
import torchpruner
import torchvision
import numpy as np

# 以下代码示例了对每一个BN层去除其weight系数绝对值前20%小的层

# 加载模型
model = torchvision.models.vgg11_bn()

# 创建ONNXGraph对象，绑定需要被剪枝的模型
graph = torchpruner.ONNXGraph(model)
##build ONNX静态图结构，需要指定输入的张量
graph.build_graph(inputs=(torch.zeros(1, 3, 224, 224),))

# 遍历所有的Module
for key in list(graph.modules):
    module = graph.modules[key]
    # 如果该module对应了BN层
    if isinstance(module.nn_object, torch.nn.BatchNorm2d):
        # 获取该对象
        nn_object = module.nn_object
        # 排序，取前20%小的权重值对应的index
        weight = nn_object.weight.detach().cpu().numpy()
        index = np.argsort(np.abs(weight))[: int(weight.shape[0] * 0.2)]
        result = module.cut_analysis("weight", index=index, dim=0)
        model, context = torchpruner.set_cut(model, result)
        if context:
            # graph 存放了各层参数和输出张量的 numpy.ndarray 版本，需要更新
            graph = torchpruner.ONNXGraph(model)  # 也可以不重新创建 graph
            graph.build_graph(inputs=(torch.zeros(1, 3, 224, 224),))

# 新的model即为剪枝后的模型
print(model)
