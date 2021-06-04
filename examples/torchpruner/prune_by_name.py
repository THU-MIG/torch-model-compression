import sys

sys.path.append("..")
import torch
import torchpruner
import torchvision

# 以下代码示例了给定pytorch模块的名称，以及指定剪枝的通道，对pytorch的模型进行剪枝操作
# 假设被剪枝的模块为模型的conv1卷积模块，那么模块的名称为self.conv1
# 以剪枝前4个通道为例

# 加载模型
model = torchvision.models.resnet50()

# 创建ONNXGraph对象，绑定需要被剪枝的模型
graph = torchpruner.ONNXGraph(model)
##build ONNX静态图结构，需要指定输入的张量
graph.build_graph(inputs=(torch.zeros(1, 3, 224, 224),))

## graph经过build后，即创建了静态图结构，静态图结构由Node和Operator对象构成
## 另外为了方便操作，graph中还创建了Module结构
## 每个Module和pytorch模型中的nn.Module对象一一对应，并提供了剪枝的操作接口

# 获取conv1模块对应的module
conv1_module = graph.modules["self.conv1"]

# 对前四个通道进行剪枝分析,指定对weight权重进行剪枝,剪枝前四个通道
# weight权重out_channels对应的通道维度为0
result = conv1_module.cut_analysis(attribute_name="weight", index=[0, 1, 2, 3], dim=0)

# result会返还分析结果，提供当weight的前四个通道发生了剪枝，那么model中有哪些权重需要被去除
print(result)

# 剪枝执行模块执行剪枝操作，对模型完成剪枝过程
model, context = torchpruner.set_cut(model, result)

# 新的model即为剪枝后的模型
print(model)
