import sys

sys.path.append("..")
import torch
import torchpruner
import torchvision

# 以下代码示例展示了对模型进行剪枝后，对模型的恢复操作

# 加载模型
model = torchvision.models.resnet50()

# 创建ONNXGraph对象，绑定需要被剪枝的模型
graph = torchpruner.ONNXGraph(model)
##build ONNX静态图结构，需要指定输入的张量
graph.build_graph(inputs=(torch.zeros(1, 3, 224, 224),))

# 获取conv1模块对应的module
conv1_module = graph.modules["self.conv1"]

# 对前四个通道进行剪枝分析,指定对weight权重进行剪枝,剪枝前四个通道
# weight权重out_channels对应的通道维度为0
result = conv1_module.cut_analysis(attribute_name="weight", index=[0, 1, 2, 3], dim=0)

# result会返还分析结果，提供当weight的前四个通道发生了剪枝，那么model中有哪些权重需要被去除
print(result)

# 剪枝执行模块执行剪枝操作，对模型完成剪枝过程.context变量提供了用于剪枝恢复的上下文
model, context = torchpruner.set_cut(model, result)
# 新的model即为剪枝后的模型
print(model)

# model的被去除的通道被恢复，其参数与原有参数一致
model = torchpruner.recovery_cut(model, result, context)
# 被恢复的model
print(model)

# set_zero则将参数置0,而不是去除
model, context = torchpruner.set_zero(model, result)
print(model)
# 恢复被置0的参数
model = torchpruner.recovery_zero(model, result, context)
print(model)
