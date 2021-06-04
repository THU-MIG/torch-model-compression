import sys

sys.path.append("..")
import torch
import torchpruner
import torchpruner.model_tools as tools
import torchvision

# 以下包含了model_tools函数的用法

# 创建模型
model = torchvision.models.resnet50()

## 根据名称获取nn.Module对象
conv1 = tools.get_object(model, "self.conv1")

## 根据名称设定nn.Module对象
model = tools.set_object(model, "self.conv1", torch.nn.Identity())

## 查找指定类的所有对象对应的名称
name_list = tools.get_names_by_class(model, torch.nn.Conv2d)

# 创建模型
model = torchvision.models.resnet50()

## 替换指定名称的所有对象
def replace_function(name, origin_object):
    return torch.nn.Identity()


model = tools.replace_object_by_names(model, name_list, replace_function)

# 创建模型
model = torchvision.models.resnet50()
##替换指定类的所有nn.Module对象
model = tools.replace_object_by_class(model, torch.nn.Conv2d, replace_function)

# 创建模型
model = torchvision.models.resnet50()
##查找连续的一组nn.Module对象,在图结构中具体顺序关系
graph = torchpruner.ONNXGraph(model)
graph.build_graph(inputs=(torch.zeros(1, 3, 224, 224),))
name_group_list = tools.get_name_groups_by_classes(
    graph, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
)

##替换连续的一组nn.Module对象
def replace_conv_bn(name, module_list):
    return torch.nn.Identity(), torch.nn.Identity()


model = tools.replace_object_by_name_groups(model, name_group_list, replace_conv_bn)
