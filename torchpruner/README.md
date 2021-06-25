# torchpruner 模型剪枝工具
## 介绍
torchpruner是一个模型剪枝工具。针对pytorch模型，模型剪枝工具可以自动分析模型的结构，
以确定当某个模块的某些通道被去除后，模型中的哪些参数需要被去除，以保证剪枝后的模型维度的统一，并对模型进行结构化剪枝。
torchpruner 模型剪枝工具主要包含了三部分：  
1） 模型结构分析工具  
2） 模型剪枝操作工具  
3） 模型操作工具集  

## requirement
* onnx>=1.6  
* onnxruntime>=1.5  
* pytorch>=1.7  

## 安装
python setup.py install  

## 工具使用文档
[torchpruner文档](DOCUMENT.md)
## 工具使用示例
使用示例如下：
```python
import torch
import torchpruner
import torchvision
#加载模型
model=torchvision.models.resnet50()

#创建ONNXGraph对象，绑定需要被剪枝的模型
graph=torchpruner.ONNXGraph(model)
##build ONNX静态图结构，需要指定输入的张量
graph.build_graph(inputs=(torch.zeros(1,3,224,224),))

#获取conv1模块对应的module
conv1_module=graph.modules['self.conv1']
#剪枝分析
result=conv1_module.cut_analysis(attribute_name='weight',index=[0,1,2,3],dim=0)
#执行剪枝操作
model,context=torchpruner.set_cut(model,result)
#对卷积模块进行剪枝操作

```
简单示例见examples  
[examples](examples)
