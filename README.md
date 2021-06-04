# pytorch自动化模型压缩工具库

## 介绍
pytorch自动化模型压缩工具库是针对于pytorch模型的自动化模型压缩工具库，用户无需理解模型的结构，即可以直接对模型完成剪枝和查找替换等操作。  
工具库包含有两个部分，第一个部分为torchpruner模型分析与修改工具库，通过该工具库可以自动对模型完成模块修改和通道修改等常规操作，而无需关注模型的结构细节。  
第二个部分为torchslim模型压缩算法库，包含了模型重参数化、剪枝、感知量化训练等多种模型压缩算法，，用户仅需给出需要被压缩的模型并定义好用于训练的hook函数，即可以对模型进行自动压缩，并输出被压缩模型产物。  

## requirement
* onnx>=1.6  
* onnxruntime>=1.5  
* pytorch>=1.7  
* tensorboardX>=1.8
* scikit-learn  

## 安装
python setup.py install  

## 总揽  
### torchpruner
torchpruner为pytorch模型分析与修改工具库，包含了以下功能：  
1）自动分析模型结构和剪枝    
2）特定模型结构查找与替换    
### torchslim
torchslim内包含了模型压缩的特定算法：  
1）ACNet重参数化方法  
2）ResRep模型剪枝方法  
3）CSGD模型剪枝方法  
4）QAT量化感知训练，并将pytorch模型导出为tenosrrt模型  
### examples
examples文件夹主要包含了多种支持的模型的测试列表support_model，torchpruner工具库的使用示例以及torchslim工具库的使用示例。  
1）support_model：支持的若干种模型  
2）torchpruner：使用torchpruner剪枝和模块修改的简单示例  
3）torchslim：使用torchslim 在分类模型上的简单示例  

## torchpruner模型修改
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

## torchslim模型压缩
```python
import torchslim

#predict_function的第一个参数为model，第二个参数为一个batch的data,data已经被放置到了GPU上
config['task_name']='resnet56_prune'
config['epoch']=90
config['lr']=0.1
config['prune_rate']=0.5
config['save_path']="model/save/path"
config['dataset_generator']=dataset_generator
config['predict_function']=predict_function
config['evaluate_function']=evaluate_function
config['calculate_loss_function']=calculate_loss_function

model=torch.load("model/path")

#创建solver
solver=torchslim.pruning.resrep.ResRepSolver(model,config)
#执行压缩
solver.run()
```

## 使用说明
详细使用说明见各自文件夹README.md  
[torchpruner](torchpruner/README.md)  
[torchslim](torchslim/README.md)  
