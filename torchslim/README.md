# torchslim 算法库
## 介绍
torchslim是一个模型压缩算法工具集，包含了若干种模型压缩算法  
torchpruner 模型剪枝工具主要包含了三部分：  
1） Acnet重参数化方法  
2） ResRep剪枝方法  
3） QAT感知量化训练，tensorrt部署  
## requirement
torchpruner  
pytorch>=1.3  
tensorrt>=7.2 (仅QAT模型导出需要，不进行量化导出可不安装)  
## 依赖安装
pip install -r requirements.txt  
## 使用文档
[torchslim使用文档](DOCUMENT.md)  
## 工具使用示例
示例用法见pytorch-cifar文件夹中的文件  
[pytorch-cifar](/examples/torchslim/pytorch-cifar/)  
