# torchslim 算法库
## 介绍
torchslim 是一个模型压缩算法工具集，包含了若干种模型压缩算法
1） ACNet、CnC、ACBCorner等一系列重参数化方法
2） ResRep剪枝方法
2） CSGD剪枝方法
3） QAT感知量化训练，tensorrt部署
## requirement
torchpruner
pytorch>=1.3
tensorrt>=7.2 (仅QAT后导出TensorRT模型需要，不进行量化导出可不安装)
## 依赖安装
pip install -r requirements.txt
## 使用文档
[torchslim使用文档](DOCUMENT.md)
## 工具使用示例
示例用法见pytorch-cifar文件夹中的文件
[pytorch-cifar](../examples/torchslim/pytorch-cifar/README.md)
