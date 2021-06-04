# torchslim优化resnet56模型
## 介绍
pytorch-cifar 为模型压缩工具torchslim使用的示例代码，以resnet56为例展示了模型压缩工具集torchslim的使用方法：  
1）pytorch-cifar/main.py为模型的训练文件  
2）pytorch-cifar/acnet.py为重参数化方法使用示例  
3）pytorch-cifar/prune.py为剪枝方法的使用示例  
4）pytorch-cifar/qat.py为量化感知训练的使用示例  
经过QAT之后，模型导出为tensorrt格式，可直接部署在tensorrt上  
## 事例执行步骤
### 训练resnet56基模型
python main.py --topic resnet56 --gpu 0  
训练完成后，模型被存储在checkpoint/resnet56/ckpt.pth 目录下
### acnet训练
python acnet.py  
训练完成后，模型被存储在checkpoints/resnet56_acnet/model.pth 目录下  
### 剪枝
python prune.py  
训练完成后，模型被存储在checkpoints/resnet56_resrep/model.pth 目录下  
### 量化
python qat.py  
训练完成后，模型被存储在checkpoints/resnet56_qat/model.pth 目录下  