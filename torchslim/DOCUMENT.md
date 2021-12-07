# torchslim
## 介绍
torchslim是在torchpruner的基础上开发的一系列模型压缩库.   
给定需要被压缩的模型后, 用户定义一系列的超参数和与训练模型任务相关的函数钩子,然后即可以使用torchslim工具对模型进行模型压缩操作.  
目前从大类上包含了: 重参数化/剪枝/量化等方法
其中重参数化方法,支持ACNet重参数化算法.  
剪枝支持ResRep剪枝算法.  
量化支持QAT感知量化训练,并支持onnx导出和tensorrt部署.  
该库的开发遵循了两项原则:
1) 尽可能适用于多种任务和多种模型  
2) 所有的算法产物尽量可用于直接部署  
以上原则使得torchslim可以尽量满足多种模型结构和多种任务,并服务于实际的模型部署开发  

## 使用说明
torchslim中开发的算法均使用一个Solver对象来完成模型压缩的训练流程,不同的算法对应了不同的Solver.  
每一个Solver在创建的过程中接收一个config字典来完成模型的配置操作,接收model对象,作为被优化的对象.  
config字典主要需要配置两项内容:
1) torchslim算法的超参数  
常见的内容有lr, epoch等
2) 与模型训练相关的函数钩子  
常见的内容有predict_function/calcuate_loss_function/evaluate_function等  
Solver创建完成后, 使用run函数运行模型压缩  
下面通过简单的ResRep例子代码,展示配置过程:  
首先加载模型:   
```python
import torchvision
model=torchvision.models.resnet50()
```
定义数据集加载函数dataset_generator:  
```python
def dataset_generator():
    trainset=torchvision.datasets.ImageFolder(root='/data/to/train')
    valset=torchvision.datasets.ImageFolder(root='data/to/val')
    return trainset,valset
```
定义模型训练用的predict_function,calculate_loss_function,evaluate_function:
```python
#predict_function的第一个参数为model，第二个参数为一个batch的data,data已经被放置到了GPU上
def predict_function(model,data):
    X,Y=data
    prediction=model(X)
    return prediction

#calculate_loss_function 第一个参数为prediction，第二个参数为data
def calculate_loss_function(prediction,data):
    X,Y=data
    loss=F.cross_entropy(prediction,Y)
    return loss

#evaluate_function 第一个参数为prediction，第二个参数为data
def evaluate_function(prediction,data):
    _,Y=data
    _, predicted = predict.max(1)
    correct=predicted.eq(Y).sum().item()
    return {"acc":correct/predict.size(0)}
    
```
配置solver的config
```python
#predict_function的第一个参数为model，第二个参数为一个batch的data,data已经被放置到了GPU上
config['task_name']='resnet56_prune'
config['epoch']=90
config['lr']=0.1
config['prune_rate']=0.5
config['dataset_generator']=dataset_generator
config['predict_function']=predict_function
config['evaluate_function']=evaluate_function
config['calculate_loss_function']=calculate_loss_function
#创建solver
import torchslim

solver=torchslim.pruning.resrep.ResRepSolver(model,config)
solver.run()
    
```

## 方法介绍
### ACNet重参数化
ACNet参数化方法是在训练过程中使用多个分支的卷积结构，在部署的过程中将多个分支的卷积结构合并的方法，从而使得在训练过程中可以有效利用多个卷积核参数量的优势来提升模型精度，
而在推理的过程中不增加运算开销。  
#### Solver
torchslim.reparameter.acnet.ACNetSolver  
#### config的参数
* 用于训练的超参数:
    * lr: 学习率，默认值为0.1
    * epoch： 数据集训练多个遍，默认为 360
    * batch_size: 每次迭代使用多少数据，，默认值为128
    * test_batch_size: 测试时每次使用多少数据,默认为128
    * momentum: 对于有动量设定的优化器的动量值，默认值为0.9
    * weight_decay: 权重惩罚系数，默认值为0.0001
* ACNet 相关的超参数:
    * acnet_type: acnet的类型,当前仅可选择acnet_cr,默认为acnet_cr
    * save_deploy_format: 模型在存储时是否合并多分支卷积,默认值为True
* 系统资源相关的配置
    * task_name: 任务的名称，这项参数决定了数据输入到tensorboardx上的显示名称以及模型的存储路径
    * num_workers: 读取数据使用的workers，默认值为0
    * devices: 一个整形list，代表了使用的GPU的标号，默认为[]
    * log_interval: 每多少次训练打印一次日志，默认为200
* 存储相关的参数:
    * save_dir: 模型的存储路径，默认为checkpoints
    * save_keyword: 用于存储模型的评价标准的key值，最优模型的存储由save_keyword决定
* 与训练任务相关的钩子函数：  
    * predict_function: 根据模型和数据返回预测值，参数形式为(model,data)->prediction, 其中model为当前模型，data为一个iteration的数据，返回值为模型的输出。
    * calculate_loss_function: 根据prediction值和data，计算模型的损失函数，并将损失函数返回，参数形式为(prediction,data)->loss, 其中prediction为predict_function的返回值，data为一个iteration的数据，返回值为loss。
    * evaluate_function: 根据prediction值和data，计算评估指标，评估指标将会被输出在log上，以及根据save_keyword作为模型存储的依据，参数形式为(prediction,data)->dict，prediction和data同calculate_loss_function，返回值为一个dict，其中key为评估指标的名称，value为评估值。
    * dataset_generator: 组织训练中使用的的dataset，返回训练集和验证集的dataset，
    参数形式为()->trainset,valset
    * optimizer_generator: optimizer生成函数,形式为(config,params)->optimizer, 其中config即为solver的config,params为模型的参数字典. 返回值为一个优化器, 默认情况下optmizer_generator返回SGD优化器.
    * schduler_generator: scheduler生成函数,形式为(config,optimzer)->scheduler, shcheduler默认每一个epoch调用一次,其中config为sovler的config,optmizer为optimizer_generator生成的优化器, 默认情况下scheduler_generator返回cos_lr_scheduler

### Resrep剪枝方法
Resrep剪枝方法是一种带有权重惩罚的剪枝方法，不同于直接在Conv层上对参数添加Lasso Decay，Resrep方法在模型的bn层之后，添加了一个单位矩阵，并对单位矩阵做Lasso Decay惩罚，然后在训练过程中，逐渐将较小的单位矩阵的列减除。
在部署的过程中，额外的单位矩阵，bn层和Conv层可以融合在一起，从而起到实际的提速作用。
#### Solver
torchslim.pruning.resrep.ResRepSolver  
#### config的参数
* 用于训练的超参数:
    * lr: 学习率，默认值为0.1
    * epoch： 数据集训练多个遍，默认为 360
    * batch_size: 每次迭代使用多少数据，默认值为128
    * test_batch_size: 测试时每次使用多少数据, 默认为128
    * momentum: 对于有动量设定的优化器的动量值，默认值为0.9
    * weight_decay: 权重惩罚系数，默认值为0.0001
* ResRep 相关的超参数:
    * save_deploy_format: 模型合并Conv BN Compactor,默认True
    * warmup_epoch: 前若干个epoch不执行剪枝操作,默为 5
    * compactor_momentum: compactor 单位矩阵的momentum值,默认为0.99
    * lasso_decay: lasso_decay惩罚项,默认为0.0001
    * prune_rate: 模型的剪枝率, 默认为0.5
    * prune_groups: 每次剪枝减除的组数,默认为1
    * groups_size: 每组在单个compactor减除的channel数,默认为8
    * min_channels: 保留的最少channels数,默认为8
 * 与模型结构相关的参数:
    * input_shapes: 模型的输入的size组成的list,一个二维的list,第一维代表了有多少个输入,第二维代表了输入的size,如果其为None,input_shapes将默认为dataset返回的samples的第一个张量的size,注意size不包含batch维度. 默认值为None
    * prune_module_names: 一个list,list内为需要被剪枝的bn/conv层的名称,如果项为None,则模型自动查找需要剪枝的BN/Conv层. 默认值为None. 注意: 尽管prune_module_names可以设定需要被剪枝的bn/conv层,但是由于depthwise卷积在本方法中不支持,因此与depthwise卷积相邻的bn层和groups!=1的conv层将被自动忽略
    * auto_find_module_strategy: 自动查找bn/conv层的策略, 可选'linear_bn'|'all_bn'|'linear_conv'|'all_conv','all'代表所有的bn/conv层都被考虑,'linear'仅考虑简单串行结构中的bn/conv,例如在resnet50中,仅添加残差结构中的第一个bn和第二bn,第三个bn由于其剪枝会级联破坏resnet50中的主干结构,因此不做考虑 ,默认为'linear_bn'. 注意: 1)与prune_module_names一致 尽管prune_module_names可以设定需要被剪枝的bn/conv层,但是由于depthwise卷积在本方法中不支持,因此与depthwise卷积相邻的bn层和groups!=1的conv层将被自动忽略将被自动忽略. 2) ConvTranspose 也被视为卷积的一种. 注意auto_find_module_strategy智能进行简单的模块查找,不能针对各种结构都能有效的查找到需要被剪枝的模块,当auto_find_module_strategy无法正确查找bn和conv层时,推荐使用prune_module_names来指定被剪枝的层.
* 系统资源相关的配置
    * task_name: 任务的名称，这项参数决定了数据输入到tensorboardx上的显示名称以及模型的存储路径
    * num_workers: 读取数据使用的workers，默认值为0
    * devices: 一个整形list，代表了使用的GPU的标号，默认为[]
    * log_interval: 每多少次训练打印一次日志，默认为200
* 存储相关的参数:
    * save_dir: 模型的存储路径，默认为checkpoints
    * save_keyword: 用于存储模型的评价标准的key值，最优模型的存储由save_keyword决定
* 与训练任务相关的钩子函数：  
    * predict_function: 根据模型和数据返回预测值，参数形式为(model,data)->prediction, 其中model为当前模型，data为一个iteration的数据，返回值为模型的输出。
    * calculate_loss_function: 根据prediction值和data，计算模型的损失函数，并将损失函数返回，参数形式为(prediction,data)->loss, 其中prediction为predict_function的返回值，data为一个iteration的数据，返回值为loss。
    * evaluate_function: 根据prediction值和data，计算评估指标，评估指标将会被输出在log上，以及根据save_keyword作为模型存储的依据，参数形式为(prediction,data)->dict，prediction和data同calculate_loss_function，返回值为一个dict，其中key为评估指标的名称，value为评估值。
    * dataset_generator: 组织训练中使用的的dataset，返回训练集和验证集的dataset，
    参数形式为()->trainset,valset
    * optimizer_generator: optimizer生成函数,形式为(config,params)->optimizer, 其中config即为solver的config,params为模型的参数字典. 返回值为一个优化器, 默认情况下optmizer_generator返回SGD优化器.
    * schduler_generator: scheduler生成函数,形式为(config,optimzer)->scheduler, shcheduler默认每一个epoch调用一次,其中config为sovler的config,optmizer为optimizer_generator生成的优化器, 默认情况下scheduler_generator返回cos_lr_scheduler

### CSGD剪枝方法
CSGD 剪枝方法是一种基于通道聚类的剪枝方法，在模型训练之前将每一层的通道聚类，然后在训练过程中添加约束，使得每一层被聚类的通道趋同，经过训练后，最终将趋同的通道合并去除。
#### Solver
torchslim.pruning.acnet.CSGDSolver
#### config的参数
* 用于训练的超参数:
    * lr: 学习率，默认值为0.1
    * epoch： 数据集训练多个遍，默认为 360
    * batch_size: 每次迭代使用多少数据，默认值为128
    * test_batch_size: 测试时每次使用多少数据, 默认为128
    * momentum: 对于有动量设定的优化器的动量值，默认值为0.9
    * weight_decay: 权重惩罚系数，默认值为0.0001
* ResRep 相关的超参数:
    * prune_epoch: 在哪个epoch执行剪枝操作，将冗余的通道去除，后续的epoch则进行finetune训练
    * centri_strength：聚类约束强度，默认为3e-3
    * cluster_percent：每一层通道数量压缩率，剪枝率大约为1-(1-cluster_percent)^2
 * 与模型结构相关的参数:
    * input_shapes: 模型的输入的size组成的list,一个二维的list,第一维代表了有多少个输入,第二维代表了输入的size,如果其为None,input_shapes将默认为dataset返回的samples的第一个张量的size,注意size不包含batch维度. 默认值为None
    * prune_module_names: 一个list,list内为需要被剪枝的bn/conv层的名称,如果项为None,则模型自动查找需要剪枝的BN/Conv层. 默认值为None. 注意: 尽管prune_module_names可以设定需要被剪枝的bn/conv层,但是由于depthwise卷积在本方法中不支持,因此与depthwise卷积相邻的bn层和groups!=1的conv层将被自动忽略
    * auto_find_module_strategy: 自动查找bn/conv层的策略, 可选linear_conv'|'all_conv','all'代表所有的bn/conv层都被考虑,'linear'仅考虑简单串行结构中的bn/conv,例如在resnet50中,仅添加残差结构中的第一个bn和第二bn,第三个bn由于其剪枝会级联破坏resnet50中的主干结构,因此不做考虑 ,默认为'linear_conv' 2) ConvTranspose 也被视为卷积的一种. 注意auto_find_module_strategy智能进行简单的模块查找,不能针对各种结构都能有效的查找到需要被剪枝的模块,当auto_find_module_strategy无法正确查找bn和conv层时,推荐使用prune_module_names来指定被剪枝的层.
* 系统资源相关的配置
    * task_name: 任务的名称，这项参数决定了数据输入到tensorboardx上的显示名称以及模型的存储路径
    * num_workers: 读取数据使用的workers，默认值为0
    * devices: 一个整形list，代表了使用的GPU的标号，默认为[]
    * log_interval: 每多少次训练打印一次日志，默认为200
* 存储相关的参数:
    * save_dir: 模型的存储路径，默认为checkpoints
    * save_keyword: 用于存储模型的评价标准的key值，最优模型的存储由save_keyword决定
* 与训练任务相关的钩子函数：  
    * predict_function: 根据模型和数据返回预测值，参数形式为(model,data)->prediction, 其中model为当前模型，data为一个iteration的数据，返回值为模型的输出。
    * calculate_loss_function: 根据prediction值和data，计算模型的损失函数，并将损失函数返回，参数形式为(prediction,data)->loss, 其中prediction为predict_function的返回值，data为一个iteration的数据，返回值为loss。
    * evaluate_function: 根据prediction值和data，计算评估指标，评估指标将会被输出在log上，以及根据save_keyword作为模型存储的依据，参数形式为(prediction,data)->dict，prediction和data同calculate_loss_function，返回值为一个dict，其中key为评估指标的名称，value为评估值。
    * dataset_generator: 组织训练中使用的的dataset，返回训练集和验证集的dataset，
    参数形式为()->trainset,valset
    * optimizer_generator: optimizer生成函数,形式为(config,params)->optimizer, 其中config即为solver的config,params为模型的参数字典. 返回值为一个优化器, 默认情况下optmizer_generator返回SGD优化器.
    * schduler_generator: scheduler生成函数,形式为(config,optimzer)->scheduler, shcheduler默认每一个epoch调用一次,其中config为sovler的config,optmizer为optimizer_generator生成的优化器, 默认情况下scheduler_generator返回cos_lr_scheduler

### QAT感知量化训练
QAT即为感知量化训练,即在训练的过程中对模型中的参数和激活值进行量化/反量化过程,模拟模型在真实量化过程中的损失.  
在本QAT实现中,采用了tensor粒度的对称量化,经过QAT的模型, 在存储的过程中会直接转换成onnx形式,并再转换成tensorrt格式,该格式可以直接部署在tensorrt GPU上.
而在推理的过程中不增加运算开销。  
#### Solver
torchslim.reparameter.quantizing.qat.QATSolver
#### config的参数
* 用于训练的超参数:
    * lr: 学习率，默认值为0.001
    * epoch： 数据集训练多个遍，默认为 360
    * batch_size: 每次迭代使用多少数据，默认值为128
    * test_batch_size: 测试时每次使用多少数据,默认为128
    * momentum: 对于有动量设定的优化器的动量值，默认值为0.9
    * weight_decay: 权重惩罚系数，默认值为0.0001
* 系统资源相关的配置
    * task_name: 任务的名称，这项参数决定了数据输入到tensorboardx上的显示名称以及模型的存储路径
    * num_workers: 读取数据使用的workers，默认值为0
    * devices: 一个整形list，代表了使用的GPU的标号，默认为[]
    * log_interval: 每多少次训练打印一次日志，默认为200
* 存储相关的参数:
    * save_dir: 模型的存储路径，默认为checkpoints
    * save_keyword: 用于存储模型的评价标准的key值，最优模型的存储由save_keyword决定
* 与训练任务相关的钩子函数：  
    * predict_function: 根据模型和数据返回预测值，参数形式为(model,data)->prediction, 其中model为当前模型，data为一个iteration的数据，返回值为模型的输出。
    * calculate_loss_function: 根据prediction值和data，计算模型的损失函数，并将损失函数返回，参数形式为(prediction,data)->loss, 其中prediction为predict_function的返回值，data为一个iteration的数据，返回值为loss。
    * evaluate_function: 根据prediction值和data，计算评估指标，评估指标将会被输出在log上，以及根据save_keyword作为模型存储的依据，参数形式为(prediction,data)->dict，prediction和data同calculate_loss_function，返回值为一个dict，其中key为评估指标的名称，value为评估值。
    * dataset_generator: 组织训练中使用的的dataset，返回训练集和验证集的dataset，
    参数形式为()->trainset,valset
    * optimizer_generator: optimizer生成函数,形式为(config,params)->optimizer, 其中config即为solver的config,params为模型的参数字典. 返回值为一个优化器, 默认情况下optmizer_generator返回SGD优化器.
    * schduler_generator: scheduler生成函数,形式为(config,optimzer)->scheduler, shcheduler默认每一个epoch调用一次,其中config为sovler的config,optmizer为optimizer_generator生成的优化器, 默认情况下scheduler_generator返回cos_lr_scheduler


