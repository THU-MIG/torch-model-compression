# torchpruner 文档
## API总览
* 基本用法
    * Graph对象
    * Module对象
    * Model Pruner
    * Function Module
    * Model Tools
* 高级用法
    * DataNode Object
    * Operator Object
    * Module Pruner 注册
    * Operator 注册

## 基本用法
### Graph对象
Graph对象是pytorch模型的静态图表示形式.可以由pytorch模型转换而成.  
对pytorch模型的剪枝的分析和图结构的分析都在Graph对象上完成.  
Graph对象包含了多组DataNode对象和Opeartor对象,DataNode和Opeartor对象互相连接,构成了模型的静态图结构.  
为了方便使用Graph对象包含了多组Module对象,每一个Module对象一边绑定了pytorch模型中的一个nn.Module 对象,  
另外一边绑定了pytorch nn.Module对象在运算过程中使用的DataNode和Operator对象.  
用户通过Module对象实现对pytorch模型的剪枝分析和图结构分析.  

#### 创建Graph对象
Graph对象目前仅支持ONNX类型,可通过torchpruner.ONNXGraph创建,然后通过build_graph构造图结构  
方法:  
torchpruner.ONNXGraph(model:torch.nn.Module,onnx_device:str='CPU')  
说明:  
torchpruner.ONNXGraph创建一个ONNXGraph对象,该对象绑定了model  
参数:  
model:torch.nn.Module 一个待转换成静态图分析的pytorch model  
onnx_device:str 在onnx格式下前向运算的过程中使用的设备,可选择 'CPU'|'GPU' 默认为'CPU'  
返回值:  
一个ONNXGraph对象  
使用示例:  
```python
import torch
import torchpruner
import torchvision
#加载模型
model=torchvision.models.resnet50()

#创建ONNXGraph对象，绑定需要被剪枝的模型
graph=torchpruner.ONNXGraph(model)

```

#### 构造Graph
Graph对象创建好以后,通过build_graph函数,构造静态图结构  
方法:  
ONNXGraph.build_graph(inputs, fill_value=True, training=False)  
说明:  
根据输入,创建ONNXGraph静态图结构  
参数:  
inputs: 模型的输入张量,tuple类型,tuple的每一个元素是一个torch.Tensor张量  
fill_value: 是否填充ONNXGraph的中间张量的计算结果,默认为True  
training: 是否训练模式构造图结构,默认为True  
使用示例:   
```python
# 一个输入,batch size =1 channel=3 height=224 width=224
inputs=(torch.zeros(1,3,224,224),)
graph.build_graph(inputs)
```

#### forward
在Graph中,执行前向运算,并根据输入,更新图结构中DataNode的计算结构  
方法:  
ONNXGraph.forward(inputs)  
说明:  
根据输入,对ONNXGraph执行前向运算,中间计算结构保存在DataNode中  
参数:  
inputs: 模型的输入张量,tuple类型,tuple的每一个元素是一个torch.Tensor张量  

#### set_device
设置onnx图执行的使用的device  
方法:  
ONNXGraph.set_device(device:str)  
说明:  
设置onnx图执行使用的device,可选择'CPU'|'GPU'  
参数:  
device: 'CPU'|'GPU'  

#### get_device
获取onnx图执行使用的device  
方法:  
ONNXGraph.get_device()  
说明:  
获取onnx图执行使用的device  'CPU'|'GPU'  

#### flops
计算pytorch model的flops  
方法:  
ONNXGraph.flops()  
返回值:  
浮点数,单位为m  

#### graph对象中的属性
##### modules:Dict[str,Module]
一个 OrderedDict 字典, 可通过模块名称访问Module对象  
使用示例:  
```python
#访问pytorch 模型中 nn.Module conv1 对应的Module对象
module=graph.modules['self.conv1']
```
Module的使用方法详见高级使用  

##### nodes:DictDict[str,DataNode]
一个 OrderedDict 字典, 存储了图解构中的所有数据节点,其中模型的参数对应的DataNode可以通过在pytorch model中的参数名称访问  
模块的中间输出,使用数字标识  
使用示例:  
```python
#访问pytorch 模型中 nn.Module conv1的权重
node=graph.node['self.conv1.weight']

#访问 110号中间变量
node=graph.node['110']
```
DataNode的使用用法,详见高级使用

##### operators:Dict[str,operator.OperatorNode]
一个 OrderedDict 字典,存储了图解钩中的所有操作节点  
OpeartorNode的使用方法详见高级使用  

### Module对象
Module对象是连接pytorch模型和Graph图结构的用户操作接口,通过graph.modules字典直接获得    
一个Module对象关联了pytorch模型中对应的nn.Module对象,同时关联了该nn.Module对象的静态图表示对应的DataNode和OperatorNode  
用户通过使用Module对象,实现剪枝分析和特定图结构的搜索  

#### 获取Module对象
Module对象由ONNXGraph结构调用build_graph函数时生成  
通过graph.modules字典直接获得  
以下展示module对象的获取方法  
```python
# 通过名称索引 获取pytorch模型 conv模块对应的Module
conv_module=graph.modules['self.conv']

#遍历所有的模块
for key in graph.modules.keys():
    module=graph.modules[key]
    print(module)
```

#### 使用Module分析剪枝的参数变化
通过Module模块可以分析对pytorch特定模块的参数进行剪枝后,整个模型的其他参数需要发生什么样的变化,以保障模型依然可以合理运行  
举个例子,在VGG模型结构中,当前一个卷积模块的第1个输出通道对应的参数被去除时,其后一个卷积模块的输入通道对应的参数也要被去除  
Module则提供接口用于分析这种关联关系,并将分析结果返回  
方法:  
Module.cut_analysis(attribute_name,index,dim)  
说明:  
给定指定的模块的指定属性,在指定的维度上,去除指定索引的参数, 返回整个pytorch模型中的参数应该发生什么样的改变  
参数:  
attribute_name: 一个字符串,指定了属性的名称,可以是属性的形式例如 'weight',也可以是属性.属性的形式 例如 'bn.weight',最终必须指向pytorch模型中的一个张量  
index: 一个list,指定需要被去除的元素的索引  
dim: 索引的维度  
返回值:
一个字典,包含了 'iner' 'terminal' 'operator' 三个key  
iner 代表了pytorch模型中,中间张量需要改变的index和dim信息  
terminal 代表了pytorch模型中,实际参数需要改变的index dim信息  
operator 代表了由于张量更改,operatorNode中属性的变化  
每一个key对应的元素也是一个字典,其中name对应了DataNode在ONNXGraph.nodes字典中的名称  
value对应了需要发生改变的index和dim  

使用示例:  
下面展示了使用cut_analysis 模型接口,对模型中self.conv模块的前4通道进行分析  
```python
# 通过名称索引 获取pytorch模型 conv模块对应的Module
conv_module=graph.modules['self.conv']

result=conv_module.cut_analysis('weight',index=[0,1,2,3],dim=0)
print(result)
```

#### Module中的属性
##### nn_object
nn_object 绑定了pytorch model中对应的nn.Module对象,在不知道Module名称情况下,使用模块类别进行筛选时具有重要作用  
下面展示了通过nn_object 筛选出 nn.Conv2d类型的模块进行剪枝分析的操作  
```python
# 通过名称索引 获取pytorch模型 conv模块对应的Module
for key in graph.modules.keys():
    module=graph.modules[key]
    if isinstance(module.nn_object,nn.Conv2d):
        result=module.cut_analysis('weight',index=[0,1,2,3],dim=0)
        print(result)
```

##### sub_modules:Dict[str,Module]
sub_modules是一个Dict 包含了该Module模块的子Module,可通过pytorch模型nn.Module模块中对应的属性名称来进行访问  
下面展示其访问方法  
```python
conv_module=graph.modules['self.conv']
conv_weight_module=conv_module.sub_modules['weight']
```

##### in_data:List[DataNode]
in_data 是一个List,包含了所有的输入DataNode  

##### out_data:List[DataNode]
out_data 是一个List,包含了所有的输出DataNode  

##### operators:list[OperatorNode]
operators 是一个List,包含了所有的操作节点  

### Model Pruner
Model Pruner 提供了根据分析结果,对模型进行剪枝操作的工具包.  
经过剪枝操作后,被指定剪枝的参数从模型中去除, 被剪枝的模型的实际flops降低.  
另外,MOdel Pruner还提供了模型剪枝后的恢复操作, 以适用于基于权重贡献的NAS的通道搜索.  

#### 对模型进行剪枝
Model Pruner 可以将分析结果直接应用到被分析的模型上, 对模型实现剪枝操作.  
方法:  
torchpruner.set_cut(model,cut_dict)  
说明:  
输入model和使用cut_alaysis得到的分析结果为参数,对模型执行剪枝操作  
返回被剪枝的模型和被去除的剪枝参数的上下文,被去除的剪枝参数上下文可以用于模型的剪枝后的恢复
参数:
model:被剪枝的pytorch模型
cut_dict: 使用Module.cut_analysis得到的剪枝分析结构
返回值:
返回值有两个  
1) 被剪枝后的模型
2) 被去除的参数的上下文

使用示例:  
```python
# 获取卷积module
conv_module=graph.modules['self.conv']
# 去卷积的前四个通道,
analysis_result=conv_module.cut_analysis('weight',index=[0,1,2,3],dim=0)
# 执行剪枝操作
model,param_context=torchpruner.set_cut(model,analysis_result)

```

#### 对被剪枝的模型进行恢复
被剪枝的模型可以通过参数上下文恢复到未剪枝的状态  
方法:  
torchpruner.recovery_cut(model,cut_dict,param_context)  
说明:
根据参数上下文,恢复被剪枝的模型,被恢复的参数与被剪枝时相同  
参数:  
model:被剪枝后的模型  
cut_dict:cut_analysis得到的分析结果  
param_context: 被剪枝的参数上下文  
返回值:
被恢复的模型  

使用示例:
使用示例:  
```python
# 获取卷积module
conv_module=graph.modules['self.conv']
# 去卷积的前四个通道,
analysis_result=conv_module.cut_analysis('weight',index=[0,1,2,3],dim=0)
# 执行剪枝操作
model,param_context=torchpruner.set_cut(model,analysis_result)
# 恢复模型
model=torchpruner.recovery_cut(model,analysis_result,model)

```

#### 对优化器进行剪枝
优化器针对每个被优化的参数可能绑定了状态信息,当模型的参数被剪枝后,其模型的参数对应的优化器中存储的状态信息的参数也需要被剪枝.  
这样才能保证该优化器还可以继续对模型进行剪枝操作.  
方法:  
torchpruner.set_cut_optimizer(model,optimizer,cut_dict)  
说明:  
根据model和使用cut_analysis得到的结果,对optimizer中对应的参数进行剪枝  
参数:
model:需要被剪枝的模型  
optmizer:模型model的优化器  
cut_dict:cut_analysis得到的分析结果  
返回值:  
1) 被剪枝后的优化器
2) 优化器中状态信息的上下文  
使用示例:  
```python
#创建优化器
optimizer=torch.optim.SGD(model,parameters(),lr=0.1,momentum=0.9)
# 获取卷积module
conv_module=graph.modules['self.conv']
# 去卷积的前四个通道,
analysis_result=conv_module.cut_analysis('weight',index=[0,1,2,3],dim=0)
# 执行剪枝操作
model,param_context=torchpruner.set_cut(model,analysis_result)
# 对优化器进行剪枝
optmizer, state_param_context=torchpruner.set_cut_optimizer(model,optmizer,analysis_result)
```

#### 对优化器进行恢复
与set_cut和recovery对应的,还存在recovery_optimizer操作  
方法:  
torchpruner.recovery_optimizer(model,optimizer,cut_dict,param_context)
说明:  
根据模型,cut_dict和set_cut_optmizer的上下文,恢复优化器  
参数:  
model: 需要被剪枝的模型  
optmizer: 模型的优化器  
cut_dict: cut_analysis得到的分析结果  
param_context: 优化器被剪枝的状态参数的上下文  
返回值:  
被恢复的优化器  
使用示例:  
```python
#创建优化器
optimizer=torch.optim.SGD(model,parameters(),lr=0.1,momentum=0.9)
# 获取卷积module
conv_module=graph.modules['self.conv']
# 去卷积的前四个通道,
analysis_result=conv_module.cut_analysis('weight',index=[0,1,2,3],dim=0)
# 执行剪枝操作
model,param_context=torchpruner.set_cut(model,analysis_result)
# 对优化器进行剪枝
optmizer, state_param_context=torchpruner.set_cut_optimizer(model,optmizer,analysis_result)
#对优化器恢复
optimizer=torchpruner.recovery_optimizer(model,optmizer,analysis_result,state_param_context)
```

#### 对模型进行参数置零与恢复
torchpruner还支持对将模型中的参数置零,而不是将参数去除,相应的还提供了将置0的参数进行恢复的方法,  
分别为:     
1)torchpruner.set_zero(model,cut_dict)  
2)torchpruner.recovery_zero(model,analysis_result,model)  
用法与set_cut和recovery_cut相同  


### Function Module
Function Module是一种运行模式,在pytorch模型的运行过程中,将其中的函数,例如torch.add F.relu等替换成对应的等价的Function Module执行,  从而让函数本身模块化. 
在Function Module模式中,nn.Module中的每一个函数调用都对应了一个Function Module对象的,Function Module对象被存储在当前nn.Module对象的 __function_module_list中,
每一个原有的函数调用,会替换成对应Function Module的forward方法调用, 其执行的逻辑不变.   
函数模块化对模型的结构查找以及全自动的模型量化支持具有重要作用.  
目前支持F.relu torch.add torch.mul torch.cat 四种函数的模块化.  

#### 开启Function Module Mode
开启Function Module Mode 其原有的函数操作都会被替换成对应的Function Module的调用  
方法:  
torchpruner.activate_function_module()

#### 关闭Function Module Mode
关闭Function Module Mode,模型恢复正常执行  
方法:  
torchpruner.deactivate_function_module()  

#### 查看当前FUnction Module Mode是否开启
torchpruner提供了方法查看当前Function Module Mode的状态  
方法:  
torchpruner.function_module_activate()  
返回值:  
True或者False,True代表当前Function Module Mode开启,False则代表关闭  

#### 将模型转换为Function Module Mode形式
开启Function Module Mode后需要对已有的模型进行一次前向传播,才能在_function_module_list中创建Function Module对象.  
并对当前模型进行标记, 表明其创建了_function_module_list对象  
当Function Module Mode关闭后,模型即恢复原有的执行逻辑,不受_function_module_list注册的影响.  
方法:  
torchpruner.init_function_module(model,inputs)  
说明:  
根据指定输入为模型添加_function_module_list,并对模型进行标记,表明其function module已注册.  
参数:  
model:需要注册_function_module_list的模型  
inputs: 模型输入  
返回值:
已经注册_function_module_list的模型  

#### 额外说明:
在model tools中将与模型查找和替换一起,详细展示Function Module的用法  

### Model Tools 
Model Tools是一组函数,提供了对pytorch模型中模块的查找替换等操作,在使用示例中将和Function Module联合使用,部分展示将一个普通模型转换为一个伪量化模型  

#### 根据名称获取模型中的对象
方法:  
torchpruner.model_tools.get_object(model,name)  
介绍:  
根据名称,获取指定模块对象  
参数:  
model:pytorch模型  
name: 对象名称,由 self.attr1.attr2表示  
返回值:  
指定nn.Module对象  

使用示例:  
```python
#创建模型
model=nn.Sequential(
    nn.Conv2d(3,16,3,1,1),
    nn.BatchNorm2d(16),
    nn.Relu(),
    nn.Conv2d(16,10,3,1,1)
)
#获取第二个元素
bn=torchpruner.get_object(model,'self.1')
```

#### 根据名称设置对象  
方法:  
torchpruner.model_tools.set_object(model,name,nn_module)
介绍:  
根据对象名称,对模型设置新的对象,并返回设置对象后的模型  
参数:  
model:pytorch模型
name:对象名称,由self.attr1.attr2表示  
返回值:  
重新设定对象的模型  

使用示例:  
```python
#创建模型
class Net(nn.Module):
    def __init__(self):
        self.conv=nn.Conv2d(3,16,3,1,1)
        self.bn=nn.BatchNorm2d(16)
        self.activation=nn.Relu()
    
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.activation(x)
        return x

model=Net()
#替换激活函数为Sigmoid
bn=torchpruner.set_object(model,'self.activation',nn.Sigmoid())
```


#### 根据类别查找对象
方法:  
torchpruner.model_tools.get_names_by_class(model,object_class,include_super_class=True)  
介绍:  
根据类别,返回模型中所有的对象名,返回一个list  
参数:  
model:pytorch模型
object_class: 查找对象的类别  
include_super_class:  是否包含超类,如果包含,超类也会被查找  
返回值:  
所有的对象名称组成的list  

使用示例:  
```python
#创建模型
class Net(nn.Module):
    def __init__(self):
        self.conv_bn=nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.BatchNorm2d(16))
        self.activation=nn.Relu()
        self.conv2=nn.Conv2d(16,10,3,1,1)
    
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.activation(x)
        x=self.conv2(x)
        return x

model=Net()

# 查找所有的卷积
name_list=torchpruner.model_tools.get_names_by_class(model,nn.Conv2d,False)
# 结果为:['self.conv_bn.0','self.conv2']

#查找所有的模块
name_list=torchpruner.model_tools.get_names_by_class(model,nn.Module,True)
# 结果为:['self'] 当发现第一个符合要求的对象时即停止查找,不会查找其子元素

#查找为nn.Module的模块, include_super_class=False
name_list=torchpruner.model_tools.get_names_by_class(model,nn.Module,False)
#结果为[] 所有的模块其type都不为nn.Module
```


#### 根据类别列表查找对象
model_tools支持对一组顺序的类别结构进行查找(前一个模块的输出为下一个模块的输入),并返回二维的查找列表.  
即为所有的符合匹配结果的组,每个组存储了查找的多个模块的名称  
方法:  
torchpruner.model_tools.get_name_groups_by_classes(graph, object_class_list,include_super_class=True)  
介绍:  
根据静态图结构,和对象类列表,返回所有的符合要求的对象组的名称  
参数:  
graph:模型的ONNXGraph对象  
object_class_list:一个list,包含了若干个类  
include_super_class: 查找是否包含超类,默认为True  
返回值:  
二维的列表组,一维代表了所有的匹配结果,二维代表了被查找模块的名称.  

使用示例:   
```python
#创建模型
class Net(nn.Module):
    def __init__(self):
        self.conv_bn=nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.BatchNorm2d(16))
        self.activation=nn.Relu()
        self.conv2=nn.Conv2d(16,10,3,1,1)
    
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.activation(x)
        x=self.conv2(x)
        return x

model=Net()

#创建静态图结构
graph=torchpruner.ONNXGraph(model)
graph.build_graph((torch.zeros(1,3,32,32),))

# 查找conv bn relu结构
name_groups_list= torchpruner.model_tools.get_name_groups_by_classes(graph, [nn.Conv2d,nn.BatchNorm2d,nn.Relu])  
# 结果为[['self.conv_bn.0','self.conv_bn.1','self.activation']]
```


#### 根据名称列表替换对象
model_tools 支持根据名称将指定模块替换成其他模块，替换模块的方法使用一个replace_function来执行  
方法：  
torchpruner.model_tools.replace_object_by_names(model,name_list,replace_function)  
介绍：  
将model中的指定的名称列表的模块，根据replace_function的规则，替换成其他模块  
其中replace_function 的第一个参数为模块的名称，第二个参数为原有的对象，返回值为被替换成的模块对象  
参数：  
model: 需要被模块替换的模型  
name_list: 模块名称列表  
replace_function: 替换函数，由用户指定，参数形式为(module_name,origin_nn_module)->new_nn_module  
返回值：  
模块替换后的模型  

使用示例：  
```python
#创建模型
class Net(nn.Module):
    def __init__(self):
        self.conv_bn=nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.BatchNorm2d(16))
        self.activation=nn.Relu()
        self.conv2=nn.Conv2d(16,10,3,1,1)
    
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.activation(x)
        x=self.conv2(x)
        return x

model=Net()

# 查找所有的卷积结构
name_list=torchpruner.model_tools.get_names_by_class(model,nn.Conv2d,False)
#结果为 ['self.conv_bn.0','self.conv2']

#定义replace function
#将所有的卷积层，替换成深度可分离卷积
def replace_function(module_name,origin_module):
    #定义depthwise卷积
    depthwise_conv=nn.Conv2d(origin_module.in_channels,origin_module.in_channels,
    origin_module.kernel_size,origin_module.stride,origin_module.padding,groups=origin_module.groups)
    pointwise_conv=nn.Conv2d(origin_module.in_channels,origin_module.out_channels,1,1,0)
    return nn.Sequential(depthwise_conv,pointwise_conv)

#执行替换操作
model=torchpruner.model_tools.replace_object_by_names(model,name_list,replace_function)
```


#### 根据组名称列表替换满足要求的一组对象
与替换单个名称的nn.Module对应,model_tools提供了按照组的方式,替换一组对象.   
在操作逻辑上,往往先通过torchpruner.model_tools.get_name_groups_by_classes获取符合条件的组名称列表,然后再实现replace_function批量将需要替换的组模块进行替换  
方法:  
torchpruner.model_tools.replace_object_by_name_groups(model,group_name_list,replace_function)  
介绍:  
根据提供的一组模块的名称列表和replace_function将模型中对应名称的模块替换  
参数:  
model: 需要转换的pytorch模型  
group_name_list: 组名称列表,是一个二维的list.第一维代表了不同的组,第二维为该组中需要被替换的模块名称  
replace_function: 替换函数,第一个参数为替换模块的名称列表,第二个参数为原有的对象列表  
返回值:  
被替换后的模块  

使用示例:  
```python
#创建模型
class Net(nn.Module):
    def __init__(self):
        self.conv_bn=nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.BatchNorm2d(16))
        self.activation=nn.Relu()
        self.conv2=nn.Conv2d(16,10,3,1,1)
    
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.activation(x)
        x=self.conv2(x)
        return x

model=Net()

#创建静态图结构
graph=torchpruner.ONNXGraph(model)
graph.build_graph((torch.zeros(1,3,32,32),))

# 查找conv bn relu结构
name_groups_list= torchpruner.model_tools.get_name_groups_by_classes(graph, [nn.Conv2d,nn.BatchNorm2d,nn.Relu])  
# 结果为[['self.conv_bn.0','self.conv_bn.1','self.activation']]

#定义replace function
#将conv 与 bn融合
def replace_function(module_names,origin_modules):
    #定义depthwise卷积
    depthwise_conv=nn.Conv2d(origin_module.in_channels,origin_module.in_channels,
    origin_module.kernel_size,origin_module.stride,origin_module.padding,groups=origin_module.groups)
    pointwise_conv=nn.Conv2d(origin_module.in_channels,origin_module.out_channels,1,1,0)
    return merge_conv_bn(origin_modules[0],origin_modules[1]),nn.Identity(),origin_modules[2]

#执行替换操作
model=torchpruner.model_tools.replace_object_by_names(model,name_list,replace_function)

#融合函数如下:
def merge_conv_bn(conv:nn.Conv2d,bn:nn.BatchNorm2d):
    bn_weight=bn.weight
    bn_bias=bn.bias
    bn_running_mean=bn.running_mean
    bn_running_var=bn.running_var
    eps=bn.eps

    conv_weight=conv.weight
    conv_bias=conv.bias

    with torch.no_grad():
        factor=bn_weight/torch.sqrt(bn_running_var+eps)
    with torch.no_grad():
        conv_weight=conv_weight*factor.view(-1,1,1,1)
    if conv_bias is None:
        conv_bias=0
    with torch.no_grad():
        conv_bias=(conv_bias-bn_running_mean)*factor+bn_bias

    conv_merge=nn.Conv2d(conv.in_channels,conv.out_channels,conv.kernel_size,conv.stride,
    conv.padding,conv.dilation,conv.groups,bias=True)
    conv_merge.weight=nn.Parameter(conv_weight)
    conv_merge.bias=nn.Parameter(conv_bias)
    return conv_merge
```


##### 使用Function Module Mode和Model Tools 完成 pytorch->QAT->tensorrt
详细内容见torchslim/quantizing  


### DataNode Object
DataNode对象代表了静态图的数据节点，是Graph对象构成静态图的重要元素之一。  
DataNode对象代表了了静态图中数据节点输入数据，参数数据以及中间模块的计算数据结果。 
DataNode和OperatorNode相互连接构造了模型的静态图结构。  
下面对DataNode进行介绍

#### DataNode在何时会被访问
DataNode可能出现在以下几种情况中
1）通过ONNXGraph对象访问：  
```python
#使用属性nodes访问对象
for key in graph.nodes.keys():
    print(graph.nodes[key])

```
2)在Module模块中被访问：  
```python
#使用属性nodes访问对象
conv_module=graph.modules['self.conv']

#访问卷积的权重对应的DataNode
weight_node=conv_module.sub_modules['weight'].terminal_node

#访问卷积模块的输入 X：
input_node=conv_module.in_data[0]

#访问卷及模块的输出 Y:
output_node=conv_module.out_data[0]
```
3)在Operator模块中被访问：  
```python
#获取类型为卷积的第一个Operator
conv_operator=None
for key in graph.operators.keys():
    operator=graph.operators[key]
    if operator.type=='Conv':
        conv_operator=operator

#获取其三组输入DataNode，分别代表了X，weight，bias
X,weight,bias=conv_operator.in_data

#获取输出DataNode，代表Y
Y=conv_operator.out_data[0]
```

#### DataNode中的属性
##### data
data为一个Numpy存储了DataNode中的实际数据  
##### graph
当前DataNode属于的Graph
##### in_opeartor
in_operator为一个OperatorNode，代表了DataNode由什么数据节点产生  
其中模型的参数一般不包含in_operator,这种没有in_operator的数据节点被称为terminal_node，具有in_operator的数据节点为iner_node  
##### out_operators
out_operators是一组OperatorNode组成的List，代表了当前DataNode做输入被哪些其他的OperatorNode使用，一个模型的输出节点，没有out_operators  

#### DataNode中的方法
##### DataNode.cut_analysis
方法：
DataNode.cut_analysis(index,dim)  
说明：  
在当前数据节点上，假设去除dim列的index指定的若干个维度，分析模型其他数据节点和属性发生怎样的变化
参数：  
index: 一个整数组成的一维list，每一个数字代表了被去除的维度  
dim: 被去除index的维度标  
返回值：  
分析结果，见Module.cut_analysis

##### DataNode.size
方法:  
DataNode.size()
返回值：  
n个元素的tuple，代表了DataNode的N个维度的大小  

##### DataNode.type
方法：
DataNode.type()
返回值：  
DataNode数据的类型  

##### DataNode.is_terminal
方法:  
DataNode.is_terminal()  
描述：  
是否为terminal节点  
返回值：
True/False  

##### DataNode.is_input
方法:  
DataNode.is_input()  
描述：  
是否为输入节点  
返回值：
True/False  

##### DataNode.is_output
方法:  
DataNode.is_output()  
描述：  
是否为输出节点  
返回值：
True/False  


### OperatorNode Object
OperatorNode对象代表了静态图结构中的操作节点，是构成Graph节点的重要元素之一，Graph即由OperatorNodeData构成。  
OperatorNode可执行运算，并根据输入的DataNode节点，更新输出DataNode节点的数据。  

#### 如何获取OperatorNode节点;
OperatorNode可通过以下方法访问获取  
1）通过ONNXGraph对象访问：  
```python
#使用属性nodes访问对象
for key in graph.operators.keys():
    print(operators.nodes[key])

```
2)在Module模块中被访问：  
```python
#使用属性nodes访问对象
conv_module=graph.modules['self.conv']
conv_operator=conv_module.operators[0]
```
3)在DataNode模块中被访问    
```python
#获取类型为卷积的第一个Operator
conv_operator=None
for key in graph.nodes.keys():
    node=graph.nodes[key]
    print(node.in_operator)
    print(node.out_operators)
```

#### OperatorNode中的属性：
##### OperatorNode.name
当前OperatorNode的名称  
##### OperatorNode.type
当前OperatorNode的类型  
##### OperatorNode.protocal
当前OperatorNode的protocal，一般为ONNX  
##### OperatorNode.params
当前OperatorNode的参数表，为一个dict，key为参数名，value为数值  
##### OperatorNode.in_data
当前OperatorNode 的输入DataNode,为一个list  
##### OperatorNode.out_data
当前OperatorNode的输出DataNode，为一个list  

#### OperatorNode中的方法：  
##### OperatorNode.flops
计算当前Operator运算的计算量  
方法:  
OperatorNode.flops()  
返回值:  
当前OperatorNode的计算量  
##### OperatorNode.analysis
analysis是用于剪枝关系分析的重要函数,其接收一个DataNode和一个Mask对象, DataNode代表当前DataNode已经被剪枝.  
Mask对象则标记了DataNode对象的哪些位置的参数被去除, analysis函数则分析当当前DataNode的Mask标记的位置被去除时,其OperatorNode邻域的其他哪些DataNode发生怎样的参数变化,
这种参数变化,使用Mask标记.  
方法:  
analysis(node,mask)  
参数:  
node: 当前被剪枝的DataNode  
mask: 当前被剪枝DataNode的剪枝位置  
返回值:  
一个Dict,key是DataNode的名称,value是DataNode的剪枝标记Mask  
##### OperatorNode.fill_shape
根据输入DataNode的shape推断,输出DataNode的shape,一般在fill_value之前执行  
方法:  
OperatorNode.fill_shape()  
##### OperatorNode.fill_value
根据输入DataNode的值,计算输出DataNode的值  
方法:  
OperatorNode.fill_value()  

### Module Pruner注册  
Module Pruner是用于对模型进行剪枝操作的模块，Module Pruner以类别为单位规定了每一种nn.Module的剪枝处理操作规则．　　
torchpruner.set_cut/recovery_cut/set_zero/recovery_zero 函数均通多针对需要被剪枝的Module,调用Module Pruner模块完成剪枝操作.  
在torchpruner.set_cut/recovery_cut等操作的逻辑中,其依次自顶向下,对python model中的nn.Module进行遍历,如果当前nn.Module的类注册了对应的Module Pruner  
则调用对应的Module Pruner对当前模块进行剪枝. 由于torch.Tensor和nn.Parameter均进行了注册,因此无论如何模型的参数都可以被剪枝.  
但是部分模块,在更改参数的同时,还需要更改其部分属性,这样才能保证模型的合理运行. 例如针对depthwise卷积, 参数去除,其groups属性也要发生变化, 因此需要对groups属性进行相应的改变.  
因此针对用户自定义的需要修改模型属性的模块, torchpruner提供了Module Pruner的注册功能.  
当前已经有部分nn.Module注册了相应的Module Pruner,他们是:  
torch.Tensor  
nn.Parameter  
nn.Conv2d  
nn.Conv3d  
nn.BatchNorm2d  
nn.BatchNorm3d  
下面则详细说明如何注册一个自定义的nn.Module模块:  

#### Module Pruner编写过程中用到的常用函数
Module Pruner 在操作过程中涉及到了对模型的张量进行剪枝. 下面提供了若干对模型中的张量进行剪枝的方法  
##### torchpruner.module_pruner.set_cut_tensor
对类型为torch.Tensor的张量进行剪枝操作,返回被剪枝的张量  
方法:  
set_cut_tensor(tensor,cut_dims)  
参数:  
tensor:被剪枝的张量  
cut_dims: 是一个size与tensor的dim一致的list,如果当前dim不剪枝则为None,如果当前dim剪枝则为一个存储了需要被剪枝的index的list  
返回值:  
被剪枝完的张量, 被去除的张量的上下文  
##### torchpruner.module_pruner.recovery_cut_tensor
对被剪枝的张量根据参数上下文进行恢复  
方法:  
recovery_cut_tensor(tenosr,cut_dims,param_list)  
参数:  
tensor: 被剪枝后的张量  
cut_dims: 同set_cut_tensor  
param_list: 被去除的参数的上下文  
返回值:  
恢复后的张量
##### torchpruner.module_pruner.set_zero_tensor
同torchpruner.module_pruner.set_cut_tensor,不同点为此处将张量对应位置置0  
##### torchpruner.module_pruner.recovery_cut_tensor
同torchpruner.module_pruner.recovery_cut_tensor,不同点为将被置0的参数恢复

#### 自定义一个Module Pruner
下面以nn.Conv2d为例, 说明Module Pruner的自定义  
定义一个Module Punrer需要继承ModulePruner基类:  
```python
import torchpruner
import torchpruner.module_pruner as module_pruner

class Conv2dPruner(module_pruner.BasePruner):
    def __init__(self,name):
        super(Conv2dPruner).__init__(self,name)
```
BasePruner中包含了4个基本方法,分别对应了torchpruner 剪枝和恢复的四种操作:  
BasePruner.set_cut <-> torchpruner.set_cut  
BasePruner.recovery_cut <-> torchpruenr.recovery_cut  
BasePruner.set_zero <-> torchpruner.set_zero  
BasePruner.recovery_zero <-> torchpruner.recovery  
在剪枝的操作的过程中,执行不同的函数,即调用了对应BasePruner中的方法,在实际使用过程中,仅需要实现其中对应的方法即可  
例如 针对剪枝操作,仅需要实现BasePruner.set_cut接口,如果需要对模型进行恢复,则需要将BasePruner.recovery_cut接口实现  

下面实现Conv2dPruner的set_cut接口:
```python
import torchpruner
import torchpruner.module_pruner as module_pruner

class Conv2dPruner(module_pruner.BasePruner):
    def __init__(self,name):
        super(Conv2dPruner).__init__(self,name)
        #创建ParameterPruner对象,对weight和bias进行剪枝
        self.weight_pruner=ParameterPruner(name+".weight")
        self.bias_pruner=ParameterPruner(name+".bias")
    
    def set_cut(self,nn_module,cut_dict):
        #剪枝weight
        nn_module.weight,weight_context=self.weight_pruner.set_cut(nn_module.weight,cut_dict)
        #还可以直接使用set_cut_tensor剪枝
        #nn_module.weight,weight_context=module_pruner.set_cut_tensor(nn_module.weight,cut_dict)

        #剪枝bias
        nn_module.bias,bias_context=self.bias_pruner.set_cut(nn_module.bias,cut_dict)
        #还可以直接使用set_cut_tensor剪枝
        #nn_module.bias,bias_context=module_pruner.set_cut_tensor(nn_module.bias,cut_dict)

        #设置in_channels 属性
        nn_module.in_channels=nn_module.weight.data.size(1)
        #设置out_channels 属性
        nn_module.out_channels=nn_module.weight.data.size(0)
        #查找是否Conv Operator的属性发生变化,更改group参数
        onnx_name=self.name+".Conv"
        if onnx_name in cut_dict['operator']:
            ONNX_params=cut_dict['operator'][onnx_name]
            nn_module.groups-=ONNX_params["group"]
        return nn_module,{**weight_context,**bias_context}
```

#### Module Pruner注册
最后将定义的Module Pruner注册
```python
torchpruner.module_pruner_reg.regist(nn.Conv2d,Conv2dPruner)
```

### Operator 注册
OperatorNode是torchpruner实现剪枝关系分析和模型前向运算的基础.  
其剪枝关系分析和前向运算分别依赖于三个方法,分别为:  
Operator.analysis  
Operator.fill_shape  
Operator.fill_value  
其中Operator.analysis用于当前OperatorNode的剪枝关系分析,其接收一个DataNode和Mask,并返会该OperatorNode的邻域DataNode发生什么样的变化  
在torchpruner中, 常用的多组ONNX标准的OperatorNode都已经被注册,其中analysis函数负责依赖关系的分析,fill_shape和fill_value 函数则通过onnxruntime来填充DataNode的shape和value.  
然而在实际的情况中,依然会存在特殊的不常用的Operator,ONNX支持的Operator,用户自定义的Operator,在这种情况下则需要手动实现Operator的注册,从而使得Graph可以完成推理.  
下面以AveragePool为例(尽管AveragePool2d算子在torchpruner中已经被支持), 来展示如何注册一个算子:  
```python
import torchpruner
import torchpruner.operator

class AveragePool2dOperatorNode(torchpruner.operator.OperatorNode):
    def __init__(self,node):
        super(AveragePool2dOperatorNode,self).__init__(node)
    
    def analysis(self,node,mask):
        in_or_out,rank=self.rank(node)
        if in_or_out=='in':
            return {self.out_data[0].name:mask},None
        if in_or_out=='out':
            return {self.in_data[0].name:mask},None
    
    def fill_shape(self):
        input_size=self.in_data.size()
        stride=self.params['stride']
        padding=self.params['padding']
        output_size=input_size[:]
        output_size[2]=(output_size[2]+padding)//stride
        output_size[3]=(output_size[3]+padding)//stride
        self.out_data[0]._size=output_size
    
    def fill_value(self):
        import torch
        input_tensor=torch.tensor(self.in_data.data)
        output_tensor=F.avg_pool2d(input_tensor,self.params['stride'],self.params['padding'])
        self.out_data[0].data=output_tensor.numpy()
```
注册算子:  
```python
import torchpruner
torchpruner.operator_register.regist('AveragePool2d',AveragePool2dOperatorNode)
```
analysis函数是其中较为难以理解的部分,尤其是针对复杂的算子,analysis函数会非常复杂. 在大多数情况下,需要被注册的算子都可以规约为已有的OperatorNode的analysis函数的某种情况.  
在这种情况下可以使用已有的OperatorNode注册算子,例如AveragePool使用了onnx_operator.onnx_local_pool作为注册OperatorNode, 和AveragePool具有相同分析逻辑的MaxPool则可以使用相同的方式进行分析  
当前已有OperatorNode详见 torchpruner.operator.onnx_operator  
注册算子:  
```python
import torchpruner
import torchpruner.operator
torchpruner.operator_register.regist('AveragePool2d',torchpruner.operator.onnx_operator.onnx_local_pool)
```


