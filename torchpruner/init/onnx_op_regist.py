from torchpruner.register import operator_reg
from torchpruner.operator import onnx_operator

# The point wise onnx op
pointwise_ops = [
    "Abs",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Cast",
    "Ceil",
    "Reciprocal",
    "Clip",
    "Cos",
    "Cosh",
    "Dropout",
    "DynamicQuantizeLinear",
    "Elu",
    "Erf",
    "Exp",
    "Floor",
    "HardSigmoid",
    "Hardmax",
    "Identity",
    "IsInf",
    "IsNaN",
    "LeakyRelu",
    "Log",
    "LogSoftmax",
    "Neg",
    "NonZero",
    "Not",
    "Relu",
    "Round",
    "Selu",
    "Shrink",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Softmax",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tan",
    "Tanh",
    "Max",
    "Mean",
    "Min",
    "LpNormalization",
    "LRN",
    "MeanVarianceNormalization",
    "CumSum",
]
for op in pointwise_ops:
    operator_reg.regist(op, onnx_operator.onnx_pw)


## boardcast and pointwise
## just support n to 1
boardcast_pointwise_ops = [
    "Add",
    "And",
    "BitShift",
    "Div",
    "Equal",
    "Greater",
    "Less",
    "Mod",
    "Mul",
    "Sub",
    "Or",
    "Pow",
    "Sum",
    "Xor",
    "PRelu",
    "ThresholdedRelu",
]
for op in boardcast_pointwise_ops:
    operator_reg.regist(op, onnx_operator.onnx_bc_pw)

# no action operation
no_action_ops = [
    "Constant",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Range",
    "Shape",
    "Size",
    "MonMaxsuppression",
    "ConstantOfShape",
    "EyeLike",
    "Multinomial",
    "Gather",
    "GatherElements",
    "GatherND",
    "If",
    "Loop",
]
for op in no_action_ops:
    operator_reg.regist(op, onnx_operator.onnx_no_action)

# unsupport cut operation
unsupport_cut_ops = [
    "Det",
    "StringNormalizer" "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "SplitToSequence",
    "ReverseSequence",
    "ConcatFromSequence",
]
for op in unsupport_cut_ops:
    operator_reg.regist(op, onnx_operator.onnx_unsupport)

# ordinary reduce
ordinary_reduce_ops = [
    "ArgMax",
    "ArgMin",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
]
for op in ordinary_reduce_ops:
    operator_reg.regist(op, onnx_operator.onnx_reduce)

# global pooling reduce
global_pool_reduce_ops = ["GlobalLpPool", "GlobalAveragePool", "GlobalMaxPool"]
for op in global_pool_reduce_ops:
    operator_reg.regist(op, onnx_operator.onnx_GlobalPool)

# Conv like operation
conv_ops = ["Conv", "ConvTranspose", "ConvInteger"]
for op in conv_ops:
    operator_reg.regist(op, onnx_operator.onnx_conv)

# local pool
local_pool_ops = ["AveragePool", "MaxPool", "MaxUnPool", "LpPool", "Upsample"]
for op in local_pool_ops:
    operator_reg.regist(op, onnx_operator.onnx_local_pool)

##onnx mapping
# BatchNormalization GRU InstanceNormalization
# LSTM MatMul MatMulInteger
# QuantizeLinear DequantizeLinear RNN
# RoiAlign MaxRoiPool
operator_reg.regist("GRU", onnx_operator.onnx_GRU)
operator_reg.regist("LSTM", onnx_operator.onnx_LSTM)
operator_reg.regist("RNN", onnx_operator.onnx_LSTM)
operator_reg.regist("BatchNormalization", onnx_operator.onnx_BatchNormalization)
operator_reg.regist("InstanceNormalization", onnx_operator.onnx_InstanceNormalization)
operator_reg.regist("MatMul", onnx_operator.onnx_MatMul)
operator_reg.regist("MatMulInteger", onnx_operator.onnx_MatMulInteger)
operator_reg.regist("RoiAlign", onnx_operator.onnx_RoiAlign)
operator_reg.regist("MaxRoiPool", onnx_operator.onnx_MaxRoiPool)
operator_reg.regist("QuantizeLinear", onnx_operator.onnx_QDQ)
operator_reg.regist("DequantizeLinear", onnx_operator.onnx_QDQ)

# complex ops
operator_reg.regist("Concate", onnx_operator.onnx_Concat)
operator_reg.regist("Gemm", onnx_operator.onnx_Gemm)
operator_reg.regist("Concat", onnx_operator.onnx_Concat)
operator_reg.regist("Split", onnx_operator.onnx_Split)
operator_reg.regist("Squeeze", onnx_operator.onnx_Squeeze)
operator_reg.regist("Unsqueeze", onnx_operator.onnx_Unsqueeze)
operator_reg.regist("Expand", onnx_operator.onnx_Expand)
operator_reg.regist("Flatten", onnx_operator.onnx_Flatten)
operator_reg.regist("Reshape", onnx_operator.onnx_Reshape)
operator_reg.regist("Slice", onnx_operator.onnx_Slice)
operator_reg.regist("Resize", onnx_operator.onnx_Resize)
operator_reg.regist("Pad", onnx_operator.onnx_Pad)
operator_reg.regist("Transpose", onnx_operator.onnx_Transpose)
operator_reg.regist("TopK", onnx_operator.onnx_TopK)
operator_reg.regist("Compress", onnx_operator.onnx_Compress)
operator_reg.regist("SpaceToDepth", onnx_operator.onnx_SpaceToDepth)
operator_reg.regist("DepthToSpace", onnx_operator.onnx_DepthToSpace)
operator_reg.regist("Where", onnx_operator.onnx_where)
operator_reg.regist("ScatterElements", onnx_operator.onnx_ScatterElements)
