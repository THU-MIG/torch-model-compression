import torch
import torch.nn as nn
from torchpruner.register import module_pruner_reg
from torchpruner.module_pruner import (
    TensorPruner,
    ParameterPruner,
    BNPruner,
    ConvPruner,
    LinearPruner,
)

# op module
module_pruner_reg.regist(nn.Conv1d, ConvPruner)
module_pruner_reg.regist(nn.Conv2d, ConvPruner)
module_pruner_reg.regist(nn.Conv3d, ConvPruner)
module_pruner_reg.regist(nn.Parameter, ParameterPruner)
module_pruner_reg.regist(torch.Tensor, TensorPruner)
module_pruner_reg.regist(nn.BatchNorm1d, BNPruner)
module_pruner_reg.regist(nn.BatchNorm2d, BNPruner)
module_pruner_reg.regist(nn.BatchNorm3d, BNPruner)
module_pruner_reg.regist(nn.Linear, LinearPruner)
