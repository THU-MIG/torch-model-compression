import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic.qat as nniqat
import numpy as np
import torch.nn.init as init
import math
from .base_rep_module import RepModule, Compactor, return_arg0

# conv bn merge function
def merge_conv_bn(conv, bn):
    if bn is None:
        return conv
    bn_weight = bn.weight
    bn_bias = bn.bias
    bn_running_mean = bn.running_mean
    bn_running_var = bn.running_var
    eps = bn.eps

    conv_weight = conv.weight
    conv_bias = conv.bias
    # Transpose Condition
    is_deconv = isinstance(conv, nn.ConvTranspose2d)
    with torch.no_grad():
        factor = bn_weight / torch.sqrt(bn_running_var + eps)
        if is_deconv:
            conv_weight = conv_weight * factor.view(1, -1, 1, 1)
        else:
            conv_weight = conv_weight * factor.view(-1, 1, 1, 1)
        if conv_bias is None:
            conv_bias = 0
        conv_bias = (conv_bias - bn_running_mean) * factor + bn_bias

    float_class = conv.__class__
    qat_classes = (nniqat.ConvBn1d, nniqat.ConvBn2d, nniqat.ConvBn3d)
    if isinstance(conv, qat_classes):
        while float_class is not None and float_class not in qat_classes:
            float_class = getattr(float_class, "__base__", None)
        if float_class is not None:
            float_class = float_class.__bases__[1]
    if float_class is None:
        float_class = nn.ConvTranspose2d if is_deconv else nn.Conv2d

    conv_merge = float_class(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
        **({ "output_padding": conv.output_padding } if is_deconv else {}),
    )
    conv_merge.weight = nn.Parameter(conv_weight)
    conv_merge.bias = nn.Parameter(conv_bias)
    return conv_merge


def merge_conv_compactor(conv, compactor):
    conv_weight = conv.weight
    conv_bias = conv.bias
    compactor_weight = compactor.conv.weight

    # Transpose Condition
    if isinstance(conv, nn.ConvTranspose2d):
        conv_weight = conv_weight.transpose(1, 0)

    with torch.no_grad():
        reshaped_compactor_weight = compactor_weight.reshape(
            compactor_weight.size(0), compactor_weight.size(1)
        )
        conv_weight = torch.matmul(
            reshaped_compactor_weight,
            conv_weight.reshape(conv_weight.shape[0], -1),
        ).reshape((compactor_weight.size(0), *conv_weight.shape[1:]))
        if conv_bias is not None:
            conv_bias = torch.matmul(
                reshaped_compactor_weight,
                conv_bias.unsqueeze(1),
            ).view(-1)
    # Transpose Condition
    if isinstance(conv, nn.ConvTranspose2d):
        conv_weight = conv_weight.transpose(1, 0)

    conv.weight.data = conv_weight.data
    if conv.bias is not None:
        conv.bias.data = conv_bias.data
    conv.out_channels = conv.weight.size(0)
    return conv

class ACNet_CR(RepModule):
    def __init__(self, conv: nn.Conv2d, deploy=False, with_bn=True):
        super(ACNet_CR, self).__init__()
        self.with_bn = with_bn
        self.fused_conv = None
        if deploy or conv.kernel_size == 1:
            self.fused_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                conv.bias is not None,
                conv.padding_mode,
            )
        else:
            use_bias_in_conv = conv.bias is not None and not with_bn
            self.conv_full = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                use_bias_in_conv,
                conv.padding_mode,
            )
            self.bn_full = nn.BatchNorm2d(conv.out_channels) if with_bn else return_arg0
            self.conv_row = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                (1, conv.kernel_size[1]),
                conv.stride,
                (0, conv.padding[1]),
                conv.dilation,
                conv.groups,
                use_bias_in_conv,
                conv.padding_mode,
            )
            self.bn_row = nn.BatchNorm2d(conv.out_channels) if with_bn else return_arg0
            self.conv_col = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                (conv.kernel_size[0], 1),
                conv.stride,
                (conv.padding[0], 0),
                conv.dilation,
                conv.groups,
                use_bias_in_conv,
                conv.padding_mode,
            )
            self.bn_col = nn.BatchNorm2d(conv.out_channels) if with_bn else return_arg0
        if with_bn and self.fused_conv is None:
            torch.nn.init.constant_(self.bn_full.weight, 0.33)
            torch.nn.init.constant_(self.bn_row.weight, 0.33)
            torch.nn.init.constant_(self.bn_col.weight, 0.33)

    def forward(self, x):
        if self.fused_conv is not None:
            return self.fused_conv(x)
        else:
            filter_full = self.bn_full(self.conv_full(x))
            filter_col = self.bn_col(self.conv_col(x))
            filter_row = self.bn_row(self.conv_row(x))
            return filter_full + filter_col + filter_row

    def convert(self):
        if self.fused_conv is not None:
            return self.fused_conv
        else:
            conv_col = merge_conv_bn(self.conv_col, self.bn_col if self.with_bn else None)
            conv_row = merge_conv_bn(self.conv_row, self.bn_row if self.with_bn else None)
            conv_full = merge_conv_bn(self.conv_full, self.bn_full if self.with_bn else None)
            bias = None
            with torch.no_grad():
                if conv_full.bias is not None:
                    bias = conv_col.bias + conv_row.bias + conv_full.bias
            with torch.no_grad():
                weight = conv_full.weight
                weight[:, :, weight.size(2) // 2, :] += conv_row.weight[:, :, 0, :]
                weight[:, :, :, weight.size(3) // 2] += conv_col.weight[:, :, :, 0]
            conv_full.weight = nn.Parameter(weight)
            if conv_full.bias is not None:
                conv_full.bias = nn.Parameter(bias)
            return conv_full


class weight_norm(nn.Module):
    def __init__(self, channels, init_value=0.1, dim=(1, 2, 3)):
        super(weight_norm, self).__init__()
        self.scale = nn.Parameter(torch.ones(channels) * init_value)
        self.dim = dim

    def forward(self, kernel):
        var = torch.norm(kernel, p=2, dim=self.dim)
        return (self.scale / var).view(-1, 1, 1, 1) * kernel


class Efficient_ACNet_CR(RepModule):
    def __init__(self, conv: nn.Conv2d, deploy=False, with_bn=True):
        super(Efficient_ACNet_CR, self).__init__()
        self.fused_conv = None
        init_value = 0.1
        assert with_bn == True
        if deploy or conv.kernel_size == 1:
            self.fused_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                conv.bias,
                conv.padding_mode,
            )
        else:
            self.stride = conv.stride
            self.padding = conv.padding
            self.groups = conv.groups
            self.dilation = conv.dilation
            self.kernel_full = nn.Parameter(
                torch.randn(
                    (
                        conv.out_channels,
                        conv.in_channels // self.groups,
                        conv.kernel_size[0],
                        conv.kernel_size[1],
                    )
                )
            )
            self.kernel_col = nn.Parameter(
                torch.randn(
                    (
                        conv.out_channels,
                        conv.in_channels // self.groups,
                        1,
                        conv.kernel_size[1],
                    )
                )
            )
            self.kernel_row = nn.Parameter(
                torch.randn(
                    (
                        conv.out_channels,
                        conv.in_channels // self.groups,
                        conv.kernel_size[0],
                        1,
                    )
                )
            )
            self.init_parameters(self.kernel_col)
            self.init_parameters(self.kernel_row)
            self.init_parameters(self.kernel_full)
            self.weight_norm_full = weight_norm(conv.out_channels, init_value)
            self.weight_norm_row = weight_norm(conv.out_channels, init_value)
            self.weight_norm_col = weight_norm(conv.out_channels, init_value)
            # self.kernel_single=nn.Parameter(torch.randn((conv.out_channels,conv.in_channels//self.groups,1,1)))
            # self.weight_norm_single=weight_norm(conv.out_channels,init_value,running_mean,norm_type)
            self.bias = None
            if conv.bias is not None:
                self.bias = nn.Parameter(torch.zeros(conv.out_channels))

    def init_parameters(self, kernel):
        # init.xavier_uniform_(kernel)
        # init._no_grad_normal_(kernel,0,0.05)
        pass

    def forward(self, x):
        if self.fused_conv is not None:
            return self.fused_conv(x)
        kernel = self.weight_norm_full(self.kernel_full)
        kernel[
            :, :, :, kernel.size(3) // 2 : (kernel.size(3) + 1) // 2
        ] += self.weight_norm_row(self.kernel_row)
        kernel[
            :, :, kernel.size(2) // 2 : (kernel.size(2) + 1) // 2, :
        ] += self.weight_norm_col(self.kernel_col)
        # kernel[:,:,kernel.size(2)//2:(kernel.size(2)+1)//2,kernel.size(3)//2:(kernel.size(3)+1)//2]+=self.weight_norm_single(self.kernel_single)
        result = F.conv2d(
            x, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return result

    def convert(self):
        pass
