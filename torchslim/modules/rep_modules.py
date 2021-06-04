import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import math

# conv bn merge function
def merge_conv_bn(conv, bn):
    bn_weight = bn.weight
    bn_bias = bn.bias
    bn_running_mean = bn.running_mean
    bn_running_var = bn.running_var
    eps = bn.eps

    conv_weight = conv.weight
    conv_bias = conv.bias

    with torch.no_grad():
        factor = bn_weight / torch.sqrt(bn_running_var + eps)
    with torch.no_grad():
        conv_weight = conv_weight * factor.view(-1, 1, 1, 1)
    if conv_bias is None:
        conv_bias = 0
    with torch.no_grad():
        conv_bias = (conv_bias - bn_running_mean) * factor + bn_bias

    conv_merge = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
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
        conv_weight = torch.sum(
            conv_weight.unsqueeze(0) * compactor_weight.unsqueeze(2), dim=1
        )
        if conv_bias != None:
            conv_bias = torch.matmul(
                compactor_weight.view(
                    compactor_weight.size(0), compactor_weight.size(1)
                ),
                conv_bias.unsqueeze(1),
            ).view(-1)
    # Transpose Condition
    if isinstance(conv, nn.ConvTranspose2d):
        conv_weight = conv_weight.transpose(1, 0)

    conv.weight.data = conv_weight.data
    if conv.bias != None:
        conv.bias.data = conv_bias.data
    else:
        conv.bias = None
    conv.out_channels = conv.weight.size(0)
    return conv


class RepModule(nn.Module):
    def __init__(self):
        super(RepModule, self).__init__()


class ACNet_CR(RepModule):
    def __init__(self, conv: nn.Conv2d):
        super(ACNet_CR, self).__init__()
        self.conv = None
        if conv.kernel_size == 1:
            self.conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                conv.bias,
            )
        else:
            self.conv_full = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                False,
            )
            self.bn_full = nn.BatchNorm2d(conv.out_channels)
            self.conv_row = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                (1, conv.kernel_size[1]),
                conv.stride,
                (0, conv.padding[1]),
                conv.dilation,
                conv.groups,
                False,
            )
            self.bn_row = nn.BatchNorm2d(conv.out_channels)
            self.conv_col = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                (conv.kernel_size[0], 1),
                conv.stride,
                (conv.padding[0], 0),
                conv.dilation,
                conv.groups,
                False,
            )
            self.bn_col = nn.BatchNorm2d(conv.out_channels)
            torch.nn.init.constant_(self.bn_full.weight, 0.33)
            torch.nn.init.constant_(self.bn_row.weight, 0.33)
            torch.nn.init.constant_(self.bn_col.weight, 0.33)

    def forward(self, x):
        if self.conv is not None:
            return self.conv(x)
        else:
            filter_full = self.bn_full(self.conv_full(x))
            filter_col = self.bn_col(self.conv_col(x))
            filter_row = self.bn_row(self.conv_row(x))
            return filter_full + filter_col + filter_row

    def convert(self):
        if self.conv is not None:
            return self.conv
        else:
            conv_col = merge_conv_bn(self.conv_col, self.bn_col)
            conv_row = merge_conv_bn(self.conv_row, self.bn_row)
            conv_full = merge_conv_bn(self.conv_full, self.bn_full)
            with torch.no_grad():
                bias = conv_col.bias + conv_row.bias + conv_full.bias
            with torch.no_grad():
                weight = conv_full.weight
                weight[:, :, weight.size(2) // 2, :] += conv_row.weight[:, :, 0, :]
                weight[:, :, :, weight.size(3) // 2] += conv_col.weight[:, :, :, 0]
            conv_full.weight = nn.Parameter(weight)
            conv_full.bias = nn.Parameter(bias)
            return conv_full


class Compactor(nn.Module):
    def __init__(self, num_features):
        super(Compactor, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        identity_mat = np.eye(num_features, dtype=np.float32)
        self.conv.weight.data.copy_(
            torch.from_numpy(identity_mat).reshape(num_features, num_features, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)


class ModuleCompactor(nn.Module):
    def __init__(self, module):
        super(ModuleCompactor, self).__init__()
        self.module = module
        if isinstance(module, nn.BatchNorm2d):
            self.compactor = Compactor(self.module.num_features)
            return
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            self.compactor = Compactor(self.module.out_channels)
            return
        raise RuntimeError("Unsupport type for compactor " + str(type(self.module)))

    def forward(self, x):
        x = self.module(x)
        x = self.compactor(x)
        return x


class weight_norm(nn.Module):
    def __init__(self, channels, init_value=0.1, dim=(1, 2, 3)):
        super(weight_norm, self).__init__()
        self.scale = nn.Parameter(torch.ones(channels) * init_value)
        self.dim = dim

    def forward(self, kernel):
        var = torch.norm(kernel, p=2, dim=self.dim)
        return (self.scale / var).view(-1, 1, 1, 1) * kernel


class Efficient_ACNet_CR(RepModule):
    def __init__(self, conv: nn.Conv2d):
        super(Efficient_ACNet_CR, self).__init__()
        self.conv = None
        init_value = 0.1
        if conv.kernel_size == 1:
            self.conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                conv.bias,
            )
        else:
            self.stride = conv.stride
            self.padding = conv.padding
            self.groups = conv.groups
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
        if self.conv is not None:
            return self.conv(x)
        kernel = self.weight_norm_full(self.kernel_full)
        kernel[
            :, :, :, kernel.size(3) // 2 : (kernel.size(3) + 1) // 2
        ] += self.weight_norm_row(self.kernel_row)
        kernel[
            :, :, kernel.size(2) // 2 : (kernel.size(2) + 1) // 2, :
        ] += self.weight_norm_col(self.kernel_col)
        # kernel[:,:,kernel.size(2)//2:(kernel.size(2)+1)//2,kernel.size(3)//2:(kernel.size(3)+1)//2]+=self.weight_norm_single(self.kernel_single)
        result = F.conv2d(
            x, kernel, self.bias, self.stride, self.padding, groups=self.groups
        )
        return result
