from tokenize import group
import torch.nn as nn
import torch.nn.init as init
import torch
from .base_rep_module import RepModule, return_arg0


class ACBCorner(RepModule):
    def __init__(
        self,
        conv: nn.Conv2d = None,
        in_channels=-1,
        out_channels=-1,
        kernel_size=-1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        deploy=False,
        use_affine=True,
        non_linear=None,
        with_bn=True,
    ):
        super(ACBCorner, self).__init__()
        if conv is not None:
            in_channels = conv.in_channels
            out_channels = conv.out_channels
            kernel_size = conv.kernel_size
            if isinstance(kernel_size, tuple):
                assert kernel_size[0] == kernel_size[1]
                kernel_size = kernel_size[0]
            stride = conv.stride
            padding = conv.padding
            if isinstance(padding, tuple):
                for i in padding[1:]:
                    assert padding[0] == i
                padding = padding[0]
            dilation = conv.dilation
            if isinstance(dilation, tuple):
                for i in dilation[1:]:
                    assert dilation[0] == i
                dilation = dilation[0]
            groups = conv.groups
            bias = conv.bias is not None
            padding_mode = conv.padding_mode
        else:
            assert in_channels > 0
            assert out_channels > 0
            assert kernel_size > 0
            bias = bias is not None and bias is not False

        assert kernel_size == 3  # only designed for 3x3 conv
        self.with_bn = with_bn
        self.fused_conv = None
        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )
        else:
            use_bias_in_conv = bias and not with_bn
            self.square_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=use_bias_in_conv,
                padding_mode=padding_mode,
            )
            self.square_bn = (
                nn.BatchNorm2d(num_features=out_channels, affine=use_affine) if self.with_bn else return_arg0
            )

            if padding - kernel_size // 2 >= 0:
                # Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                # Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust
                # to align the sliding windows (Fig 2 in the paper).
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                # A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                # Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                stride=stride,
                padding=ver_padding,
                dilation=dilation,
                groups=groups,
                bias=use_bias_in_conv,
                padding_mode=padding_mode,
            )

            self.hor_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=hor_padding,
                dilation=dilation,
                groups=groups,
                bias=use_bias_in_conv,
                padding_mode=padding_mode,
            )
            self.ver_bn = (
                nn.BatchNorm2d(num_features=out_channels, affine=use_affine) if self.with_bn else return_arg0
            )
            self.hor_bn = (
                nn.BatchNorm2d(num_features=out_channels, affine=use_affine) if self.with_bn else return_arg0
            )

            self.corner_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(2, 2),
                stride=stride,
                padding=padding,
                dilation=2 * dilation,
                groups=groups,
                bias=use_bias_in_conv,
                padding_mode=padding_mode,
            )
            self.corner_bn = (
                nn.BatchNorm2d(num_features=out_channels, affine=use_affine) if self.with_bn else return_arg0
            )

            if non_linear is None:
                self.non_linear = return_arg0
            else:
                self.non_linear = non_linear

            if not self.with_bn:
                # IMPORTANT
                # CANNOT BE IGNORED
                self.reduce_init_values()

    def reduce_init_values(self):
        with torch.no_grad():
            for i in [
                self.square_conv,
                self.ver_conv,
                self.hor_conv,
                self.corner_conv,
            ]:
                i.weight.data *= 1 / 4
                if i.bias is not None:
                    i.bias.data *= 1 / 4

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[
            :,
            :,
            square_h // 2 - asym_h // 2 : square_h // 2 - asym_h // 2 + asym_h,
            square_w // 2 - asym_w // 2 : square_w // 2 - asym_w // 2 + asym_w,
        ] += asym_kernel

    def get_equivalent_kernel_bias(self):
        if self.with_bn:
            hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
            ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
            cor_k, cor_b = self._fuse_bn_tensor(self.corner_conv, self.corner_bn)
            square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        else:
            arr = [
                [i.weight.data, i.bias.data if i.bias is not None else None]
                for i in [
                    self.hor_conv,
                    self.ver_conv,
                    self.corner_conv,
                    self.square_conv,
                ]
            ]
            hor_k, hor_b = arr[0]
            ver_k, ver_b = arr[1]
            cor_k, cor_b = arr[2]
            square_k, square_b = arr[3]
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        square_k[:, :, ::2, ::2] += cor_k
        return square_k, hor_b + ver_b + cor_b + square_b if hor_b is not None else None

    def convert(self):
        if self.fused_conv is not None:
            return self.fused_conv
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.square_conv.in_channels,
            out_channels=self.square_conv.out_channels,
            kernel_size=self.square_conv.kernel_size,
            stride=self.square_conv.stride,
            padding=self.square_conv.padding,
            dilation=self.square_conv.dilation,
            groups=self.square_conv.groups,
            bias=deploy_b is not None,
            padding_mode=self.square_conv.padding_mode,
        )
        self.__delattr__("square_conv")
        self.__delattr__("square_bn")
        self.__delattr__("hor_conv")
        self.__delattr__("hor_bn")
        self.__delattr__("ver_conv")
        self.__delattr__("ver_bn")
        self.__delattr__("corner_conv")
        self.__delattr__("corner_bn")
        self.fused_conv.weight.data = deploy_k
        if deploy_b is not None:
            self.fused_conv.bias.data = deploy_b
        return self.fused_conv

    def forward(self, input):
        if self.fused_conv is not None:
            return self.non_linear(self.fused_conv(input))
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop : -self.crop]
                hor_input = input[:, :, self.crop : -self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            corner_outputs = self.corner_conv(input)
            corner_outputs = self.corner_bn(corner_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs + corner_outputs
            return self.non_linear(result)
