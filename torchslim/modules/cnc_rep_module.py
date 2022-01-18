import torch
import torch.nn as nn
from .rep_modules import RepModule, return_arg0


def conv_bn(
    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros"
):
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=False,
        padding_mode=padding_mode,
    )
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module("conv", conv_layer)
    se.add_module("bn", bn_layer)
    return se


class CnCRep(RepModule):
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
        deploy=False,
        non_linear=None,
        with_bn=True,
    ):
        super(CnCRep, self).__init__()
        self.with_bn = with_bn
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

        if non_linear is None:
            self.non_linear = return_arg0
        else:
            self.non_linear = non_linear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert kernel_size % 2 == 1

        self.fused_conv = None
        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )

        else:
            use_bias_in_conv = bias and not with_bn
            if padding >= 1:
                assert padding_mode == "zeros"
                self.pad = nn.ZeroPad2d(padding=(padding, padding, padding, padding))
            else:
                self.pad = nn.Identity()
            if self.with_bn:
                self.cnc_origin = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    padding_mode=padding_mode,
                )
            else:
                self.cnc_origin = nn.Sequential()
                self.cnc_origin.add_module(
                    "conv",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=use_bias_in_conv,
                        padding_mode=padding_mode,
                    ),
                )

            self.cnc_left_up = nn.Sequential()
            self.cnc_left_up.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=int((kernel_size + 1) / 2),
                    stride=stride,
                    padding=0,
                    dilation=dilation,
                    groups=groups,
                    bias=use_bias_in_conv,
                    padding_mode=padding_mode,
                ),
            )
            if self.with_bn:
                self.cnc_left_up.add_module("bn", nn.BatchNorm2d(out_channels))

            self.cnc_left_down = nn.Sequential()
            self.cnc_left_down.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=int((kernel_size + 1) / 2),
                    stride=stride,
                    padding=0,
                    dilation=dilation,
                    groups=groups,
                    bias=use_bias_in_conv,
                    padding_mode=padding_mode,
                ),
            )
            if self.with_bn:
                self.cnc_left_down.add_module("bn", nn.BatchNorm2d(out_channels))

            self.cnc_right_up = nn.Sequential()
            self.cnc_right_up.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=int((kernel_size + 1) / 2),
                    stride=stride,
                    padding=0,
                    dilation=dilation,
                    groups=groups,
                    bias=use_bias_in_conv,
                    padding_mode=padding_mode,
                ),
            )
            if self.with_bn:
                self.cnc_right_up.add_module("bn", nn.BatchNorm2d(out_channels))

            self.cnc_right_down = nn.Sequential()
            self.cnc_right_down.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=int((kernel_size + 1) / 2),
                    stride=stride,
                    padding=0,
                    dilation=dilation,
                    groups=groups,
                    bias=use_bias_in_conv,
                    padding_mode=padding_mode,
                ),
            )
            if self.with_bn:
                self.cnc_right_down.add_module("bn", nn.BatchNorm2d(out_channels))

            self.cnc_center = nn.Sequential()
            self.cnc_center.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    dilation=dilation,
                    groups=groups,
                    bias=use_bias_in_conv,
                    padding_mode=padding_mode,
                ),
            )
            if self.with_bn:
                self.cnc_center.add_module("bn", nn.BatchNorm2d(out_channels))

            self.cnc_center_crop = (kernel_size - 1) // 2 * dilation

            if not self.with_bn:
                # IMPORTANT
                # CANNOT BE IGNORED
                self.reduce_init_values()

    def reduce_init_values(self):
        with torch.no_grad():
            for i in [
                self.cnc_origin,
                self.cnc_center,
                self.cnc_left_up,
                self.cnc_left_down,
                self.cnc_right_up,
                self.cnc_right_down,
            ]:
                i.conv.weight.data *= 1 / 6
                if i.conv.bias is not None:
                    i.conv.bias.data *= 1 / 6

    def forward(self, inputs):
        if self.fused_conv is not None:
            return self.non_linear(self.fused_conv(inputs))
        inputs_pad = self.pad(inputs)
        out = self.cnc_origin(inputs)
        out += self.cnc_left_up(inputs_pad[:, :, : -self.cnc_center_crop, : -self.cnc_center_crop])
        out += self.cnc_left_down(inputs_pad[:, :, self.cnc_center_crop :, : -self.cnc_center_crop])
        out += self.cnc_right_up(inputs_pad[:, :, : -self.cnc_center_crop, self.cnc_center_crop :])
        out += self.cnc_right_down(inputs_pad[:, :, self.cnc_center_crop :, self.cnc_center_crop :])
        out += self.cnc_center(
            inputs_pad[
                :, :, self.cnc_center_crop : -self.cnc_center_crop, self.cnc_center_crop : -self.cnc_center_crop
            ]
        )
        return self.non_linear(out)

    def get_equivalent_kernel_bias(self):
        merge_dict = {
            "cnc_origin": [0, 0],
            "cnc_left_up": [0, 0],
            "cnc_left_down": [self.kernel_size // 2, 0],
            "cnc_right_up": [0, self.kernel_size // 2],
            "cnc_right_down": [self.kernel_size // 2, self.kernel_size // 2],
            "cnc_center": [self.kernel_size // 2, self.kernel_size // 2],
        }
        kernel = torch.zeros_like(self.cnc_origin.conv.weight.data)
        bias = torch.zeros(self.cnc_origin.conv.weight.data.shape[0])
        if self.with_bn:
            for k, v in merge_dict.items():
                t = self.__getattr__(k)
                d, _, k1, k2 = t.conv.weight.data.shape
                bn_mean, bn_sigma, bn_gamma, bn_beta = (
                    t.bn.running_mean,
                    (t.bn.running_var + t.bn.eps).sqrt(),
                    t.bn.weight,
                    t.bn.bias,
                )
                kernel[:, :, v[0] : v[0] + k1, v[1] : v[1] + k2] += (
                    t.conv.weight.data * bn_gamma.reshape(d, 1, 1, 1) / bn_sigma.reshape(d, 1, 1, 1)
                )
                bias += bn_beta - bn_gamma * bn_mean / bn_sigma
        else:
            for k, v in merge_dict.items():
                t = self.__getattr__(k)
                d, _, k1, k2 = t.conv.weight.data.shape
                kernel[:, :, v[0] : v[0] + k1, v[1] : v[1] + k2] += t.conv.weight.data
                if t.conv.bias is not None:
                    bias += t.conv.bias.data
        return kernel, bias

    def convert(self):
        if self.fused_conv is not None:
            return self.fused_conv
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.cnc_origin.conv.in_channels,
            out_channels=self.cnc_origin.conv.out_channels,
            kernel_size=self.cnc_origin.conv.kernel_size,
            stride=self.cnc_origin.conv.stride,
            padding=self.cnc_origin.conv.padding,
            dilation=self.cnc_origin.conv.dilation,
            groups=self.cnc_origin.conv.groups,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("cnc_origin")
        self.__delattr__("cnc_left_up")
        self.__delattr__("cnc_left_down")
        self.__delattr__("cnc_right_up")
        self.__delattr__("cnc_right_down")
        self.__delattr__("cnc_center")
        return self.fused_conv
