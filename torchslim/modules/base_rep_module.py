import torch
import torch.nn as nn
import numpy as np

__all__ = ("RepModule", "Compactor", "ModuleCompactor", "return_arg0")


class RepModule(nn.Module):
    def __init__(self):
        super(RepModule, self).__init__()

    @staticmethod
    def deploy(name, module=None):
        module = module if module is not None and isinstance(module, RepModule) else name
        return module.convert()

    def convert(self):
        raise NotImplementedError("Lack RepModule::convert")

    def forward(self, *args):
        raise NotImplementedError("Lack RepModule::forward")


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


def return_arg0(x):
    return x
