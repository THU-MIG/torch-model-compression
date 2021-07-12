import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16,32,3, 1, 1)
        self.conv3 = nn.Conv2d(32,16,3,1,1)
        self.conv4 = nn.Conv2d(16,32,1,1)
        self.conv5=nn.Conv2d(32,10,1)
        # self.upsample=nn.UpsamplingNearest2d(size=32)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2=F.relu(self.conv2(F.avg_pool2d(out1,2)))
        out3=F.relu(self.conv3(nn.functional.interpolate(out2,size=(int(32),int(32)),mode='nearest')))
        # out=torch.cat([out1,out3],dim=1)
        out=self.conv4(out3)
        out=F.avg_pool2d(out,32)
        out=self.conv5(out)
        return out