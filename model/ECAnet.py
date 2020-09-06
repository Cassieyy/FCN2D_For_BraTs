import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
class ECAlayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECAlayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 不管什么样的输入 输出得到1*1*C的tensor
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
    
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
 
class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class ECAnet(nn.Module):
    def __init__(self):
        in_chan = 4
        out_chan = 3
        super(ECAnet, self).__init__()
        self.down1 = Downsample_block(in_chan, 64)
        self.ECAlayer1 = ECAlayer(64)
        self.down2 = Downsample_block(64, 128)
        self.ECAlayer2 = ECAlayer(128)
        self.down3 = Downsample_block(128, 256)
        # self.ECAlayer3 = ECAlayer(256)
        self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_chan, 1)
        self.outconvp1 = nn.Conv2d(64, out_chan, 1)
        self.outconvm1 = nn.Conv2d(64, out_chan, 1)

    def forward(self, x):
        x, y1 = self.down1(x)
        x = self.ECAlayer1(x)
        x, y2 = self.down2(x)
        x = self.ECAlayer2(x)
        x, y3 = self.down3(x)
        # x = self.ECAlayer3(x)
        x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)

        return x1

class ECA2net(nn.Module):
    def __init__(self):
        in_chan = 4
        out_chan = 3
        super(ECA2net, self).__init__()
        self.down1 = Downsample_block(in_chan, 64)
        self.ECAlayer1 = ECAlayer(64)
        self.down2 = Downsample_block(64, 128)
        self.ECAlayer2 = ECAlayer(128)
        self.down3 = Downsample_block(128, 256)
        self.ECAlayer3 = ECAlayer(256)
        self.down4 = Downsample_block(256, 512)
        self.ECAlayer4 = ECAlayer(512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_chan, 1)
        self.outconvp1 = nn.Conv2d(64, out_chan, 1)
        self.outconvm1 = nn.Conv2d(64, out_chan, 1)

    def forward(self, x):
        x, y1 = self.down1(x)
        x = self.ECAlayer1(x)
        x, y2 = self.down2(x)
        x = self.ECAlayer2(x)
        x, y3 = self.down3(x)
        x = self.ECAlayer3(x)
        x, y4 = self.down4(x)
        x = self.ECAlayer4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)

        return x1


if __name__ == "__main__":
    ECAmodel = ECA2net()
    input = torch.randn((4, 4, 160, 160))
    ECAout = ECAmodel(input)
    print(ECAout.shape)