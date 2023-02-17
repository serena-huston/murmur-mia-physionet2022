import torch
import torch.nn as nn
import torch.nn.functional as F 

class DoubleConv(nn.Module):
    # conv_layer1 -> conv_layer2

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, padding="same", kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, padding="same", kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ContractPath(nn.Module):
    # max pool -> conv_layer1 -> conv_layer2

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.contract = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.contract(x)


class ExpandPath(nn.Module):
    # Upsample -> concatenate -> conv_layer1 -> conv_layer2

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[0] - x1.size()[0]
        diffX = x2.size()[1] - x1.size()[1]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
       
        x = torch.cat([x2, x1])
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # self.fc = nn.Linear(256, 4)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        return x


