# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

from cnn_segmentation.unet_segmentation_arch import * 
from cnn_segmentation.cnn_data_preprocessing import * 
import torch
import torch.nn as nn
import torch.nn.functional as F 


class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.start = DoubleConv(n_channels, n_channels*2)
        self.down1 = ContractPath(n_channels*2, n_channels*4)
        self.down2 = ContractPath(n_channels*4, n_channels*8)
        self.down3 = ContractPath(n_channels*8, n_channels*16)
        self.down4 = ContractPath(n_channels*16, n_channels*32)

        factor = 2 if bilinear else 1

        self.up1 = ExpandPath(n_channels*32, n_channels*16// factor, bilinear)
        self.up2 = ExpandPath(n_channels*16, n_channels*8 // factor, bilinear)
        self.up3 = ExpandPath(n_channels*8, n_channels*4 // factor, bilinear)
        self.up4 = ExpandPath(n_channels*4, n_channels*2, bilinear)

        self.outc = OutConv(n_channels*2, n_classes)

    def forward(self, x):
        x1 = self.start(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight.data)
    