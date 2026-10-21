import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, decoder_channels, encoder_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(decoder_channels + encoder_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 512, 256)
        self.up2 = Up(256, 256, 128)
        self.up3 = Up(128, 128, 64)
        self.up4 = Up(64, 64, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x): 
        

        x = x.unsqueeze(0)  # (1, H, W)  
        x = x.unsqueeze(0)  # (1, 1, H, W)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x.squeeze(0).squeeze(0), x5  # 返回 (B, H, W) 形状



import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

def masked_mse_loss(output, target, mask):
    return ((output - target)**2 * mask.float()).sum() / mask.sum()




model = UNet(n_channels=1, n_classes=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)




