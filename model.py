import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEdgeDetector(nn.Module):
    def __init__(self):
        super(UNetEdgeDetector, self).__init__()
        self.down1 = self.contract_block(3, 32)
        self.down2 = self.contract_block(32, 64)
        self.bottom = self.contract_block(64, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = self.contract_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = self.contract_block(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def contract_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(F.max_pool2d(c1, 2))
        c3 = self.bottom(F.max_pool2d(c2, 2))
        u1 = self.up1(c3)
        u1 = torch.cat([u1, c2], dim=1)
        u1 = self.conv_up1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, c1], dim=1)
        u2 = self.conv_up2(u2)
        return self.out(u2)
