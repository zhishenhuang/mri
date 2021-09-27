""" Full assembly of the parts to form the complete network """

import torch.nn.functional as Func
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_chans, n_classes, chans=64, bilinear=True,skip=False):
        super(UNet, self).__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skip = skip

        self.inc = DoubleConv(in_chans, chans)
        self.down1 = Down(chans, chans*2)
        self.down2 = Down(chans*2, chans*4)
        self.down3 = Down(chans*4, chans*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(chans*8, chans*16 // factor)
        self.up1 = Up(chans*16, chans*8 // factor, bilinear)
        self.up2 = Up(chans*8, chans*4 // factor, bilinear)
        self.up3 = Up(chans*4, chans*2 // factor, bilinear)
        self.up4 = Up(chans*2, chans, bilinear)
        self.outc = OutConv(chans, n_classes)

    def forward(self, x):
        x0 = x.clone()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if not self.skip:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        else:
            x = self.up1(x5, x4) + x4
            x = self.up2(x, x3)  + x3
            x = self.up3(x, x2)  + x2
            x = self.up4(x, x1)  + x1
            logits = self.outc(x) + x0        
        return logits
