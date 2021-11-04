import numpy as np
import argparse
import os
import sys
import torch
import torch.fft as F
from importlib import reload
from torch.nn.functional import relu
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import utils
import logging

def outsize(hin,win,ker=(3,3),stride=(1,1),padding=(0,0),dilation=(1,1)):
    if isinstance(ker,int):
        ker=(ker,ker)
    if isinstance(stride,int):
        stride=(stride,stride)
    if isinstance(padding,int):
        padding=(padding,padding)
    if isinstance(dilation,int):
        dilation=(dilation,dilation)
    hout = int( (hin + 2*padding[0]- dilation[0]*(ker[0]-1)-1)/stride[0] + 1  )
    wout = int( (win + 2*padding[1]- dilation[1]*(ker[1]-1)-1)/stride[1] + 1  )
    return hout,wout

def DC_outsize(hin,win,ker=3,stride=1,padding=1):
    hout1,wout1 = outsize(hin,win,ker=ker,padding=padding)
    hout2,wout2 = outsize(hout1,wout1,ker=ker,padding=padding)
    return hout2,wout2

def Down_outsize(hin,win,ker=3,poolk=3,stride=1,poolpad=0,dcpad=1):
    hout1,wout1 = outsize(hin,win,ker=poolk,stride=poolk,padding=poolpad)
    hout2,wout2 = DC_outsize(hout1,wout1,ker=ker,stride=stride,padding=dcpad)
    return hout2,wout2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,convk=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if convk%2!=1:
            convk += 1
        padding = (convk-1)//2
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=convk, padding=padding),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=convk, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, poolk=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(poolk),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(nn.Module):
    def __init__(self):
        super(OutConv, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        return self.avgpool(x)


class MNet(nn.Module):
    def __init__(self, in_chans=1,imgsize=320,out_size=320,beta=1,poolk=3):
        super(MNet, self).__init__()
        self.in_channels = in_chans
        
        if isinstance(imgsize,int):
            self.imgHeg = imgsize
            self.imgWid = imgsize
        elif isinstance(imgsize,tuple):
            self.imgHeg = imgsize[0]
            self.imgWid = imgsize[1]
        ## parameter part
        self.beta = beta
        self.out_size = out_size
        
        ## network part
        self.inc   = DoubleConv(in_channels, 128)
        self.down1 = Down(128, 256,poolk=poolk)
        self.down2 = Down(256, 512,poolk=poolk)
        self.down3 = Down(512, 1024,poolk=poolk)
        self.down4 = Down(1024, 2048,poolk=poolk)
        self.outc  = OutConv()
        self.midheg,self.midwid = \
            outsize(*Down_outsize(*Down_outsize(*Down_outsize(*Down_outsize(\
                                                                            *DC_outsize(self.imgHeg,self.imgWid),poolk=poolk),poolk=poolk),poolk=poolk),poolk=poolk),ker=2,stride=2)
        self.veclen = 2048 * self.midheg * self.midwid
        lwid1 = self.veclen - ((self.veclen-self.imgHeg)*4)//11
        lwid2 = self.veclen - ((self.veclen-self.imgHeg)*8)//11
        lwid3 = self.veclen - ((self.veclen-self.imgHeg)*10)//11
        lwid4 = self.out_size
        self.fc1 = nn.Linear(self.veclen, lwid1)
        self.fc2 = nn.Linear(lwid1, lwid2)
        self.fc3 = nn.Linear(lwid2, lwid3)
        self.fc4 = nn.Linear(lwid3, lwid4)

    def forward(self, x):
        LeakyReLU = nn.LeakyReLU()
        batchsize = x.shape[0]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        M  = self.outc(x5).view(batchsize,-1)
        M  = LeakyReLU(self.fc1(M))
        M  = LeakyReLU(self.fc2(M))
        M  = LeakyReLU(self.fc3(M))
        M  = self.beta * self.fc4(M) # Mar 1, to cope with BCELossWithLogit
        # M  = torch.sigmoid( self.beta * self.fc3(M) ) # sigmoid
        # M  = Func.relu(M-self.soft) # for testing purposes only   # + self.soft # soft-thresholding
        return M
