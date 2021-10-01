import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.fft as F
from typing import List
import loupe_env.line_sampler
from loupe_env.line_sampler import *

sys.path.insert(0,'/home/huangz78/mri/unet/')
import unet.unet_model
from unet.unet_model import UNet

from utils import kplot

class LOUPE(nn.Module):
    """
        Reimplementation of Loupe (https://arxiv.org/abs/1907.11374) sampling-reconstruction framework
        with straight through estimator (https://arxiv.org/abs/1308.3432).
        The model gets two components: A learned probability mask (Sampler) and a UNet reconstructor.
        
        Args:
            in_chans  (int): Number of channels in the input to the reconstructor (2 for complex image, 1 for real image).
            out_chans (int): Number of channels in the output to the reconstructor (default is 1 for real image).
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
            shape ([int. int]): Shape of the reconstructed image
            slope (float): Slope for the Loupe probability mask. Larger slopes make the mask converge faster to
                           deterministic state.
            sparsity (float): Predefined sparsity of the learned probability mask. 1 / acceleration_ratio
            line_constrained (bool): Sample kspace measurements column by column
            preselect (bool): preselect DC components
    """
    def __init__(
        self,
        n_channels: int = 1,
        unet_skip: bool = True,
        shape: List[int] = [320, 320],
        slope: float = 5,
        sparsity: float = 0.25,
        conjugate_mask: bool = False,
        preselect: bool = True,
        preselect_num: int = 24,
        sampler: LOUPESampler = None,
        unet: UNet = None
    ):
        super().__init__()

        self.preselect = preselect
        self.preselect_num = preselect_num
        self.sparsity = sparsity
        
        # for backward compatability
        self.samplers = nn.ModuleList()
        if sampler is None:
            self.samplers.append(LOUPESampler(shape, slope, sparsity, preselect, preselect_num))
        else:
            self.samplers.append(sampler)
        
        if unet is None:
            self.unet = UNet(n_channels=n_channels,n_classes=1,bilinear=(not unet_skip),skip=unet_skip)
        else:
            self.unet = unet
#         if in_chans == 1:
#             assert self.conjugate_mask, "Reconstructor (denoiser) only take the real part of the ifft output"

    def forward(self, ystar):
        """
        Args:
            kspace (torch.Tensor): Input tensor of shape NHWC (kspace data)
        Returns:
            (torch.Tensor): Output tensor of shape NCHW (reconstructed image )
        """

        # choose kspace sampling location
        # masked_kspace: NHWC
        y, mask = self.samplers[0](ystar,self.sparsity)
        x = torch.abs(F.ifftn(y,dim=(2,3),norm='ortho'))
        x_recon = self.unet(x)      
        return x_recon, mask