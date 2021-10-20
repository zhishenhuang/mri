import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import os

def lpnorm(x,xstar,p='fro',mode='sum'):
    '''
    x and xstar are both assumed to be in the format NCHW
    '''
    assert(x.shape==xstar.shape)
    numerator   = torch.norm(x-xstar,p=p,dim=(2,3))
    denominator = torch.norm(xstar  ,p=p,dim=(2,3))
    if mode == 'sum':
        error = torch.sum( torch.div(numerator,denominator) )
    elif mode == 'mean':
        error = torch.mean( torch.div(numerator,denominator) )
    return error

def fft_new(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))

    image = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
        )
    )
    return image


def ifft_new(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))
    image = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
        )
    )
    return image

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = fft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = ifft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def complex_matmul(a, b):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the third last two channels ((batch), (coil), 2, nx, ny).
    if len(a.size()) == 3:
        return torch.cat(((a[0, ...] * b[0, ...] - a[1, ...] * b[1, ...]).unsqueeze(0),
                          (a[0, ...] * b[1, ...] + a[1, ...] * b[0, ...]).unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat(((a[:, 0, ...] * b[:, 0, ...] - a[:, 1, ...] * b[:, 1, ...]).unsqueeze(1),
                          (a[:, 0, ...] * b[:, 1, ...] + a[:, 1, ...] * b[:, 0, ...]).unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat(((a[:, :, 0, ...] * b[:, :, 0, ...] - a[:, :, 1, ...] * b[:, :, 1, ...]).unsqueeze(2),
                          (a[:, :, 0, ...] * b[:, :, 1, ...] + a[:, :, 1, ...] * b[:, :, 0, ...]).unsqueeze(2)), dim=2)


def complex_conj(a):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the last two channels.
    if len(a.size()) == 3:
        return torch.cat((a[0, ...].unsqueeze(0), -a[1, ...].unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat((a[:, 0, ...].unsqueeze(1), -a[:, 1, ...].unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat((a[:, :, 0, ...].unsqueeze(2), -a[:, :, 1, ...].unsqueeze(2)), dim=2)

class OPATA(nn.Module):
    # Gram operator for multi-coil Cartesian MRI
    # Initialize: Sensitivity maps: [Batch, Coils, 2, M, N]
    # Input: Image: [Batch, 2, M, N]
    #        Mask: [Batch, 2, M, N]
    # Return: Image: [Batch, 2, M, N]
    def __init__(self, Smap, lambda1):
        super(OPATA, self).__init__()
        self.Smap = Smap
        self.lambda1 = lambda1

    def forward(self, im, mask):
        BchSize, num_coil, _, M, N = self.Smap.size()
        im_coil = im.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        Image_s = complex_matmul(im_coil, self.Smap)
        k_full = fft2(Image_s.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        mask = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        k_under = k_full * mask
        Im_U = ifft2((k_under).permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        Im_Us = complex_matmul(Im_U, complex_conj(self.Smap)).sum(1)
        return Im_Us + self.lambda1 * im
    
def cg_block(smap, mask, b0, z_pad, lambda1, tol, M=None, dn=None):
    # A specified conjugated gradietn block for MR system matrix A.
    # Not very efficient.
    # Sensitivity map: [Batch, Coils, 2, M, N]
    # Dn: denoised image from CNN, [Batch, 2, M, N]
    # Z_pad: Ifake = alised image, [Batch, 2, M, N]
    ATA = OPATA(smap, lambda1)
    x0 = torch.zeros_like(b0)
    if dn is not None:
        x0 = dn
    num_loop = 0
    r0 = b0 - ATA(x0, mask)
    p0 = r0
    rk = r0
    pk = p0
    xk = x0
    while torch.norm(rk).data.cpu().numpy().tolist() > tol:
        rktrk = torch.pow(torch.norm(rk), 2)
        pktapk = torch.sum(complex_matmul(complex_conj(pk), ATA(pk, mask)))
        alpha = rktrk / pktapk
        xk1 = xk + alpha * pk
        rk1 = rk - alpha * ATA(pk, mask)
        rk1trk1 = torch.pow(torch.norm(rk1), 2)
        beta = rk1trk1 / rktrk
        pk1 = rk1 + beta * pk
        xk = xk1
        rk = rk1
        pk = pk1
        num_loop = num_loop + 1

    return xk

class CG(torch.autograd.Function):
    # Modified solver for (A'A+\lambda I)^{-1} (\lambda A'y + Dn)
    # For reference, see: https://arxiv.org/abs/1712.02862
    # Input: Dn, Z_pad: [Batch, 2, M, N]
    #        Mask: [Batch, 2, M, N]
    #        smap: [Batch, ncoil, 2, M, N]
    #        tol: exiting threshold
    # Return: Image: [Batch, 2, M, N]
    @staticmethod
    def forward(ctx, dn, tol, lambda1, smap, mask, z_pad):
        tol = torch.tensor(tol).to(device=dn.device, dtype=dn.dtype)
        lambda1 = torch.tensor(lambda1).to(device=dn.device, dtype=dn.dtype)
        ctx.save_for_backward(tol, lambda1, smap, mask, z_pad)
        return cg_block(smap, mask, dn * lambda1 + z_pad, z_pad, lambda1, tol, dn=dn)

    @staticmethod
    def backward(ctx, dx):
        tol, lambda1, smap, mask, z_pad = ctx.saved_tensors
        return lambda1 * cg_block(smap, mask, dx, z_pad, lambda1, tol), None, None, None, None, None


