import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import relu
from sklearn.metrics import mean_squared_error as mse
import torch.fft as F
from scipy.linalg import circulant
import torch.nn as nn


# Jan 24, removed W1, W2, W1T, W2T
# Jan 25, too slow to do matrix multiplication


# def A_fdpg(X,M):
#     return torch.real(F.ifftn(torch.tensordot(M.to(DType) , F.fftn(X,dim=(0),norm='ortho'), \
#                 dims=([1],[0])),dim=(0),norm='ortho'))

def kplot(y,roll=False,log=False,cmap=None,flip=True):
    '''
    This function plots the reconstructed image.
    -- Options:
    roll: whether to move low frequencies in the middle of the Image
    log : whether to show the image in log scale
    cmap: color map choice for the plot
    '''
    if isinstance(y,torch.Tensor):
        y = y.numpy()
    yshape = y.shape
    if len(yshape)==3: # one complex-valued image
        if yshape[-1]==1 and np.iscomplex(y).any(): # complex number input
            ynew = np.zeros((yshape[0],yshape[1],2))
            ynew[:,:,0] = np.real(y[:,:,0])
            ynew[:,:,1] = np.imag(y[:,:,0])
            y           = ynew
            yshape      = y.shape
        if log:
            im1 = np.log( np.absolute(y[:,:,0]) )
            if yshape[-1] == 2:
                im2 = np.log( np.absolute(y[:,:,1]) )
        else:
            im1 = y[:,:,0]
            if yshape[-1] == 2:
                im2 = y[:,:,1]
        if roll:
            im1 = np.roll(im1,tuple(n//2 for n in im1.shape[:2]), axis=(0,1))  # move the center frequency to the center of the image
            if yshape[-1] == 2:
                im2 = np.roll(im2,tuple(n//2 for n in im2.shape[:2]), axis=(0,1))
        if yshape[-1] == 2:
            fig,axs = plt.subplots(2,1,figsize=(15,10))
            hd1 = axs[0].imshow(im1,cmap=cmap)
            axs[0].set_title('Real part mag')
            plt.colorbar(hd1,ax=axs[0])
            hd2 = axs[1].imshow(im2,cmap=cmap)
            axs[1].set_title('Imag part mag')
            fig.colorbar(hd2,ax=axs[1])
            plt.show()
        else:
            fig,axs = plt.subplots(1,1,figsize=(5,5))
            hd1 = axs.imshow(im1,cmap=cmap)
            # axs.set_title('Real part mag')
            plt.colorbar(hd1,ax=axs)
            plt.show()
    elif len(yshape)==2: # one real-valued image
        fig,axs = plt.subplots(1,1,figsize=(10, 10))
        if roll:
            y = np.roll(y,tuple(n//2 for n in y.shape[:2]), axis=(0,1))  # move the center frequency to the center of the image
        if not log:
            try:
                hd1 = axs.imshow(y,cmap=cmap)
            except TypeError:
                hd1 = axs.imshow(np.abs(y),cmap=cmap)
        else:
            try:
                hd1 = axs.imshow(np.log(y),cmap=cmap)
            except TypeError:
                hd1 = axs.imshow(np.log(np.abs(y)),cmap=cmap)
        axs.set_title('Image')
        plt.colorbar(hd1,ax=axs)
        plt.show()
    elif len(yshape)==1: # mask
        fig,axs = plt.subplots(1,1,figsize=(5, 5))
        mask    = np.reshape(y,(-1,1))
        mask2D  = np.tile(mask,(1,mask.size))
        if flip:
            hd1 = axs.imshow(1-mask2D,cmap='Greys')
        else:
            hd1 = axs.imshow(mask2D,cmap='Greys')
        axs.set_xticks([])
        axs.set_xticks([], minor=True) # for minor ticks
        axs.set_ylabel('Frequencies')
        plt.colorbar(hd1,ax=axs)
        plt.rcParams.update({'font.size': 25})
        plt.show()

def shiftsamp(sparsity,imgHeg):
    '''
    shiftsamp returns the sampled mask from the top and the bottom of an Image
    output: mask, maskInd, erasInd
    mask is a binary vector
    maskInd is the collection of sampled row markers
    erasInd is the collection of erased row markers
    '''
    assert(sparsity<imgHeg)
    if sparsity <= 1:
        quota    = int(imgHeg*sparsity)
    else:
        quota    = int(sparsity)
    maskInd  = np.concatenate((np.arange(0,quota//2 ),np.arange(imgHeg-1,imgHeg-1-quota//2-quota%2,-1)))
    erasInd  = np.setdiff1d(np.arange(imgHeg),maskInd)
    mask     = torch.ones(imgHeg)
    mask[erasInd] = 0
    return mask,maskInd,erasInd


def raw_normalize(M,budget,threshold=0.5):
    '''
    M: full mask 
    budget: how many frequencies to sample
    threshold: deafult 0.5
    to be applied after sigmoid but before binarize!
    '''
    d = M.shape[0]
    assert(budget <= d)
    alpha = budget/d
    nnz   = torch.sum(M>threshold)
    pbar  = nnz/d
    with torch.no_grad():
        if  nnz > budget:
            sampinds  = np.argsort(M.detach().numpy())[::-1][0:budget]
            eraseinds = np.setdiff1d(np.arange(0,M.shape[0],1),sampinds)
            M[eraseinds] = 0
        elif nnz < budget:
            M_tmp = 1-(1-alpha)/(1-pbar)*(1-M)
            sampinds  = np.argsort(M_tmp.detach().numpy())[::-1][0:budget]
            eraseinds = np.setdiff1d(np.arange(0,M_tmp.shape[0],1),sampinds)
            M_out = torch.ones_like(M_tmp)
            M_out[eraseinds] = 0
            M = M_out
    return M

def mask_prob(img,fix=10,other=30,roll=True):
    fix = int(fix)
    other = int(other)
    imgHeg = img.shape[0]
    y = np.fft.fftn(img,norm='ortho')
    p = np.sum(np.abs(y),axis=1)
    fixInds  = np.concatenate((np.arange(0,round(fix//2) ),np.arange(imgHeg-1,imgHeg-1-round(fix/2),-1)))
    p[fixInds] = 0
    p = p/np.sum(p) # normalize probability vector
    addInds  = np.random.choice(np.arange(imgHeg),size=other,replace=False,p=p)
    maskInds = np.concatenate((fixInds,addInds))
    mask          = np.zeros(imgHeg)
    mask[maskInds]= 1
    if roll:
        mask = np.roll(mask,shift=imgHeg//2,axis=0)
    return mask

def mask_naiveRand(imgHeg,fix=10,other=30,roll=False):
    '''
    return a naive mask: return all known low-frequency
    while sample high frequency at random based on sparsity budget
    return UNROLLED mask!
    '''
    fix = int(fix)
    other = int(other)
    _, fixInds, _ = shiftsamp(fix,imgHeg)
    IndsLeft      = np.setdiff1d(np.arange(imgHeg),fixInds)
    RandInds      = IndsLeft[np.random.choice(len(IndsLeft),other,replace=False)]
    maskInd       = np.concatenate((fixInds,RandInds))
    erasInd       = np.setdiff1d(np.arange(imgHeg),maskInd)
    mask          = torch.ones(imgHeg)
    mask[erasInd] = 0
    if not roll:
        return mask,maskInd,erasInd
    else:
        mask = torch.fft.fftshift(mask)
        return mask,None,None

def mask_makebinary(M,beta=1,threshold=0.5,sigma=True):
    '''
    return a mask in the form of binary vector
    threshold the continuous mask into a binary mask
    '''
    if sigma:
        Mval = torch.sigmoid(beta*M)
    else:
        Mval = M
    MASK = torch.ones(Mval.shape)
    MASK[Mval<=threshold] = 0
    return MASK

def mask_complete(highmask,imgHeg,rolled=True,dtyp=torch.double):
    '''
    mold the highmask into a complete full length mask
    fill observed low frequency with 1
    '''
    base = imgHeg - highmask.size()[0]
    fullmask = torch.zeros((imgHeg),dtype=dtyp)
    if rolled:
        coreInds = np.arange(int(imgHeg/2)-int(base/2), int(imgHeg/2)+int(base/2))
    else:
        coreInds = np.concatenate((np.arange(0,base//2),np.arange(Heg-1,Heg-1-base//2-base%2,-1)))
    fullmask[coreInds] = 1
    fullmask[np.setdiff1d(np.arange(imgHeg),coreInds)] = highmask
    return fullmask

def mask_filter(M,base=10,roll=False):
    '''
    M:  input mask, a vector or a matrix. 
        If it is a matrix, the second dimension is assumed to be the mask dimension.
    roll: whether to centralize the low frequencies
    base: core frequency basis
    '''
    if len(M.shape)==1:
        Heg = M.shape[0]
        if roll:
            coreInds = np.arange(int(Heg//2)-int(base//2), int(Heg//2)+int(base//2)+base%2)
        else:
            coreInds = np.concatenate((np.arange(0,base//2),np.arange(Heg-1,Heg-1-base//2-base%2,-1)))
        M_high = M[np.setdiff1d(np.arange(Heg),coreInds)]  
        
    elif len(M.shape)==2:
        Heg = M.shape[1]
        if roll:
            coreInds = np.arange(int(Heg//2)-int(base//2), int(Heg//2)+int(base//2)+base%2)
        else:
            coreInds = np.concatenate((np.arange(0,base//2),np.arange(Heg-1,Heg-1-base//2-base%2,-1)))
        M_high = M[:,np.setdiff1d(np.arange(Heg),coreInds)]
        
    return M_high

def get_x_f_from_yfull(mask,yfull,DTyp=torch.cfloat):
    z_f = torch.fft.ifftshift(torch.tensordot(torch.diag(mask).to(DTyp),yfull,dims=([1],[0])))
    x_f = torch.abs(F.ifftn(z_f,dim=(0,1),norm='ortho'))
    return x_f

# warmup protocol
def mask_pnorm(y,fix=10,other=30,p=2):
    '''
    This function returns a mask for the warmup purpose created based on energy per row of the k-space image
    '''
    imgHeg   = y.shape[0]
    _, fixInds, _ = shiftsamp(fix,imgHeg)
    energy   = torch.squeeze(torch.sum(torch.abs(y)**p,dim=1))
    eRank    = torch.argsort(energy,descending=True).numpy()
    IndsLeft = np.setdiff1d(eRank,fixInds)
    yLeft    = y[IndsLeft,:,:]
    energy   = torch.squeeze(torch.sum(torch.abs(yLeft)**p,dim=1))
    eRank    = torch.argsort(energy,descending=True).numpy()
    IndsSamp = eRank[0:other]
    maskInd  = np.concatenate((fixInds,IndsSamp))
    erasInd  = np.setdiff1d(np.arange(imgHeg),maskInd)
    mask     = torch.ones(imgHeg)
    mask[erasInd] = 0
    return mask,maskInd,erasInd
