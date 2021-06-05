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

def sigmoid_binarize(M,threshold=0.5):
    sigmoid = nn.Sigmoid()
    mask = sigmoid(M)
    mask_pred = torch.ones_like(mask)
    for ind in range(M.shape[0]):
        mask_pred[ind,mask[ind,:]<=threshold] = 0
    return mask_pred

def rolling_mean(x,window):
    window = int(window)
#   y = np.zeros(x.size-window)
#   for ind in range(y.size):
#       y[ind] = np.mean(x[ind:ind+window])

    # Stephen: for large data, the above gets a bit slow, so we can do this:
#   y = np.convolve(x, np.ones(window)/window, mode='valid')
#   return y
    # or https://stackoverflow.com/a/27681394
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

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
        
def visualization(randqual,mnetqual,greedyqual=None,randspar=None,mnetspar=None,greedyspar=None,\
                  log1=False,log2=False):
    if (randspar is not None) and (mnetspar is not None):
        fig,axs = plt.subplots(2,1,figsize=(12, 6))
    else:
        fig,axs = plt.subplots(1,1,figsize=(12, 6))
    axs[0].set_xlabel('iters', fontsize=16)
    axs[0].set_ylabel('mask loss (rel. l2 recon. err.)', color='r', fontsize=16)
    axs[0].scatter(np.arange(0,len(randqual),1),randqual, color='r',label='rand.',marker='d',s=60)
    axs[0].scatter(np.arange(0,len(mnetqual),1),mnetqual, color='g',label='mnet.',marker='X',s=60)
    if greedyqual is not None:
        axs[0].scatter(np.arange(0,len(greedyqual),1),greedyqual,color='b',label='greedy',marker='*',s=60)
    axs[0].tick_params(axis='x', labelsize='large')
    axs[0].tick_params(axis='y', labelsize='large')
    axs[0].legend(loc='best')
    if log1:
        axs[0].set_yscale('log')
    

    if (randspar is not None) and (mnetspar is not None):
        axs[1].set_xlabel('iters', fontsize=16)
        axs[1].set_ylabel('sampling ratio', fontsize=16)
        axs[1].scatter(np.arange(0,len(randspar),1),randspar,color='r',label='rand.',marker='d',s=50)
        axs[1].scatter(np.arange(0,len(mnetspar),1),mnetspar,color='g',label='mnet.',marker='X',s=50)
        if greedyspar is not None:
            axs[1].scatter(np.arange(0,len(greedyspar),1),greedyspar,color='b',label='greedy',marker='*',s=60)
        if log2:
            axs[1].set_yscale('log')
        axs[1].tick_params(axis='x', labelsize='large')
        axs[1].tick_params(axis='y', labelsize='large')
        axs[1].legend(loc='best')
    
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

def mask_makebinary(M,beta=1,threshold=0.5,sigma=True): # done
    '''
    return a mask in the form of binary vector
    threshold the continuous mask into a binary mask
    '''
    if sigma:
        Mval = torch.sigmoid(beta*M)
    else:
        Mval = M
    MASK = torch.ones(Mval.shape)
    for ind in range(M.shape[0]):
        MASK[ind,Mval[ind,:]<=threshold] = 0
    return MASK

def mask_complete(highmask,imgHeg,rolled=True,dtyp=torch.float): # done
    '''
    mold the highmask into a complete full length mask
    fill observed low frequency with 1
    '''
    layer = highmask.shape[0]
    base = imgHeg - highmask.size()[1]
    fullmask = torch.zeros((layer,imgHeg),dtype=dtyp)
    if rolled:
        coreInds = np.arange(int(imgHeg/2)-int(base/2), int(imgHeg/2)+int(base/2))
    else:
        coreInds = np.concatenate((np.arange(0,base//2),np.arange(Heg-1,Heg-1-base//2-base%2,-1)))
    fullmask[:,coreInds] = 1
    fullmask[:,np.setdiff1d(np.arange(imgHeg),coreInds)] = highmask
    return fullmask

def mask_filter(M,base=10,roll=False): # done
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

def raw_normalize(M,budget,threshold=0.5): # done
    '''
    M: full mask 
    budget: how many frequencies to sample
    threshold: deafult 0.5
    to be applied after sigmoid but before binarize!
    '''
    d = M.shape[1]
    assert(budget <= d)
    alpha = budget/d
    with torch.no_grad():
        for ind in range(M.shape[0]):
            nnz   = torch.sum(M[ind,:]>threshold)
            pbar  = nnz/d
            if  nnz > budget:
                sampinds  = np.argsort(M[ind,:].detach().numpy())[::-1][0:budget]
                eraseinds = np.setdiff1d(np.arange(0,M.shape[1],1),sampinds)
                M[ind,eraseinds] = 0
            elif nnz < budget:
                M_tmp = 1-(1-alpha)/(1-pbar)*(1-M[ind,:])
                sampinds  = np.argsort(M_tmp.detach().numpy())[::-1][0:budget]
                eraseinds = np.setdiff1d(np.arange(0,M_tmp.shape[0],1),sampinds)
                M_out = torch.ones_like(M_tmp)
                M_out[eraseinds] = 0
                M[ind,:] = M_out
    return M

def get_x_f_from_yfull(mask,yfull,DTyp=torch.cfloat): # done
    if len(mask.shape) == 1:
        mask = mask.repeat(yfull.shape[0],1)
    subsamp_z = torch.zeros(yfull.shape).to(DTyp)
    for ind in range(mask.shape[0]):
        subsamp_z[ind,mask[ind,:]==1,:] = yfull[ind,mask[ind,:]==1,:]
        # torch.tensordot( torch.diag(mask[ind,:]).to(DTyp),yfull[ind,:,:],dims=([1],[0]) )
    z_f = torch.fft.ifftshift(subsamp_z , dim=(1,2))
    x_f = torch.abs(F.ifftn(z_f,dim=(1,2),norm='ortho'))
    return x_f

def apply_mask(mask,yfull,DTyp=torch.cfloat,mode='r'): # done
    '''
    yfull should have dimension (batchsize, Heg, Wid), and is assumed to be a complex image
    '''
    if len(mask.shape) == 1:
        mask = mask.repeat(yfull.shape[0],1)
    subsamp_z = torch.zeros(yfull.shape).to(DTyp)
    for ind in range(mask.shape[0]):
        subsamp_z[ind,mask[ind,:]==1,:] = yfull[ind,mask[ind,:]==1,:]
        # torch.tensordot( torch.diag(mask[ind,:]).to(DTyp),yfull[ind,:,:],dims=([1],[0]) )
    if mode == 'r':
        z = torch.zeros((subsamp_z.shape[0],2,subsamp_z.shape[1],subsamp_z.shape[2]))
        z[:,0,:,:] = torch.real(subsamp_z)
        z[:,1,:,:] = torch.imag(subsamp_z)
        return z
    elif mode == 'c':
        return subsamp_z

def mnet_wrapper(mnet,z,budget,imgshape,dtyp=torch.float,normalize=False): # done
    if len(z.shape)==3:
        highmask_raw  = torch.sigmoid( mnet( z.view(z.shape[0],1,imgshape[0],imgshape[1]) ) )   
    elif len(z.shape)==4:
        highmask_raw  = torch.sigmoid(mnet(z))
        
    if normalize:
        highmask = mask_makebinary( raw_normalize(highmask_raw,budget) , sigma=False )
    else:
        highmask = mask_makebinary( highmask_raw , sigma=False )
    mnetmask = mask_complete( highmask,imgshape[0],rolled=True,dtyp=dtyp )
    return mnetmask

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
