import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import relu
import torch.fft as F
from scipy import ndimage
import skimage
from skimage.metrics import structural_similarity as ss
from skimage.metrics import peak_signal_noise_ratio as psnr_
import copy

# Jan 24, removed W1, W2, W1T, W2T
# Jan 25, too slow to do matrix multiplication


# def A_fdpg(X,M):
#     return torch.real(F.ifftn(torch.tensordot(M.to(DType) , F.fftn(X,dim=(0),norm='ortho'), \
#                 dims=([1],[0])),dim=(0),norm='ortho'))


def nn_weights_init(m):
    classname = m.__class__.__name__
#     print(m)
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# def mnet_weights_init(m):
#     classname = m.__class__.__name__
# #     print(classname)
#     if classname == 'DoubleConv':
#         for ind in range(4):
#             subclassname = m.double_conv[ind].__class__.__name__
#             if subclassname.find('Conv') != -1:
#                 nn.init.normal_(m.double_conv[ind].weight.data, 0.0, 0.02)
#                 print(classname,'1')
#             elif subclassname.find('BatchNorm') != -1:
#                 nn.init.normal_(m.double_conv[ind].weight.data, 1.0, 0.02)
#                 print(classname,'2-1')
#                 nn.init.constant_(m.double_conv[ind].bias.data, 0)
#                 print(classname,'2-2')
#             elif subclassname.find('Linear') != -1:
#                 nn.init.normal_(m.double_conv[ind].weight.data, 0.0, 0.02)
#                 print(classname,'3-1')
#                 nn.init.constant_(m.double_conv[ind].bias.data, 0)
#                 print(classname,'3-2')
#     elif classname == 'OutConv':
#         pass   
#     else:
#         if classname.find('Conv') != -1:
#             nn.init.normal_(m.weight.data, 0.0, 0.02)
#             print(classname,'4')
#         elif classname.find('BatchNorm') != -1:
#             nn.init.normal_(m.weight.data, 1.0, 0.02)
#             print(classname,'5-1')
#             nn.init.constant_(m.bias.data, 0)
#             print(classname,'5-2')
#         elif classname.find('Linear') != -1:
#             nn.init.normal_(m.weight.data, 0.0, 0.02)
#             print(classname,'6-1')
#             nn.init.constant_(m.bias.data, 0)
#             print(classname,'6-2')
def setdiff1d(a,b):
    '''
    set difference between 1d Tensors a and b
    elements in a and b are expected to be Unique
    '''
    comb = torch.cat((a,b))
    uniques, counts = comb.unique(return_counts=True)
    diffElems = uniques[counts==1]
    return diffElems

def sigmoid_binarize(M,threshold=0.5,device='cpu'):
    sigmoid = nn.Sigmoid()
    mask = sigmoid(M)
    mask_pred = torch.ones_like(mask,device=device)
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

def kplot(y,roll=False,log=False,cmap=None,flip=True,img_name=None):
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
        if img_name is not None:
            axs.set_title(img_name)
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
        if img_name is not None:
            axs.set_title(img_name)
        plt.colorbar(hd1,ax=axs)
        plt.show()
    elif len(yshape)==1: # mask
        fig,axs = plt.subplots(1,1,figsize=(5, 5))
        if roll:
            y = np.roll(y,yshape[0]//2,axis=0)
        mask    = np.reshape(y,(-1,1))
        mask2D  = np.tile(mask,(1,mask.size))
        if flip:
            hd1 = axs.imshow(1-mask2D,cmap='Greys')
        else:
            hd1 = axs.imshow(mask2D,cmap='Greys')
        axs.set_xticks([])
        axs.set_xticks([], minor=True) # for minor ticks
        axs.set_ylabel('Frequencies')
        if not flip:
            plt.colorbar(hd1,ax=axs)
        if img_name is not None:
            axs.set_title(img_name)
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
    maskInd  = np.concatenate((np.arange(0,quota//2),np.arange(imgHeg-1,imgHeg-1-quota//2-quota%2,-1)))
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
        mask = F.fftshift(mask)
        return mask,None,None
    
def mask_equidist(imgHeg,fix=10,other=30,roll=False):
    '''
    return an equi-distance mask
    '''
    fix = int(fix)
    other = int(other)
    _, fixInds, _ = shiftsamp(fix,imgHeg)
    IndsLeft = np.setdiff1d(np.arange(imgHeg),fixInds)
    sampinds = np.linspace(0,imgHeg-fix-1,other).astype(int)
    equiInds = IndsLeft[sampinds]
    maskInds = np.concatenate((fixInds,equiInds))
    erasInds = np.setdiff1d(np.arange(imgHeg),maskInds)
    mask     = torch.ones(imgHeg)
    mask[erasInds] = 0
    if not roll:
        return mask
    else:
        mask = F.fftshift(mask)
        return mask
    

def mask_makebinary(M,beta=1,threshold=0.5,sigma=True,device='cpu'): # done for gpu
    '''
    M: [#imgs, mask_dimension], e.g. [5 imgs, 320 lines in fullmask]
    return a mask in the form of binary vector
    threshold the continuous mask into a binary mask
    '''
    if sigma:
        Mval = torch.sigmoid(beta*M)
    else:
        Mval = M
    MASK = torch.ones(Mval.shape,device=device)
    for ind in range(M.shape[0]):
        MASK[ind,Mval[ind,:]<=threshold] = 0
    return MASK

def mask_complete(highmask,imgHeg,rolled=True,dtyp=torch.float,device='cpu'): # done for gpu
    '''
    highmask: [#imgs, mask_dimension], e.g. [5 imgs, 296 lines in highmask]
    mold the highmask into a complete full length mask
    fill observed low frequency with 1
    '''
    layer = highmask.shape[0]
    base  = imgHeg - highmask.size()[1]
    fullmask = torch.zeros((layer,imgHeg),dtype=dtyp,device=device)
    if rolled:
        coreInds = torch.arange(imgHeg//2-base//2, imgHeg//2+base//2+base%2,1,device=device)
    else:
        coreInds = torch.cat((torch.arange(0,base//2,1,device=device),torch.arange(Heg-1,Heg-1-base//2-base%2,-1,device=device)))
    fullmask[:,coreInds] = 1
    fullmask[:,setdiff1d(torch.arange(0,imgHeg,device=device),coreInds)] = highmask
    return fullmask

def mask_filter(M,base=10,roll=False,device='cpu'): # done for gpu
    '''
    M:  input mask, a vector or a matrix. 
        If it is a matrix, the second dimension is assumed to be the mask dimension.
    roll: whether to centralize the low frequencies
    base: core frequency basis
    '''
    if len(M.shape)==1:
        Heg = M.shape[0]
        if roll:
            coreInds = torch.arange(int(Heg//2)-int(base//2), int(Heg//2)+int(base//2)+base%2,device=device)
        else:
            coreInds = torch.cat((torch.arange(0,base//2,device=device),torch.arange(Heg-1,Heg-1-base//2-base%2,-1,device=device)))
        M_high = M[setdiff1d(torch.arange(0,Heg,device=device),coreInds)]  
        
    elif len(M.shape)==2:
        Heg = M.shape[1]
        if roll:
            coreInds = torch.arange(int(Heg//2)-int(base//2), int(Heg//2)+int(base//2)+base%2,device=device)
        else:
            coreInds = torch.cat((torch.arange(0,base//2,device=device),torch.arange(Heg-1,Heg-1-base//2-base%2,-1,device=device)))
        M_high = M[:,setdiff1d(torch.arange(0,Heg,device=device),coreInds)]
        
    return M_high


def raw_normalize(M,budget,threshold=0.5,device='cpu'): # done for gpu
    '''
    M: full mask with shape [#images, imgHeg]
    budget: how many frequencies to sample
    threshold: deafult 0.5
    to be applied after sigmoid but before binarize!
    '''
    d = M.shape[1]
    allinds  = torch.arange(0,d,1,device=device)
    assert(budget <= d)
    alpha = budget/d
    with torch.no_grad():
        for ind in range(M.shape[0]):
            nnz  = torch.sum(M[ind,:]>threshold)
            pbar = nnz/d
            if  nnz >= budget:
                sampinds  = torch.argsort(M[ind,:],descending=True)[0:budget]
                eraseinds = setdiff1d(allinds,sampinds)
#                 sampinds  = np.argsort(M[ind,:].clone().detach().numpy())[::-1][0:budget]
#                 eraseinds = np.setdiff1d(np.arange(0,M.shape[1],1),sampinds)
                M[ind,eraseinds] = 0
            elif nnz < budget:
                M_tmp = 1-(1-alpha)/(1-pbar)*(1-M[ind,:])
                sampinds  = torch.argsort(M_tmp,descending=True)[0:budget]              
                eraseinds = setdiff1d(allinds,sampinds)
#                 sampinds  = np.argsort(M_tmp.detach().numpy())[::-1][0:budget]
#                 eraseinds = np.setdiff1d(np.arange(0,M_tmp.shape[0],1),sampinds)
                M_out = torch.ones_like(M_tmp,device=device)
                M_out[eraseinds] = 0
                M[ind,:] = M_out
    return M

def get_x_f_from_yfull(mask,yfull,DTyp=torch.cfloat,device='cpu'): # done for gpu
    '''
    yfull is assumed to be rolled!
    apply mask and then compute ifft to get image
    '''
    if len(mask.shape) == 1:
        mask = mask.repeat(yfull.shape[0],1)
    subsamp_z = torch.zeros(yfull.shape,device=device).to(DTyp)
    for ind in range(mask.shape[0]):
        subsamp_z[ind,mask[ind,:]==1,:] = yfull[ind,mask[ind,:]==1,:]
        # torch.tensordot( torch.diag(mask[ind,:]).to(DTyp),yfull[ind,:,:],dims=([1],[0]) )
    z_f = torch.fft.ifftshift(subsamp_z , dim=(1,2))
    x_f = torch.abs(F.ifftn(z_f,dim=(1,2),norm='ortho'))
    return x_f

def apply_mask(mask,yfull,mode='r',device='cpu'): # done for gpu
    '''
    yfull should have dimension (batchsize, Heg, Wid), and is assumed to be a complex image
    'r' mode: output 2-channel info (one layer with real info and the other with imag info) as input to 2-channel mnet
    'c' mode: output 1-channel complex info
    '''
    if len(mask.shape) == 1:
        mask = mask.repeat(yfull.shape[0],1)
    subsamp_z = torch.zeros(yfull.shape,device=device).to(yfull.dtype)
    for ind in range(mask.shape[0]):
        subsamp_z[ind,mask[ind,:]==1,:] = yfull[ind,mask[ind,:]==1,:]
        # torch.tensordot( torch.diag(mask[ind,:]).to(DTyp),yfull[ind,:,:],dims=([1],[0]) )
    if mode == 'r':
        z = torch.zeros((subsamp_z.shape[0],2,subsamp_z.shape[1],subsamp_z.shape[2]),device=device)
        z[:,0,:,:] = torch.real(subsamp_z)
        z[:,1,:,:] = torch.imag(subsamp_z)
        return z
    elif mode == 'c':
        return subsamp_z.to(device)

def mnet_wrapper(mnet,z,budget,imgshape,dtyp=torch.float,normalize=False,complete=True,detach=False,device='cpu'): # done for gpu
    if len(z.shape)==3:
        highmask_raw  = torch.sigmoid( mnet( z.view(z.shape[0],mnet.in_channels,imgshape[0],imgshape[1]) ) )   
    elif len(z.shape)==4:
        highmask_raw  = torch.sigmoid( mnet(z) )
    
    if detach:
        highmask_raw = highmask_raw.detach()
        
    if normalize:
        highmask = mask_makebinary( raw_normalize(highmask_raw,budget,device=device) , sigma=False ,device=device)
    else:
        highmask = mask_makebinary( highmask_raw , sigma=False ,device=device)
        
    if complete:
        mnetmask = mask_complete( highmask,imgshape[0],rolled=True,dtyp=dtyp ,device=device)
    else:
        mnetmask = highmask
    return mnetmask

# warmup protocol
# def mask_pnorm(y,fix=10,other=30,p=2):
#     '''
#     This function returns a mask for the warmup purpose created based on energy per row of the k-space image
#     '''
#     imgHeg   = y.shape[0]
#     _, fixInds, _ = shiftsamp(fix,imgHeg)
#     energy   = torch.squeeze(torch.sum(torch.abs(y)**p,dim=1))
#     eRank    = torch.argsort(energy,descending=True).numpy()
#     IndsLeft = np.setdiff1d(eRank,fixInds)
#     yLeft    = y[IndsLeft,:,:]
#     energy   = torch.squeeze(torch.sum(torch.abs(yLeft)**p,dim=1))
#     eRank    = torch.argsort(energy,descending=True).numpy()
#     IndsSamp = eRank[0:other]
#     maskInd  = np.concatenate((fixInds,IndsSamp))
#     erasInd  = np.setdiff1d(np.arange(imgHeg),maskInd)
#     mask     = torch.ones(imgHeg)
#     mask[erasInd] = 0
#     return mask,maskInd,erasInd



def compute_l2err(x,xstar):
    '''
    compute relative L2 error for each image (no average)
    assume the input has the format [NHW]
    '''
    assert(x.shape==xstar.shape)
    l2err = torch.zeros(len(x))
    for ind in range(len(x)):
        l2err[ind] = torch.norm(x[ind,:,:] - xstar[ind,:,:])/torch.norm(xstar[ind,:,:])
    return l2err

def compute_l1err(x,xstar):
    '''
    compute relative L1 error for each image (no average)
    assume the input has the format [NHW]
    '''
    assert(x.shape==xstar.shape)
    l1err = torch.zeros(len(x))
    for ind in range(len(x)):
        l1err[ind] = torch.norm(x[ind,:,:] - xstar[ind,:,:],p=1)/torch.norm(xstar[ind,:,:],p=1)
    return l1err

def compute_hfen(recon: torch.Tensor,gt: torch.Tensor) -> np.ndarray:
    '''
    compute hfen for each image (no average)
    assume the input has the format [NHW]
    '''
    assert(recon.shape==gt.shape)
    if type(recon) is torch.Tensor:
        gt    = gt.to(torch.cfloat).cpu()
    if type(gt) is torch.Tensor:
        recon = recon.to(torch.cfloat).cpu()
    hfens = np.zeros(len(recon))
    for ind in range(len(recon)):
        img_gt     = gt[ind]
        img_recon  = recon[ind]
        LoG_GT     = ndimage.gaussian_laplace(np.real(gt),    sigma=1) + 1j*ndimage.gaussian_laplace(np.imag(gt),    sigma=1)
        LoG_recon  = ndimage.gaussian_laplace(np.real(recon), sigma=1) + 1j*ndimage.gaussian_laplace(np.imag(recon), sigma=1)
        hfens[ind] = np.linalg.norm(LoG_recon - LoG_GT)/np.linalg.norm(LoG_GT)
    return hfens

def compute_ssim(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    '''
    assume the input is a 3D tensor
    '''
    ssims = []
    for i in range(xs.shape[0]):
        ssim = ss(
            xs[i].cpu().numpy(),
            ys[i].cpu().numpy(),
            data_range=ys[i].cpu().numpy().max(),
        )
        ssims.append(ssim)
    return np.array(ssims, dtype=np.float32)


# def compute_psnr(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
#     psnrs = []
#     for i in range(xs.shape[0]):
#         psnr = psnr_(
#             xs[i].cpu().numpy(),
#             ys[i].cpu().numpy(),
#             data_range=ys[i].cpu().numpy().max(),
#         )
#         psnrs.append(psnr)
#     return np.array(psnrs, dtype=np.float32)

def compute_psnr(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    '''
    compute psnr for each image (no average)
    assume the input has the format [NHW]
    '''
    assert(len(xs.shape)==len(ys.shape))
    if   len(xs.shape) == 3:
        num_elems = xs.shape[1] * xs.shape[2] 
    elif len(xs.shape) == 4:
        num_elems = xs.shape[1] * xs.shape[2] * xs.shape[3]
    
    psnrs = torch.zeros(len(xs))
    for ind in range(len(xs)):
        MSE = torch.sum(torch.square(xs[ind]-ys[ind])) / num_elems
        psnrs[ind] = 10 * torch.log10(1/MSE)
        
    return psnrs

def lpnorm(x,xstar,p='fro',mode='sum'):
    '''
    x and xstar are both assumed to be in the format NCHW
    '''
    assert(x.shape==xstar.shape)
    numerator   = torch.norm(x-xstar,p=p,dim=(2,3))
    denominator = torch.norm(xstar  ,p=p,dim=(2,3))
    if   mode == 'sum':
        error = torch.sum( torch.div(numerator,denominator) )
    elif mode == 'mean':
        error = torch.mean(torch.div(numerator,denominator) )
    elif mode == 'no_normalization':
        error = torch.mean(numerator)
    return error

#########################################################
# SSIM code from pputzky/irim_fastMRI/
#########################################################
def get_uniform_window(window_size, n_channels):
    window = torch.ones(n_channels, 1, window_size, window_size, requires_grad=False)
    window = window / (window_size ** 2)
    return window


def reflection_pad(x, window_size):
    pad_width = window_size // 2
    x = Func.pad(x, [pad_width, pad_width, pad_width, pad_width], mode='reflect')

    return x


def conv2d_with_reflection_pad(x, window):
    x = reflection_pad(x, window_size=window.size(-1))
    x = Func.conv2d(x, window, padding=0, groups=x.size(1))

    return x


def calc_ssim(x1, x2, window, C1=0.01, C2=0.03):
    """
    This function calculates the pixel-wise SSIM in a window-sized area, under the assumption
    that x1 and x2 have pixel values in range [0,1]. The default values for C1 and C2 are chosen
    in accordance with the scikit-image default values
    :param x1: 2d image
    :param x2: 2d image
    :param window: 2d convolution kernel
    :param C1: positive scalar, luminance fudge parameter
    :param C2: positive scalar, contrast fudge parameter
    :return: pixel-wise SSIM
    """
    x = torch.cat((x1, x2), 0)
    mu = conv2d_with_reflection_pad(x, window)
    mu_squared = mu ** 2
    mu_cross = mu[:x1.size(0)] * mu[x1.size(0):]

    var = conv2d_with_reflection_pad(x * x, window) - mu_squared
    var_cross = conv2d_with_reflection_pad(x1 * x2, window) - mu_cross

    luminance = (2 * mu_cross + C1 ** 2) / (mu_squared[:x1.size(0)] + mu_squared[x1.size(0):] + C1 ** 2)
    contrast = (2 * var_cross + C2 ** 2) / (var[:x1.size(0)] + var[x1.size(0):] + C2 ** 2)
    ssim_val = luminance * contrast
    ssim_val = ssim_val.mean(1, keepdim=True)

    return ssim_val

def ssim_uniform(input, target, window_size=11, reduction='mean'):
    """
    Calculates SSIM using a uniform window. This approximates the scikit-image implementation used
    in the fastMRI challenge. This function assumes that input and target are in range [0,1]
    input format: NCHW
    :param input: 2D image tensor
    :param target: 2D image tensor
    :param window_size: integer
    :param reduction: 'mean', 'sum', or 'none', see pytorch reductions
    :return: ssim value
    """
    window = get_uniform_window(window_size, input.size(1))
    window = window.to(input.device)
    ssim_val = calc_ssim(input, target, window)
    if reduction == 'mean':
        ssim_val = ssim_val.mean()
    elif not (reduction is None or reduction == 'none'):
        ssim_val = ssim_val.sum()

    return ssim_val

# def hfen(x,xstar,base=24):
#     '''
#     compute hfen between x and xstar
#     by default, x.shape = [batchsize,heg,wid]
#     the input can also be image pairs with shape x.shape = [heg,wid]
#     '''
#     assert(x.shape==xstar.shape)
#     if len(x.shape)==3:
#         Heg = x.shape[1]
#         y     = F.fftn(x,dim=(1,2),norm='ortho')
#         ystar = F.fftn(xstar,dim=(1,2),norm='ortho')
#         y_high     = y[:,base//2:Heg-1-base//2-base%2,:]
#         ystar_high = ystar[:,base//2:Heg-1-base//2-base%2,:]
#         return torch.norm(y_high - ystar_high,'fro')/torch.norm(ystar_high,'fro')
#     elif len(x.shape)==2:
#         Heg = x.shape[0]
#         y     = F.fftn(x,dim=(0,1),norm='ortho')
#         ystar = F.fftn(xstar,dim=(0,1),norm='ortho')
#         y_high     = y[base//2:Heg-1-base//2-base%2,:]
#         ystar_high = ystar[base//2:Heg-1-base//2-base%2,:]
#         return torch.norm(y_high - ystar_high,'fro')/torch.norm(ystar_high,'fro')


def mnet_getinput(mnet,data,base=8,budget=32,batchsize=10,unet_channels=1,return_mask=False,datatype=torch.float,device=torch.device('cpu')):
    '''
    assume the input data has the dimension [img,heg,wid]
    returned data in the format [NCHW]
    '''   
    mnet.eval()
    mnet.to(device)
    with torch.no_grad():
        lowfreqmask = mask_naiveRand(data.shape[1],fix=base,other=0,roll=False)[0]
        heg,wid  = data.shape[1],data.shape[2]
        imgshape = (heg,wid)
        yfull = F.fftn(data,dim=(1,2),norm='ortho')
        y = torch.zeros_like(yfull)
        y[:,lowfreqmask==1,:] = yfull[:,lowfreqmask==1,:]    
        x_ifft = torch.zeros(len(yfull),unet_channels,heg,wid,device=torch.device('cpu'))
        if return_mask:
            masks = torch.zeros(len(yfull),heg,device=device)

        batchind  = 0
        batchnums = int(np.ceil(data.shape[0]/batchsize))
        while batchind < batchnums:
            batch = torch.arange(batchsize*batchind, min(batchsize*(batchind+1),data.shape[0]))
            yfull_b = yfull[batch,:,:].to(device)
            y_lf    = y[batch,:,:].to(device)
            y_in    = torch.zeros(len(batch),2,heg,wid,device=device)
            y_in[:,0,:,:] = torch.real(y_lf)
            y_in[:,1,:,:] = torch.imag(y_lf)
            y_in   = F.fftshift(y_in,dim=(2,3))
            mask_b = F.ifftshift(mnet_wrapper(mnet,y_in,budget,imgshape,normalize=True,detach=True,device=device),dim=(1))

            if return_mask:
                masks[batch,:] = mask_b        
            y_mnet_b = torch.zeros_like(yfull_b,device=device)
            for ind in range(len(mask_b)):
                y_mnet_b[:,mask_b[ind,:]==1,:] = yfull_b[:,mask_b[ind,:]==1,:]
            if   unet_channels == 1:
                x_ifft[batch,0,:,:] = torch.abs(F.ifftn(y_mnet_b,dim=(1,2),norm='ortho').to(datatype)).cpu()
            elif unet_channels == 2:
                x_ifft_c = F.ifftn(y_mnet_b,dim=(1,2),norm='ortho')
                x_ifft[batch,0,:,:] = torch.real(x_ifft_c).to(datatype).cpu()
                x_ifft[batch,1,:,:] = torch.imag(x_ifft_c).to(datatype).cpu()
            batchind += 1
        
    if return_mask:
        return x_ifft, masks
    else:
        return x_ifft
    
def base_getinput(data,base=8,budget=32,batchsize=5,unet_channels=1,datatype=torch.float,mode='rand'):
    '''
    assume the input data has the dimension [img,heg,wid]
    returned data in the format [NCHW]
    '''   
    yfull = F.fftn(data,dim=(1,2),norm='ortho')
    y_lf  = torch.zeros_like(yfull)
    num_pts,heg,wid = data.shape[0],data.shape[1],data.shape[2]
    batchind  = 0
    batchnums = int(np.ceil(num_pts/batchsize))      
    while batchind < batchnums:
        batch  = torch.arange(batchind*batchsize,min((batchind+1)*batchsize,num_pts))
        if   mode == 'rand':
            lfmask = mask_naiveRand(data.shape[1],fix=base,other=budget,roll=False)[0] 
        if   mode == 'lfonly':
            lfmask = mask_naiveRand(data.shape[1],fix=base+budget,other=0,roll=False)[0]
        elif mode == 'equidist':
            lfmask = mask_equidist(data.shape[1],fix=base,other=budget,roll=False)
        batchdata_full = yfull[batch,:,:]
        batchdata      = torch.zeros_like(batchdata_full)
        batchdata[:,lfmask==1,:] = batchdata_full[:,lfmask==1,:]
        y_lf[batch,:,:] = batchdata
        batchind += 1
    
    if unet_channels == 2:                
        x_ifft = F.ifftn(y_lf,dim=(1,2),norm='ortho')
        x_in   = torch.zeros((num_pts,2,heg,wid),dtype=datatype)
        x_in[:,0,:,:] = torch.real(x_ifft)
        x_in[:,1,:,:] = torch.imag(x_ifft)       
    elif unet_channels == 1:
        x_ifft = torch.abs(F.ifftn(y_lf,dim=(1,2),norm='ortho'))                
        x_in   = torch.reshape(x_ifft, (num_pts,1,heg,wid)).to(datatype)
    
    return x_in


def loupe_eval(loupe,testdata,preselect_num,sparsity,\
               batchsize=5,\
               mode='unet',Lambda=1e-4,\
               datatype=torch.float,device=torch.device('cpu')):
    loupe.to(device)
    loupe.eval()
    if preselect_num > 0:
        assert loupe.preselect
        assert loupe.preselect_num == preselect_num
    
    n_test = testdata.shape[0]; heg = testdata.shape[1]; wid = testdata.shape[2]  
    for ind in range(n_test): # normalize data
        testdata[ind,:,:] = testdata[ind,:,:]/torch.max(torch.abs(testdata[ind,:,:]))
    print('test data size:', testdata.shape)    
    testdata  = testdata.unsqueeze(1)
    
    batchnums = int(np.ceil(n_test/batchsize))
    batchind  = 0    
  
    if mode == 'unet':          
        l1err = torch.zeros(n_test)
        l2err = torch.zeros(n_test)
        hfens = torch.zeros(n_test)
        ssims = torch.zeros(n_test)
        psnrs = torch.zeros(n_test)
        with torch.no_grad():
            while batchind < batchnums:
                batch = np.arange(batchsize*batchind, min(batchsize*(batchind+1),n_test))
                xstar = testdata[batch].to(datatype).to(device)
                ystar = F.fftn(xstar,dim=(2,3),norm='ortho')
                x,_   = loupe(ystar)
                x     = torch.squeeze(x.detach())
                xstar = torch.squeeze(xstar)
                l1err[batch] = compute_l1err(x,xstar)
                l2err[batch] = compute_l2err(x,xstar)
                hfens[batch] = torch.tensor(compute_hfen(x,xstar),dtype=datatype)
                ssims[batch] = ssim_uniform(torch.unsqueeze(x,1),torch.unsqueeze(xstar,1),reduction='mean').cpu().item()
        #         ssims[batch] = torch.tensor(compute_ssim(x,xstar))
                psnrs[batch] = compute_psnr(x,xstar)

                batchind += 1
        return l1err,l2err,hfens,ssims,psnrs
    
    elif mode == 'sigpy':
        pred_loupe  = torch.zeros((n_test,heg,wid)) 
        masks_loupe = torch.zeros((batchnums,heg))
        while batchind < batchnums:
            batch = np.arange(batchsize*batchind, min(batchsize*(batchind+1),n_test))
            databatch = data[batch]
            _,mask = loupe.samplers[0](databatch,sparsity)

            mask = torch.squeeze(mask.detach())
            masks_loupe[batchind,:] = mask
            observed_kspace = torch.zeros_like(databatch)
            imgs_recon = torch.zeros((len(databatch),heg,wid))
            observed_kspace[:,:,mask==1,:] = databatch[:,:,mask==1,:]
            
            mps = np.ones((1,heg,wid))
            for ind in range(len(observed_kspace)):
                y_tmp = observed_kspace[ind,0,:,:].view(-1,heg,wid).numpy()
                imgs_recon[ind,:,:] = torch.tensor(\
                           np.fft.ifftshift(np.abs(TotalVariationRecon(y_tmp, mps, Lambda,show_pbar=False).run())) )
            pred_loupe[batch] = imgs_recon
            batchind += 1
        
        ssim = compute_ssim(pred_loupe,testdata)
        psnr = compute_psnr(pred_loupe,testdata)
        hfen = np.zeros((n_test))
        for ind in range(n_test):
            hfen[ind] = compute_hfen(pred_loupe[ind,:,:].to(torch.cfloat),testdata[ind,:,:].to(torch.cfloat))
        rmse = np.zeros((n_test))
        for ind in range(n_test):
            rmse[ind] = torch.norm(pred_loupe[ind,:,:] - testdata[ind,:,:],2)/torch.norm(testdata[ind,:,:],2)
        return ssim, psnr, hfen, rmse

# mnet eval
def mnet_eval(testdata,mnet,model,base,budget,batchsize=25,datatype=torch.float,device=torch.device('cpu')):
    mnet.to(device)
    model.to(device)
    n_test = testdata.shape[0]
    for ind in range(n_test):
        testdata[ind,:,:] = testdata[ind,:,:]/torch.max(torch.abs(testdata[ind,:,:]))
    print('test data size:', testdata.shape)
    
    batch_nums  = int(np.ceil(n_test/batchsize))
    model_in_chans = model.in_chans if ((model.__class__.__name__=='Unet') or (model.__class__.__name__=='UNet')) else model.model.in_chans
    test_in, mask_in = mnet_getinput(mnet,testdata,base=base,budget=budget,batchsize=batchsize,unet_channels=model_in_chans,return_mask=True,device=device)
    
    l1err = torch.zeros(n_test)
    l2err = torch.zeros(n_test)
    hfens = torch.zeros(n_test)
    ssims = torch.zeros(n_test)
    psnrs = torch.zeros(n_test)
    
    batchind  = 0
    data_fidelity_loss = 0
    ssim_loss = 0
    with torch.no_grad():
        while batchind<batch_nums:
            batch = torch.arange(batchsize*batchind, min(batchsize*(batchind+1),testdata.shape[0]))
            xstar = testdata[batch,:,:].to(datatype).to(device)
            x_in  = test_in[batch].to(device)
            
            if ((model.__class__.__name__ == 'Unet') or (model.__class__.__name__ == 'UNet')):
                x = torch.squeeze(model(x_in).detach())
            elif model.__class__.__name__ == 'MoDL':
                if len(mask_in.shape) == 2: # input mask is in the format of NH
                    mask_test_in = copy.deepcopy(mask_in[batch])
                    mask_test    = torch.zeros(mask_test_in.shape[0],x_in.shape[2],x_in.shape[3],device=device)
                    for ind in range(len(mask_test)):
                        mask_tmp = mask_test_in[ind].unsqueeze(1).repeat(1,x_in.shape[3])
                        mask_test[ind] = mask_tmp
                    mask_test = mask_test.unsqueeze(1).repeat(1,2,1,1)
                    smap = torch.ones(x_in.shape,device=device).unsqueeze(1)
                x = model(x_in,mask=mask_test,smap=smap).detach()[:,0,:,:]
            data_fidelity_loss += lpnorm(x.unsqueeze(1),xstar.unsqueeze(1),p='fro',mode='sum')
            ssim_loss          += -ssim_uniform(x.unsqueeze(1),xstar.unsqueeze(1),reduction='mean')
            # to implement various criteria
            l1err[batch] = compute_l1err(x,xstar)
            l2err[batch] = compute_l2err(x,xstar)
            hfens[batch] = torch.tensor(compute_hfen(x,xstar),dtype=datatype)
            ssims[batch] = ssim_uniform(torch.unsqueeze(x,1),torch.unsqueeze(xstar,1),reduction='mean').cpu().item()
    #         ssims[batch] = torch.tensor(compute_ssim(x,xstar))
            psnrs[batch] = compute_psnr(x,xstar)

            batchind += 1
        df_loss_epoch   = data_fidelity_loss.item()/n_test
        ssim_loss_epoch = ssim_loss.item()/batch_nums
        print(f'\t data fidelity loss: {df_loss_epoch:.4f} / 0, ssim loss: {ssim_loss_epoch:.4f} / -1')
    return l1err,l2err,hfens,ssims,psnrs

# baseline eval
def baseline_eval(testdata,model,base,budget,batchsize=25,mode='rand',datatype=torch.float,device=torch.device('cpu')):
    
    model.to(device)
    for ind in range(testdata.shape[0]):
        testdata[ind,:,:] = testdata[ind,:,:]/torch.max(torch.abs(testdata[ind,:,:]))
    print('test data size:', testdata.shape)
    batch_nums  = int(np.ceil(testdata.shape[0]/batchsize))
    
    test_in = base_getinput(testdata,base=base,budget=budget,batchsize=batchsize,unet_channels=model.in_chans,datatype=torch.float,mode=mode)
    labels = testdata.to(datatype)
    del testdata  
    
    l1err = torch.zeros(test_in.shape[0])
    l2err = torch.zeros(test_in.shape[0])
    hfens = torch.zeros(test_in.shape[0])
    ssims = torch.zeros(test_in.shape[0])
    psnrs = torch.zeros(test_in.shape[0])
    
    batchind = 0
    with torch.no_grad():
        while batchind<batch_nums:
            batch = torch.arange(batchsize*batchind, min(batchsize*(batchind+1),test_in.shape[0]))
            x_in  = test_in[batch].to(device)
            x     = torch.squeeze(model(x_in).detach())
            xstar = labels[batch].to(device)
            # to implement various criteria
            l1err[batch] = compute_l1err(x,xstar)
            l2err[batch] = compute_l2err(x,xstar)
            hfens[batch] = torch.tensor(compute_hfen(x,xstar),dtype=torch.float)
            ssims[batch] = ssim_uniform(torch.unsqueeze(x,1),torch.unsqueeze(xstar,1),reduction='mean').cpu().item()
    #         ssims[batch] = torch.tensor(compute_ssim(x,xstar))
            psnrs[batch] = compute_psnr(x,xstar)

            batchind += 1
    return l1err,l2err,hfens,ssims,psnrs