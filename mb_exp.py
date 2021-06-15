import numpy as np
import argparse
import os
import sys
import random
import torch
import torch.fft as F
from importlib import reload
from torch.nn.functional import relu
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import utils,mask_backward_new
import matplotlib.pyplot as plt
# from maskbackward import mask_backward
from mask_backward_new import mask_backward, mask_eval
from utils import mask_complete , mask_makebinary, kplot, mask_filter, raw_normalize,\
                    get_x_f_from_yfull, mask_naiveRand, apply_mask, sigmoid_binarize, compute_hfen

sys.path.insert(0,'/home/huangz78/mri/unet/')
from unet_model import UNet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    imgs   = torch.tensor( np.load('/home/huangz78/data/data_gt.npz')['imgdata'] ).permute(2,0,1)
    masks  = torch.tensor( np.load('/home/huangz78/data/data_gt_greedymask.npz')['mask'].T ) # labels are already rolled
    imgNum = imgs.shape[0]
    traininds, testinds = train_test_split(np.arange(imgNum),random_state=0,shuffle=True,train_size=round(imgNum*0.8))


    ### load a mnet
    from mnet import MNet
    # mnet = MNet(out_size=320-24)
    mnet = MNet(beta=1,in_channels=2,out_size=320-24, imgsize=(320,320),poolk=3)
    mnetpath = '/home/huangz78/checkpoints/mnet.pth'
    checkpoint = torch.load(mnetpath)
    mnet.load_state_dict(checkpoint['model_state_dict'])
    print('MNet loaded successfully from: ' + mnetpath)
    mnet.eval()

    ### load a unet for maskbackward
    UNET = UNet(n_channels=1,n_classes=1,bilinear=True,skip=False)
    # unetpath = '/home/huangz78/checkpoints/unet_'+ str(UNET.n_channels) +'.pth'
    unetpath = '/home/huangz78/checkpoints/unet_1_False.pth'

    # UNET = UNet(n_channels=1,n_classes=1,bilinear=False,skip=True)
    # unetpath = '/home/huangz78/checkpoints/unet_1_True.pth'
    checkpoint = torch.load(unetpath)
    UNET.load_state_dict(checkpoint['model_state_dict'])
    print('Unet loaded successfully from: ' + unetpath )
    UNET.train()
    print('nn\'s are ready')

    # batchsize = 5
    # xstar = xfull[0:batchsize,:,:]
    xstar    = imgs[testinds[testinds.size//2:],:,:]
    for ind in range(xstar.shape[0]):
        xstar[ind,:,:] = xstar[ind,:,:]/xstar[ind,:,:].max()
    full_gredmask = masks[testinds[testinds.size//2:],:]
    corefreq = 24
    budget   = 56
    yfull = torch.fft.fftshift(F.fftn(xstar,dim=(1,2),norm='ortho')).to(torch.cfloat) # y is ROLLED!
    lowfreqmask,_,_ = mask_naiveRand(xstar.shape[1],fix=corefreq,other=0,roll=True)

    z = apply_mask(lowfreqmask,yfull,mode='r')
    highmask = sigmoid_binarize(raw_normalize(mnet(z),budget=budget))
    randmask = torch.zeros(highmask.shape)
    for ind in range(highmask.shape[0]):
        sampinds = np.random.choice(highmask.shape[1],int(highmask[ind,:].sum()),replace=False)
        randmask[ind,sampinds] = 1
    lowfmask,_,_ = mask_naiveRand(xstar.shape[1]-corefreq,fix=torch.sum(highmask[0,:]),other=0,roll=True)
    lowfmask = lowfmask.repeat(highmask.shape[0],1)
    # x_lf     = get_x_f_from_yfull(lowfreqmask,yfull)
    # highmask = sigmoid_binarize(mnet(x_lf.view(batchsize,1,xstar.shape[1],xstar.shape[2])))

    NN         = 10
    alpha_grid = 10**(np.linspace(-5.5,-3,NN))
    c_grid     = np.array([1e-4,1e-3,1e-2,1e-1,1e0])
    l2loss   = np.zeros((NN,5))
    hfen = np.zeros((NN,5))
    sparsity = np.zeros((NN,5))
    ########################################  
    ## (1) mask_backward
    ########################################    
    maxIter_mb = 15
    lr_mb      = 1e-2

    # alpha = 1e-5
    # c = 1e-2

    c_ind = 0
    for c in c_grid:
        print(f'c_ind {c_ind+1} out of {len(c_grid)}')
        a_ind = 0
        for alpha in alpha_grid:
            print(f'alpha_ind {a_ind+1} out of {len(alpha_grid)}')
            print(f'c={c} and alpha={alpha}')
            # load a unet for maskbackward
            UNET = UNet(n_channels=1,n_classes=1,bilinear=True,skip=False)
            unetpath = '/home/huangz78/checkpoints/unet_1_False.pth'
            checkpoint = torch.load(unetpath)
            UNET.load_state_dict(checkpoint['model_state_dict'])
            UNET.train()
        # highmask_refined,unet = mask_backward(highmask,xstar,unet=UNET, mnet=mnet,\
        #                   beta=1.,alpha=alpha,c=c,\
        #                   maxIter=maxIter_mb,seed=0,break_limit=maxIter_mb*3//5,\
        #                   lr=lr_mb,mode='UNET',budget=budget,normalize=False,\
        #                   verbose=True,dtyp=torch.float)
            (l2loss[a_ind,c_ind],hfen[a_ind,c_ind]),sparsity[a_ind,c_ind] =\
                            mask_backward(highmask,xstar,unet=UNET, mnet=mnet,\
                              beta=1.,alpha=alpha,c=c,\
                              maxIter=maxIter_mb,seed=0,break_limit=np.inf,\
                              lr=lr_mb,mode='UNET',budget=budget,normalize=False,\
                              dtyp=torch.float,verbose=True,testmode='sigpy',hfen=True)
            a_ind += 1
        print('\n')
        c_ind += 1
        np.savez('home/huangz78/checkpoints/mb_rec.npz',l2loss=l2loss,hfen=hfen,sparsity=sparsity)
