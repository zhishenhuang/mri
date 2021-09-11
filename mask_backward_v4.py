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
import matplotlib.pyplot as plt
from sigpy.mri.app import TotalVariationRecon
from utils import *
from solvers import ADMM_TV
import copy

sys.path.insert(0,'/home/huangz78/mri/unet/')
from unet_model import UNet

from torch.autograd import Function
class ThresholdBinarizeMask(Function):
    def __init__(self):
        """
            Straight through estimator.
            The forward step binarizes the real-valued mask.
            The backward step estimate the non differentiable > operator using sigmoid with large slope (1).
        """
        super(ThresholdBinarizeMask, self).__init__()

    @staticmethod
    def forward(ctx, input):
        batch_size = len(input)
        results = [] 

        for i in range(batch_size):
            x = input[i:i+1]
            result = (x > .5).float()
            results.append(result)

        results = torch.cat(results, dim=0)
#         ctx.save_for_backward(input)
        return results  

    @staticmethod
    def backward(ctx, grad_output):
        slope = 1
#         input = ctx.saved_tensors

        # derivative of M
        current_grad = slope

        return current_grad * grad_output
    
def nrmse(x,xstar):
    '''
    input should be torch tensors
    '''
    return torch.sqrt(torch.sum((torch.flatten(x)-torch.flatten(xstar) )**2))/torch.sqrt(torch.sum(torch.flatten(xstar)**2))   
    
def mask_eval(fullmask,xstar,\
              mode='UNET',UNET=None,dtyp=torch.float,\
              Lambda=10**(-4.3),hfen=False,device='cpu'): # done
    '''
    The input mask is assumed to be rolled.
    
    Does the predicted mask work better than the random baseline?
    check method: with respect to a given mask, push them through the same reconstructor, 
                  compare the reconstructed image in l2 norm
    the lower the output is, the better the mask is
    '''
    with torch.no_grad():
        if UNET is not None:
            UNET.eval()
        batchsize = xstar.shape[0]; imgHeg = xstar.shape[1]; imgWid = xstar.shape[2]
        for layer in range(batchsize):
            xstar[layer,:,:] = xstar[layer,:,:]/torch.max(torch.abs(xstar[layer,:,:].flatten()))
        y = torch.fft.fftshift(F.fftn(xstar,dim=(1,2),norm='ortho'),dim=(1,2))    
        z = torch.zeros(y.shape,device=device).to(y.dtype)
        for ind in range(batchsize):
            z[ind,fullmask[ind,:]==1,:] = y[ind,fullmask[ind,:]==1,:]   
        if mode=='UNET' and (UNET is not None):
            z = torch.fft.ifftshift(z , dim=(1,2)) 
            if UNET.n_channels == 2:
                x_ifft = F.ifftn(z,dim=(1,2),norm='ortho') 
                x_in   = torch.zeros((batchsize,2,imgHeg,imgWid),dtype=dtyp,device=device)
                x_in[:,0,:,:] = torch.real(x_ifft)
                x_in[:,1,:,:] = torch.imag(x_ifft)           
            elif UNET.n_channels == 1:
                x_ifft = torch.abs( F.ifftn(z,dim=(1,2),norm='ortho') ) 
                x_in   = x_ifft.view(batchsize,1,imgHeg,imgWid).to(dtyp)
            x = UNET(x_in).detach()

            if hfen:
                hfens = np.zeros(batchsize)
                for ind in range(batchsize):
                    hfens = compute_hfen(x[ind,0,:,:],xstar[ind,:,:])
                error = (nrmse(x,xstar).detach().item(),np.mean(hfens))
            else:
                error = nrmse(x,xstar).detach().item()

        elif mode == 'sigpy': # mode is 'sigpy'
            mps = np.ones((1,imgHeg,imgWid))
            err = np.zeros(batchsize)
            hfens = np.zeros(batchsize)
            xstar = xstar.numpy()
            for ind in range(batchsize):
                y_tmp    = z[ind,:,:].view(-1,imgHeg,imgWid).numpy()
                x_tmp    = np.fft.ifftshift( np.abs(TotalVariationRecon(y_tmp, mps, Lambda,show_pbar=False).run()) )  
                err[ind] = np.linalg.norm(x_tmp - xstar[ind,:,:],'fro')/np.linalg.norm(xstar[ind,:,:],'fro')
                if hfen:
                    hfens[ind] = compute_hfen(x_tmp,xstar[ind,:,:])  
            if hfen:
                error = (np.mean(err),np.mean(hfens))
            else:
                error = np.mean(err)
        if UNET is not None:
            UNET.train()
    return error


def mask_backward(highmask,xstar,\
                  maxIter=300,seed=0,eps=1.,normalize=True,\
                  budget=20,\
                  lr=1e-3,weight_decay=0,momentum=0,\
                  beta=1, alpha=5e-1,c =.1,slope=1,\
                  mode='UNET',unet_mode=1,unet=None,mnet=None,\
                  break_limit=np.inf,print_every=10,\
                  verbose=False,save_cp=False,dtyp=torch.float,\
                  testmode=None,hfen=False,return_loss_only=False,\
                  device='cpu'):
    '''
    The purpose of this function is to update mask choice (particularly for high frequency) via backward propagation. 
    The input is one image, the known base ratio and the currently employed high frequency mask for the input image.
    The output is the updated mask.
    
    xstar                : ground truth image
    highmask             : binary vector, indicating which high frequencies to sample (low frequencies assumed to be central, i.e., the highmask is 'rolled'), the initial mask to be polished
    Lambda               : ADMM parameter lambda (magnitude of TV penalty)
    rho                  : ADMM parameter rho
    mode                 : 'UNET' and ADMM' for actual usage, 'IFFT' for debugging
    maxIter              : max iterations for backward-prop update step
    unroll_block         : how many blocks of solver to unroll, default 3
    lr                   : learning rate to update mask indicator, default 1e-3
    lr_Lambda            : learning rate to update the ADMM parameter Lambda
    slope                : slope inside the sigmoid function to output a real-valued mask
    beta                 : the weight for the data fidelity term in the loss function
    alpha                : l1 penalty magnitude when selecting high frequency masks
    c                    : magnitude for consistency term || M - mnet(x_lf) ||_2
    seed                 : random seed, default 0
    break_limit          : if no change in loss or row selection for this many iteration rounds, then break
    eps                  : hard thresholding value [0,eps] union [1-eps,1], default 1
    budget               : targeted sampling budget for the highmask. Note, the budget is only for the high frequencies, not including the observed low frequencies
    normalize            : whether to normalize the output mask to the specified sampling budget
    
    -- disabled args
    perturb              : flag to inject noise into gradients when update the mask, currently disabled!
    perturb_freq         : how frequently to inject noise, currently disabled!
      
    May 18: need to make xstar as a batch because BatchNorm2d will render batchsize=1 constantly zero.
    arxived inputs       : unroll_block=8,Lambda=10**(-6.5),rho=1e2,lr_Lambda=1e-8
    '''
    torch.manual_seed(seed)
    binarize = ThresholdBinarizeMask().apply
    criterion_mnet = nn.BCEWithLogitsLoss()
    torch.autograd.set_detect_anomaly(False)
    
    batchsize,imgHeg,imgWid = xstar.shape[0],xstar.shape[1],xstar.shape[2]
    if not isinstance(xstar,torch.Tensor):
        xstar    = torch.tensor(xstar,dtype=dtyp,device=device)
    if not isinstance(highmask,torch.Tensor):
        highmask = torch.tensor(highmask,dtype=dtyp,device=device)
    for layer in range(batchsize):
        xstar[layer,:,:] = xstar[layer,:,:]/torch.max(torch.abs(xstar[layer,:,:].flatten()))
    y = F.fftshift(F.fftn(xstar,dim=(1,2),norm='ortho'),dim=(1,2))
    
    corefreq = imgHeg - highmask.shape[1]
    lowfreqmask = mask_naiveRand(imgHeg,fix=corefreq,other=0,roll=True)[0].to(device)
    mnet.eval()
    with torch.no_grad():
        imgshape = (imgHeg,imgWid)
        if mnet.in_channels == 1:        
            x_lf      = get_x_f_from_yfull(lowfreqmask,y,device=device)
            mask_pred = mnet_wrapper(mnet,x_lf,budget,imgshape,normalize=True,complete=False,detach=True,device=device)
        elif mnet.in_channels == 2:
            z         = apply_mask(lowfreqmask,y,device=device)
            mask_pred = mnet_wrapper(mnet,z,budget,imgshape,normalize=True,complete=False,detach=True,device=device)
    
    ## initialising M
    maskcontent = torch.unique(highmask)
    if (maskcontent.sum()==1) and (maskcontent.prod()==0): # check if the input mask is binary
        fullmask = mask_complete(highmask,imgHeg,dtyp=dtyp,device=device)
        M_high   = highmask.clone().detach().to(device)
        M_high[highmask==1] = .1 # inverse-sigmoid value
        M_high[highmask==0] = -.1
    else:
        fullmask = binarize(mask_complete( raw_normalize(torch.sigmoid(slope*highmask),budget,device=device),imgHeg,dtyp=dtyp,device=device))
        M_high   = highmask.clone().detach().to(device)
    M_high.requires_grad = True
    fullmask_b = fullmask.clone()
   
    if mode == 'UNET':
        if unet is None:
            unet =  UNet(n_channels=unet_mode,n_classes=unet_mode,bilinear=True,skip=False).to(device)
            checkpoint = torch.load('/home/huangz78/mri/checkpoints/unet_'+ str(unet.n_channels) +'.pth')
            unet.load_state_dict(checkpoint['model_state_dict'])
            print('Unet loaded successfully from : ' + '/home/huangz78/mri/checkpoints/unet_'+ str(unet.n_channels) +'.pth' )
            unet.train()

            optimizer = optim.RMSprop([
                    {'params': M_high},
                    {'params': unet.parameters(),'lr':1e-4}
                ], lr=lr, weight_decay=weight_decay, momentum=momentum,eps=1e-10)
        else:
            unet.train()
            optimizer = optim.RMSprop([
                    {'params': M_high},
                    {'params': unet.parameters(),'lr':1e-4}
                ], lr=lr, weight_decay=weight_decay, momentum=momentum,eps=1e-10)
    unet_init = copy.copy(unet)
    if testmode=='UNET':
        init_mask_loss = mask_eval(fullmask_b.clone().detach(),xstar,mode='UNET',UNET=unet,dtyp=dtyp,hfen=hfen,device=device)
    elif testmode == 'sigpy':
        init_mask_loss = mask_eval(fullmask_b.clone().detach(),xstar,mode='sigpy',hfen=hfen,device='cpu')
    if (testmode is not None) and verbose:
        print('loss of the input mask: ', init_mask_loss)

    repCount = 0; rCount = 0
    Iter = 0; loss_old = np.inf; x = None
    cr_per_batch = 0
    while Iter < maxIter:
        z = torch.zeros(y.shape,device=device).to(y.dtype)
        for ind in range(batchsize):
            z[ind,:,:] = torch.tensordot( torch.diag(fullmask[ind,:]).to(y.dtype),y[ind,:,:],dims=([1],[0]) )
        z = F.ifftshift(z , dim=(1,2)) 
        ## Reconstruction process
        if mode == 'UNET':
            if unet.n_channels == 2:
                x_ifft = F.ifftn(z,dim=(1,2),norm='ortho')
                x_in   = torch.zeros((batchsize,2,imgHeg,imgWid),dtype=dtyp,device=device)
                x_in[:,0,:,:] = torch.real(x_ifft)
                x_in[:,1,:,:] = torch.imag(x_ifft)              
            elif unet.n_channels == 1:
                x_ifft = torch.abs(F.ifftn(z,dim=(1,2),norm='ortho')) 
                x_in   = x_ifft.view(batchsize,1,imgHeg,imgWid).to(dtyp)
            x = unet(x_in)
                
        loss = beta * nrmse(x,xstar) + alpha * torch.norm(torch.sigmoid(slope*M_high),p=1) + c * criterion_mnet(slope*M_high.view(mask_pred.shape),mask_pred) 
    ## upper-level loss = nrmse + alpha * ||Mask_actual||_1 + c * mnet_pred_loss, where the last term is added to enforce consistency between mask_backward and mnet in the iteration process, May 7
    
        if loss.item() < loss_old:
            loss_old = loss.item()
            repCount = 0
        elif loss.item() >= loss_old:
            repCount += 1
            if repCount >= break_limit:
                print('No further decrease in loss after {} consecutive iters, ending iterations~ '.format(repCount))
                break
        optimizer.zero_grad()
        loss.backward()    
        fullmask_old = mask_makebinary(fullmask.cpu().detach().numpy(),threshold=0.5,sigma=False,device='cpu') 
        optimizer.step()
        fullmask = binarize( mask_complete(torch.sigmoid(slope*M_high),imgHeg,dtyp=dtyp,device=device) )
        
        #################################
        ## track training process, and printing information
        #################################       
        fullmask_b    = fullmask.clone().cpu().detach()
        delta_mask    = fullmask_old - fullmask_b
        mask_sparsity = torch.sum(fullmask_b).item()/(batchsize*imgHeg)
        added_rows    = torch.sum(delta_mask==-1).item()/batchsize;   reducted_rows= torch.sum(delta_mask==1).item()/batchsize
        changed_rows  = torch.abs(delta_mask).sum().item()/batchsize
        cr_per_batch += changed_rows
        if changed_rows == 0:
            rCount += 1
        else:
            rCount = 0
        if rCount > break_limit:
#             if verbose:
            print('No change in row selections after {} iters, ending iteration~'.format(rCount))
            break
        if verbose and (changed_rows>0): # if there is any changed rows, then it is reported in every iteration
            print('Iter {}, rows added: {}, rows reducted: {}, current samp. ratio: {}'.format(Iter+1,added_rows,reducted_rows,mask_sparsity))
               
        Iter += 1   
    
    if normalize:
        mask_raw = raw_normalize(torch.sigmoid(slope*M_high),budget,threshold=0.5,device=device)
    else:
        mask_raw = torch.sigmoid(slope*M_high)
    highmask_refined = mask_makebinary(mask_raw,threshold=0.5,sigma=False,device=device)
    mask_sparsity = torch.sum(highmask_refined).item()/(batchsize*imgHeg) + corefreq/imgHeg
    
    ####################################
    # check if refined masks are trivial
    mask_rep_count = 0
    for i in range(len(highmask_refined)):
        for j in range(i+1,len(highmask_refined)):
            if (highmask_refined[i,:] - highmask_refined[j,:]).abs().sum()==0:
                mask_rep_count += 1
    if mask_rep_count > len(highmask_refined)//2: # we get the same mask for than half of cases
        mask_loss = np.inf
        unet = unet_init
    elif cr_per_batch == 0:
        mask_loss = init_mask_loss
        unet = unet_init
    else:
        if testmode=='UNET':
            mask_loss = mask_eval(mask_complete(highmask_refined,imgHeg,dtyp=dtyp,device=device),xstar,mode='UNET',UNET=unet,dtyp=dtyp,hfen=hfen,device=device)
            if mask_loss >= init_mask_loss:
                mask_loss_sigpy = mask_eval(mask_complete(highmask_refined.to('cpu'),imgHeg,dtyp=dtyp,device='cpu'),xstar.to('cpu'),mode='sigpy',hfen=hfen,device='cpu')
                mask_loss = min(mask_loss,mask_loss_sigpy)
        elif testmode == 'sigpy':
            mask_loss = mask_eval(mask_complete(highmask_refined.to('cpu'),imgHeg,dtyp=dtyp,device='cpu'),xstar.to('cpu'),mode='sigpy',hfen=hfen,device='cpu')
    
    if verbose and (testmode is not None):
        print('\nreturn at Iter ind: ', Iter)
        print(f'samp. ratio: {mask_sparsity}, loss of returned mask: {mask_loss} \n')
    
    if return_loss_only:
        return mask_loss, mask_sparsity
    else:
        if unet is None:
            return highmask_refined, mask_loss, init_mask_loss
        else:
            return highmask_refined, unet, mask_loss, init_mask_loss

