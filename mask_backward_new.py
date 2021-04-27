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

from utils import mask_complete , mask_makebinary, kplot
from solvers import ADMM_TV

sys.path.insert(0,'/home/huangz78/mri/unet/')
from unet_model import UNet

def proj_eps(M,eps):
    with torch.no_grad(): 
        M[(M<=0.5)&(M>eps)] = eps
        M[(M>0.5)&(M<1-eps)] = 1-eps
        M[(M<0)] = 0
        M[(M>1)] = 1
        return M
    
    
def mask_eval(fullmask,xstar,mode='UNET',\
              UNET=None,dtyp=torch.float,\
              Lambda=10**(-6.31),rho=1e1,unroll_block=8):
    '''
    Does the predicted mask work better than the random baseline?
    check method: with respect to a given mask, push them through the same reconstructor, 
                  compare the reconstructed image in l2 norm
    '''

    imgHeg = xstar.shape[0]; imgWid = xstar.shape[1]
    if isinstance(fullmask,torch.Tensor):
        fullmask = fullmask.numpy()
    if isinstance(xstar,torch.Tensor):
        xstar = xstar.numpy()
    yraw = np.fft.fftshift(np.fft.fftn(xstar,norm='ortho')) # roll
    y = np.diag(fullmask)@yraw    

    if mode=='ADMM':
        x = ADMM_TV(torch.tensor(y),torch.tensor(fullmask),maxIter=unroll_block,Lambda=Lambda,rho=rho,imgInput=False,x_init=None)
        x = x.numpy()
    elif mode=='UNET':
        if UNET.n_channels == 2:
            y = torch.tensor(y)
            ycp = torch.zeros((1,2,imgHeg,imgWid),dtype=dtyp) # batchsize is 1 here
            ycp[:,0,:,:] = torch.reshape(torch.real(y),(1,imgHeg,imgWid))
            ycp[:,1,:,:] = torch.reshape(torch.imag(y),(1,imgHeg,imgWid))
            yprime = UNET(ycp)
            yrecon = yprime[:,0,:,:] + 1j*yprime[:,1,:,:]
            x = torch.abs(F.ifftn(torch.reshape(yrecon,(imgHeg,imgWid)),dim=(0,1),norm='ortho'))
            x = x.detach().numpy()
        elif UNET.n_channels == 1:
            x_ifft = np.abs( np.fft.ifftn(np.fft.ifftshift(y),norm='ortho') ) # undo roll
            x_in = torch.tensor(x_ifft,dtype=dtyp).view(1,1,imgHeg,imgWid)
            x = UNET(x_in).detach().numpy()
    elif mode=='sigpy':
        from sigpy.mri.app import TotalVariationRecon
        mps = np.ones((1,imgHeg,imgWid))
        y   = np.reshape(y,(-1,imgHeg,imgWid))
        x   = np.fft.fftshift( np.abs(TotalVariationRecon(y, mps, Lambda,show_pbar=False).run()) )    
#     plt.clf();plt.imshow(np.abs(x_ifft));plt.colorbar();plt.show()
#     print('x_ifft error: ', np.sqrt( np.sum((x_ifft.flatten()-xstar.flatten())**2) )/np.sqrt( np.sum( (xstar.flatten())**2 )))
    error = np.sqrt( np.sum((x.flatten()-xstar.flatten())**2) )/np.sqrt( np.sum( (xstar.flatten())**2 ))
    return error

def mask_backward(highmask,xstar,\
                  beta=1, alpha=5e-1,maxIter=300,seed=0,lr=1e-3,lr_Lambda=1e-8,eps=1.,\
                  unroll_block=8,Lambda=10**(-6.5),rho=1e2,mode='ADMM',unet_mode=1,unet=None,\
                  break_limit=20,print_every=10,\
                  verbose=False,save_cp=False,dtyp=torch.double):
    '''
    The purpose of this function is to update mask choice (particularly for high frequency) via backward propagation. 
    The input is one image, the known base ratio and the currently employed high frequency mask for the input image.
    The output is the updated mask.
    
    xstar                : ground truth image
    highmask             : binary vector, indicating which high frequencies to sample (low frequencies assumed to be central), the initial mask to be polished
    Lambda               : ADMM parameter lambda (magnitude of TV penalty)
    rho                  : ADMM parameter rho
    mode                 : 'ADMM' for actual usage, 'IFFT' for debugging
    maxIter              : max iterations for backward-prop update step
    unroll_block         : how many blocks of solver to unroll, default 3
    lr                   : learning rate to update mask indicator, default 1e-3
    lr_Lambda            : learning rate to update the ADMM parameter Lambda
    alpha                : l1 penalty magnitude when selecting high frequency masks
    seed                 : random seed, default 0
    break_limit          : if no change in loss or row selection for this many iteration rounds, then break
    eps                  : hard thresholding value [0,eps] union [1-eps,1], default 1
    
    -- disabled args
    perturb              : flag to inject noise into gradients when update the mask, currently disabled!
    perturb_freq         : how frequently to inject noise, currently disabled!
    '''
    torch.manual_seed(seed)
    if dtyp == torch.double:
        DType = torch.cdouble
    else:
        DType = torch.cfloat
    imgHeg,imgWid = xstar.shape[0],xstar.shape[1]
    xstar = torch.tensor(xstar,dtype=dtyp); highmask = torch.tensor(highmask,dtype=dtyp)
    xstar = xstar/torch.max(torch.abs(xstar.flatten()))
    y = torch.roll(F.fftn(xstar,dim=(0,1),norm='ortho'),shifts=(imgHeg//2,imgWid//2),dims=(0,1))
    ## initialising M
    M_high = highmask.clone().detach()
    M_high.requires_grad = True
    fullmask = torch.tensor( mask_complete(M_high,imgHeg,dtyp=dtyp) ) 
    fullmask_b = fullmask.clone()
    
    torch.autograd.set_detect_anomaly(False)
#     optimizer = optim.Adagrad([M_high],lr=lr)
    if mode == 'UNET':
        if unet is None:
            UNET =  UNet(n_channels=unet_mode,n_classes=unet_mode,bilinear=True,skip=False)
            checkpoint = torch.load('/home/huangz78/mri/checkpoints/unet_'+ str(UNET.n_channels) +'.pth')
            UNET.load_state_dict(checkpoint['model_state_dict'])
            print('Unet loaded successfully from : ' + '/home/huangz78/mri/checkpoints/unet_'+ str(UNET.n_channels) +'.pth' )
            UNET.train()
            optimizer = optim.RMSprop([
                    {'params': M_high},
                    {'params': UNET.parameters(),'lr':1e-4}
                ], lr=lr, weight_decay=1e-8, momentum=0.9)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
            init_mask_loss = mask_eval(fullmask_b,xstar,mode='UNET',UNET=UNET,dtyp=dtyp) * 100
            print('loss of the input mask: ', init_mask_loss)
        else:
            UNET = unet
            UNET.train()
            optimizer = optim.RMSprop([
                    {'params': M_high},
                    {'params': UNET.parameters(),'lr':1e-4}
                ], lr=lr, weight_decay=1e-8, momentum=0.9)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    else:
        optimizer = optim.SGD([M_high],lr=lr)
        print('loss of the input mask: ', mask_eval(fullmask_b,xstar,unroll_block=unroll_block,Lambda=Lambda,rho=rho,dtyp=dtyp) * 100)
#     optimizer = optim.SGD([{'params':M_high,'lr':lr},{'params':Lambda,'lr':lr_Lambda}])
    repCount = 0; rCount = 0
    flag_perturb = False; perturbList  = list([])  
  
    Iter = 0; loss_old = np.inf; x = None
    cr_per_batch = 0
    while Iter < maxIter:
        z = torch.roll(torch.tensordot(torch.diag(fullmask).to(DType),y,dims=([1],[0])) , \
                       shifts=(imgHeg//2,imgWid//2),dims=(0,1))# need to fftshift y and then fftshift y back
        ## Reconstruction process
        if mode == 'ADMM':
            x   = ADMM_TV(z,fullmask,maxIter=unroll_block,Lambda=Lambda,rho=rho,imgInput=False,x_init=None)
#             x = learnable_unrolled_algo(z,fullmask) # maybe PDHG
        if mode == 'UNET':
            if UNET.n_channels == 2:
                zcp = torch.zeros((1,2,imgHeg,imgWid),dtype=dtyp) # batchsize is 1 here
                zcp[:,0,:,:] = torch.reshape(torch.real(z),(1,imgHeg,imgWid))
                zcp[:,1,:,:] = torch.reshape(torch.imag(z),(1,imgHeg,imgWid))
                zprime = UNET(zcp)
                zrecon = zprime[:,0,:,:] + 1j*zprime[:,1,:,:]
                x = torch.abs(F.ifftn(torch.reshape(zrecon,(imgHeg,imgWid)),dim=(0,1),norm='ortho'))
            elif UNET.n_channels == 1:
                x_ifft = torch.abs( F.ifftn(z,dim=(0,1),norm='ortho') ) # undo roll
                x_in = x_ifft.view(1,1,imgHeg,imgWid).to(dtyp)
                x = UNET(x_in)
        elif mode == 'IFFT': # debug
            x   = torch.abs(F.ifftn(z,dim=(0,1),norm='ortho')) # should involve the mask to cater for the lower-level objective
                
        loss = torch.sqrt(torch.sum((torch.flatten(x)-torch.flatten(xstar) )**2))/torch.sqrt(torch.sum(torch.flatten(xstar)**2)) + alpha*torch.norm(M_high,p=1) ## upper-level loss = nrmse + alpha*||M||_1
        if loss.item() < loss_old:
            loss_old = loss.item()
            repCount = 0
        elif loss.item() >= loss_old:
            repCount += 1
#             if repCount >= perturb_freq:
#                 flag_perturb = True
            if repCount >= break_limit:
                print('No further decrease in loss after {} consecutive iters, ending iterations~ '.format(repCount))
                break

        optimizer.zero_grad()
        loss.backward()    
        fullmask_old = mask_makebinary(fullmask.detach().numpy(),threshold=0.5,sigma=False)        
        optimizer.step()
        M_high = proj_eps(M_high,eps) # soft-hard-thresholding as postprocessing
        fullmask = mask_complete(M_high,imgHeg,dtyp=dtyp)
        
        fullmask_b = mask_makebinary(fullmask.clone().detach(),threshold=0.5,sigma=False)
        mask_sparsity = torch.sum(fullmask_b).item()/fullmask_b.size()[0]
        changed_rows = np.abs(fullmask_old-fullmask_b).sum().item()
        cr_per_batch += changed_rows
        if changed_rows == 0:
            rCount += 1
        else:
            rCount = 0
        if rCount > break_limit:
            print('No change in row selections after {} iters, ending iteration~'.format(rCount))
            break
        if verbose and (changed_rows>0): # if there is any changed rows, then it is reported in every iteration
            print('Iter {}, changed rows: {}'.format(Iter,changed_rows))
        if verbose and (Iter % print_every == 0): # every print_every iters, print the quality and sparsity of the current mask
            # or we can print only 10 times: max(maxIter//10,1)
            with torch.no_grad(): ## Validation
                if cr_per_batch>0: # (changed_rows>0):
                    if mode == 'ADMM':
                        mask_loss = mask_eval(fullmask_b,xstar,unroll_block=unroll_block,Lambda=Lambda,rho=rho,dtyp=dtyp) * 100
                    elif mode=='UNET':
                        mask_loss = mask_eval(fullmask_b,xstar,mode='UNET',UNET=UNET,dtyp=dtyp) * 100
                    print('iter: {}, upper level loss: {}\n changed rows in this batch: {}, loss of current mask: {}'.format(Iter+1,loss,cr_per_batch,mask_loss))
                    print('samp. ratio: {}, Recon. rel. err: {} \n'.format(mask_sparsity,(torch.norm(torch.flatten(x)-torch.flatten(xstar),'fro')/torch.norm(torch.flatten(xstar),'fro')).item()) )
    #                 kplot(fullmask_b)           # plot the current mask
                    cr_per_batch = 0
                    scheduler.step(mask_loss)
                else:
                    print('iter: {}, upper level loss: {}'.format(Iter+1,loss))   
                
        if save_cp and (Iter%max(maxIter//10,1))==0:
            dir_checkpoint = '/home/huangz78/mri/checkpoints/'
            torch.save({'model_state_dict': UNET.state_dict()}, dir_checkpoint + 'unet_'+ str(UNET.n_channels) +'_by_mask.pth')
            print(f'\t Checkpoint saved after Iter {Iter + 1}!')
        Iter += 1
    
    highmask_refined = mask_makebinary(M_high,threshold=0.5,sigma=False)
    print('\nreturn at Iter: ', Iter)
    
    if mode=='ADMM':
        mask_loss = mask_eval(mask_complete(highmask_refined,imgHeg,dtyp=dtyp),xstar,unroll_block=unroll_block,Lambda=Lambda,rho=rho,dtyp=dtyp) * 100
    elif mode=='UNET':
        mask_loss = mask_eval(mask_complete(highmask_refined,imgHeg,dtyp=dtyp),xstar,mode='UNET',UNET=UNET,dtyp=dtyp) * 100

        
    print('loss of returned mask: ',mask_loss)
    if unet is None:
        return highmask_refined, mask_loss, init_mask_loss
    else:
        return highmask_refined, UNET

# arxived code blocks:
#         perturb=False, perturb_freq=2 
#         if flag_perturb and perturb: # purturbation
#             perturbList.append(Iter)
#             r = torch.randn(M_high.shape)
#             r = r/torch.norm(r)
#             M_high.grad = M_high.grad + 1/lr *1/5*torch.max(torch.abs(M_high)) * r
#             flag_perturb = False
#             repCount = 0
#     return MASK,Lambda,perturbList