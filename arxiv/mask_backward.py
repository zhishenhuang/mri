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
DType = torch.cdouble

def proj_eps(M,eps):
    M[(M<=0.5)&(M>eps)] = eps
    M[(M>0.5)&(M<1-eps)] = eps
    M[(M<0)] = 0
    M[(M>1)] = 1
    return M

def mask_eval(fullmask,xstar,Lambda=10**(-6.31)):
    from sigpy.mri.app import TotalVariationRecon
    imgHeg = xstar.shape[0]; imgWid = xstar.shape[1]
    if isinstance(fullmask,torch.Tensor):
        fullmask = fullmask.numpy()
    if isinstance(xstar,torch.Tensor):
        xstar = xstar.numpy()
    yraw = np.fft.fftshift(np.fft.fftn(xstar,norm='ortho'))
    y = np.diag(fullmask)@yraw
    x_ifft = np.fft.ifftn(np.fft.ifftshift(y),norm='ortho')
#     print('ifft error: ', np.sqrt( np.sum((np.abs(x_ifft).flatten()-xstar.flatten())**2) )/np.sqrt( np.sum( (xstar.flatten())**2 )))
#     plt.clf();plt.imshow(np.abs(x_ifft));plt.colorbar();plt.show()
    y = np.reshape(y,(-1,imgHeg,imgWid))
    mps = np.ones(y.shape)
    x = np.fft.fftshift( np.real(TotalVariationRecon(y, mps, Lambda,show_pbar=False,x=x_ifft).run()) )
#     plt.clf();plt.imshow(np.abs(x_ifft));plt.colorbar();plt.show()
    error = np.sqrt( np.sum((x.flatten()-xstar.flatten())**2) )/np.sqrt( np.sum( (xstar.flatten())**2 ))
#     print('TV error:   ', error)
    return error


def mask_backward(highmask,xstar,\
                  beta=1, alpha=5e-1,maxIter=500,unroll_block=3,seed=0,\
                  lr=1e-3,lr_Lambda=1e-8,Lambda=6.1e-4,rho=1e2,mode='ADMM',\
                  perturb=False,break_limit=20,perturb_freq=2,verbose=False):
    '''
    The purpose of this function is to update mask choice (particularly for high frequency) via backward propagation. 
    The input is one image, and the currently employed high frequency mask for the input image.
    The output is the updated mask.
    
    xstar                : ground truth image
    highmask             : binary vector, indicating which high frequencies to sample (low frequencies assumed to be central)
    Lambda               : ADMM parameter lambda (magnitude of TV penalty)
    rho                  : ADMM parameter rho
    mode                 : 'ADMM' for actual usage, 'IFFT' for debugging
    maxIter              : max iterations for backward-prop update step
    unroll_block         : how many blocks of solver to unroll, default 3
    lr                   : learning rate to update mask indicator, default 1e-3
    lr_Lambda            : learning rate to update the ADMM parameter Lambda
    alpha                : l1 penalty magnitude when selecting high frequency masks
    beta                 : slope of sigmoid, deafult 1
    seed                 : random seed, default 0
    perturb              : flag to inject noise into gradients when update the mask
    break_limit          : if loss not change for this many iteration rounds, then break
    perturb_freq         : how frequently to inject noise
    '''
    torch.manual_seed(seed)
    sigmoid = nn.Sigmoid()
    imgHeg,imgWid = xstar.shape[0],xstar.shape[1]
    xstar = torch.tensor(xstar,dtype=torch.double); highmask = torch.tensor(highmask,dtype=torch.double)
    xstar = xstar/torch.max(torch.abs(xstar.flatten()))
    y = torch.roll(F.fftn(xstar,dim=(0,1),norm='ortho'),shifts=(imgHeg//2,imgWid//2),dims=(0,1)).to(DType)
    ## initialising M
    fullmask = torch.tensor( mask_complete(highmask.clone().detach(),imgHeg) ) 
    fullmask_cp = fullmask.clone().detach() 
    print('loss of current mask: ', mask_eval(fullmask_cp,xstar,Lambda=Lambda) * 100)
    highmask[highmask==1] = 1-1e-2
    highmask[highmask==0] = 1e-2
    M_high = -1/beta * torch.log(1/(highmask) -1) # M is before Sigmoid ! 
    M_high.requires_grad = True
#     M = torch.tensor( 1/beta * (-np.log(1/(suppress_prob*np.random.rand(imgHeg))-1)) )
#     if initial_observation > 1:
#         M[0:int(initial_observation/2)]       = 1/beta*(-np.log(1/initial_prob-1))
#         M[imgHeg-int(initial_observation/2):] = 1/beta*(-np.log(1/initial_prob-1))
#     else:
#         M[0:int(initial_observation/2*imgHeg)]      = 1/beta*(-np.log(1/initial_prob-1))
#         M[imgHeg-int(initial_observation/2*imgHeg)] = 1/beta*(-np.log(1/initial_prob-1))
    
#     Lambda    = torch.tensor(Lambda)
#     Lambda.requires_grad = True
#     rho       = torch.tensor(rho)
#     rho.requires_grad    = True
#     optimizer = optim.Adam([{'params':M,'lr':lr},{'params':Lambda,'lr':1e-6}])
#     optimizer = optim.Adam([{'params':M},{'params':Lambda,'lr':1e-6},{'params':rho,'lr':1e2}],lr=lr)
    torch.autograd.set_detect_anomaly(False)
#     optimizer = optim.Adagrad([M_high],lr=lr)
#     optimizer = optim.SGD([{'params':M,'lr':lr},{'params':Lambda,'lr':lr_Lambda}])
#     sparsity_old = 1
    repCount = 0; sparsCount = 0
    flag_perturb = False; perturbList  = list([])
    
    Iter = 0; loss_old = np.inf; x = None
    cr_per_batch = 0
    while Iter < maxIter:
        z = torch.roll(torch.tensordot(torch.diag(fullmask_cp).to(DType),y,dims=([1],[0])) , \
                       shifts=(imgHeg//2,imgWid//2),dims=(0,1))# need to fftshift y and then fftshift y back
        ## Reconstruction process
#         breakpoint()
        if mode == 'ADMM':
            x   = ADMM_TV(z,fullmask,maxIter=unroll_block,Lambda=Lambda,rho=rho,imgInput=False,x_init=None)
        elif mode == 'IFFT': # debug
            x   = torch.real(F.ifftn(z,dim=(0,1),norm='ortho')) # should involve the mask to cater for the lower-level objective
        loss = torch.sum((x-xstar)**2)/torch.sum(xstar**2) + alpha*torch.norm(sigmoid(beta*M_high),p=1)
        if loss.item() < loss_old:
            loss_old = loss.item()
            repCount = 0
        elif loss.item() >= loss_old:
            repCount += 1
            if repCount >= perturb_freq:
                flag_perturb = True
                if repCount >= break_limit:
                    print('No further decrease in loss, breaking iterations')
                    break
# normalized entries of diagonal of M     # + alpha*torch.sum(sigmoid(beta*M)+sigmoid(beta*M)*(1-sigmoid(beta*M)))
        optimizer.zero_grad()
        loss.backward()
        if flag_perturb and perturb:
            perturbList.append(Iter)
            r = torch.randn(M_high.shape)
            r = r/torch.norm(r)
            M_high.grad = M_high.grad + 1/lr *1/5*torch.max(torch.abs(M_high)) * r
            flag_perturb = False
            repCount = 0
        fullmask_old = fullmask_cp.clone()
        optimizer.step()
#         optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr']*1/(Iter+1)
#         optimizer.param_groups[1]['lr'] = min(optimizer.param_groups[1]['lr']*1/(Iter+1) , .5*Lambda.detach().item())

#         fullmask = mask_makebinary(mask_complete(sigmoid(beta*M_high.clone().detach()),imgHeg),threshold=0.5,sigma=False)
        fullmask = mask_complete(sigmoid(beta*M_high),imgHeg)
        fullmask_cp =  mask_makebinary(fullmask.clone().detach(),threshold=0.5,sigma=False)
        
        mask_sparsity = torch.sum(fullmask_cp).item()/fullmask_cp.size()[0]

        changed_rows = np.abs(fullmask_old-fullmask_cp).sum().item()
        cr_per_batch += changed_rows
        if verbose and (changed_rows>0):
            print('iter: {},  changed rows: {}, loss of current mask: {}'.format(Iter+1,changed_rows,mask_eval(fullmask_cp.to(torch.double),xstar,Lambda=Lambda) * 100))
        if verbose and (Iter % 10 == 0):
            print('iter: {}, sparsity: {}, rel. l2 loss: {}'.format(Iter+1,mask_sparsity,(torch.norm(torch.flatten(x)-torch.flatten(xstar),'fro')/torch.norm(torch.flatten(xstar),'fro')).item()) )
            if cr_per_batch > 0 or (Iter == 0):
                kplot(fullmask_cp)           # plot the current mask
            cr_per_batch = 0
        Iter += 1
    highmask_refined = mask_makebinary(sigmoid(beta*M_high.clone().detach()),threshold=0.5,sigma=False)
    print('\nreturn at Iter: ', Iter)
    print('loss of returned mask: ',mask_eval(mask_complete(highmask_refined.to(torch.double),imgHeg),xstar,Lambda=Lambda) * 100)
#     return MASK,Lambda,perturbList
    return highmask_refined