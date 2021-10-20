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
from matplotlib import pyplot as plt
import random
import copy

from utils import *
from mnet import MNet
from mask_backward_v4 import mask_backward, mask_eval, ThresholdBinarizeMask

sys.path.insert(0,'/home/leo/mri/unet/')
from unet_model import UNet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class alternating_trainer():
    def __init__(self,
                 mnet:nn.Module,
                 unet:nn.Module,
                 maxIter_mb:int=20,
                 alpha:float=2.8*1e-5,
                 c:float=0.05,
                 maxRep:int=5,
                 lr_mb:float=1e-2,
                 lr_mn:float=1e-4,
                 lr_u:float=5e-5,
                 epochs:int=1,
                 batchsize:int=5,
                 valbatchsize:int=5,
                 corefreq:int=24,
                 budget:int=56,
                 verbose:bool=False,
                 hfen:bool=False,
                 dtyp=torch.float,
                 count_start:tuple=(0,0),
                 dir_hist=None,
                 dir_checkpoint:str='/home/huangz78/checkpoints/',
                 device=torch.device('cpu'),
                 seed:int=0
                ):
        self.mnet = mnet
        self.unet = unet
        self.maxIter_mb = maxIter_mb
        self.alpha = alpha
        self.c = c
        self.maxRep = maxRep
        self.lr_mb = lr_mb
        self.lr_mn = lr_mn
        self.lr_u = lr_u
        self.epochs = epochs
        self.batchsize = batchsize
        self.valbatchsize = valbatchsize
        self.corefreq = corefreq
        self.budget = budget
        self.verbose = verbose
        self.hfen = hfen,
        self.dtyp = dtyp
        self.checkpoint = dir_checkpoint
        self.device = device
        self.seed = seed
        
        if histpath is None:
            self.loss_before = list([]); self.loss_after = list([]); self.loss_rand = list([]); self.loss_val = list([])
        else:
            histRec = np.load(dir_hist)
            self.loss_before = list(histRec['loss_before'])
            self.loss_after  = list(histRec['loss_after'])
            self.loss_rand   = list(histRec['loss_rand'])
            self.loss_val    = list(histRec['loss_val'])
            print('training history file successfully loaded from the path: ', dir_hist)
    
    def validate(self,valfulls,val_yfulls=None,epoch=0):
        with torch.no_grad():
            valerr = 0
            valbatchind   = 0
            valbatch_nums = int(np.ceil(valfulls.shape[0]/self.valbatchsize))
            while valbatchind < valbatch_nums:
                batch = np.arange(self.valbatchsize*valbatchind, min(self.valbatchsize*(valbatchind+1),valfulls.shape[0]))
                xstar = valfulls[batch,:,:].to(self.device)
                yfull = F.fftshift(val_yfulls[batch,:,:],dim=(1,2)).to(self.device)
                lowfreqmask = mask_naiveRand(xstar.shape[1],fix=self.corefreq,other=0,roll=True)[0].to(self.device)
                imgshape = (xstar.shape[1],xstar.shape[2])
                if self.mnet.in_channels == 1:
                    x_lf     = get_x_f_from_yfull(lowfreqmask,yfull)
                    mask_val = mnet_wrapper(self.mnet,x_lf,self.budget,imgshape,normalize=True,detach=True,device=self.device)
                elif mnet.in_channels == 2:
                    y = torch.zeros((yfull.shape[0],2,yfull.shape[1],yfull.shape[2]),dtype=torch.float,device=self.device)
                    y[:,0,lowfreqmask==1,:] = torch.real(yfull)[:,lowfreqmask==1,:]
                    y[:,1,lowfreqmask==1,:] = torch.imag(yfull)[:,lowfreqmask==1,:]
                    mask_val = mnet_wrapper(mnet,y,budget,imgshape,normalize=True,detach=True,device=device)
                valerr = valerr + mask_eval(mask_val,xstar,mode='UNET',UNET=self.unet,dtyp=self.dtyp,hfen=self.hfen,device=self.device) # evaluation the equality of mnet masks
                valbatchind += 1
                if self.mnet.in_channels == 1:
                    del xstar,yfull, x_lf
                elif self.mnet.in_channels == 2:
                    del xstar, yfull, y
            self.loss_val.append(valerr/valbatch_nums)
            print(f'\n [{self.global_step+1}][{epoch+1}/{self.epochs}] validation error: {valerr/valbatch_nums} \n')
            
            torch.cuda.empty_cache()
    
    
    def save(self,epoch=0,batchind=None):
        torch.save({'model_state_dict': self.mnet.state_dict()}, self.dir_checkpoint + 'mnet_split_trained_cf_'+ str(self.corefreq) + '_bg_'+str(self.budget) + '_unet_in_chan_' + str(self.unet.n_channels) + '.pt')
        torch.save({'model_state_dict': self.unet.state_dict()}, self.dir_checkpoint + 'unet_split_trained_cf_'+ str(self.corefreq) + '_bg_'+str(self.budget) + '_unet_in_chan_' + str(self.unet.n_channels) + '.pt')
        if batchind is not None:
            print(f'\t Checkpoint saved at epoch {epoch_count+1}, iter {self.global_step + 1}, batchind {batchind+1}!')
        else:
            print(f'\t Checkpoint saved at epoch {epoch_count+1}, iter {self.global_step + 1}!')
        filepath = self.dir_checkpoint + 'alternating_update_error_track_'+self.acceleration_fold+'fold_'+ 'unet_in_chan_' + str(self.unet.n_channels) + '.npz'
        np.savez(filepath,loss_rand=self.loss_rand,loss_after=self.loss_after,loss_before=self.loss_before,loss_val=self.loss_val)
    
    
    def run(self,trainfulls,valfulls,train_yfulls=None,val_yfulls=None,unet_init=None,save_cp=False):   
        '''
        alpha: magnitude of l1 penalty for high-frequency mask
        mnet : the input mnet must match corefreq exactly
        '''

        if val_yfulls is None:
            val_yfulls = F.fftshift(F.fftn(valfulls,dim=(1,2),norm='ortho'),dim=(1,2)) # y is ROLLED!
        else:
            val_yfulls = F.fftshift(val_yfulls,dim=(1,2))
            
        if unet_init is None:
            unet_init = copy.deepcopy(self.unet)
            
        criterion_mnet = nn.BCEWithLogitsLoss()
        optimizer_m = optim.RMSprop(mnet.parameters(), lr=lr_mn, weight_decay=0) # , momentum=0)
        binarize = ThresholdBinarizeMask().apply
        
        self.acceleration_fold = str(int(trainfulls.shape[1]/(corefreq+budget)))
        
        # shuffling the training data
        shuffle_inds = torch.randperm(trainfulls.shape[0])
        trainfulls   = trainfulls[shuffle_inds,:,:]

        self.global_step = 0
        epoch_count = self.count_start[0]
        batchind    = self.count_start[1]
        batch_nums  = int(np.ceil(trainfulls.shape[0]/self.batchsize))
        
        alpha_grid      = 10**torch.arange(-5.7,-3.71,.2)
        alpha_ind_begin = torch.argmin((self.alpha-alpha_grid).abs())
        alpha_ind       = copy.deepcopy(alpha_ind_begin)
        print('input alpha = ', self.alpha)
        print('alpha in grid begins with ', alpha_grid[alpha_ind])
        target_sparsity = (self.corefreq + self.budget)/trainfulls.shape[1]
        
        # training loop
        try:
            lowfreqmask = mask_naiveRand(trainfulls.shape[1],fix=self.corefreq,other=0,roll=True)[0].to(self.device)
            while epoch_count<self.epochs:
                while batchind<batch_nums:
                    iterprog  = f'[{self.global_step+1}][{epoch_count+1}/{self.epochs}][{min(self.batchsize*(batchind+1),trainfulls.shape[0])}/{trainfulls.shape[0]}]'

                    batch = torch.arange(self.batchsize*batchind, min(self.batchsize*(batchind+1),trainfulls.shape[0]))
                    xstar = trainfulls[batch,:,:].to(self.device)
                    if train_yfulls is None:
                        yfull = F.fftshift(F.fftn(xstar,dim=(1,2),norm='ortho'),dim=(1,2)) # y is ROLLED!
                    else:
                        yfull = F.fftshift(train_yfulls[batch,:,:],dim=(1,2))
                    ########################################  
                    ## (1) mask_backward
                    ######################################## 
                    imgshape = (xstar.shape[1],xstar.shape[2])
                    if self.mnet.in_channels == 1: ######### initialize highmask as output from mnet
                        x_lf     = get_x_f_from_yfull(lowfreqmask,yfull,device=self.device)
    #                     highmask = mnet_wrapper(mnet,x_lf,budget,imgshape,normalize=True,complete=False,detach=True,device=device)
                        highmask =  self.mnet(x_lf.view(batch.size,1,xstar.shape[1],xstar.shape[2])).detach()
                    elif mnet.in_channels == 2:
                        y = torch.zeros((yfull.shape[0],2,yfull.shape[1],yfull.shape[2]),dtype=self.dtyp,device=self.device)
                        y[:,0,lowfreqmask==1,:] = torch.real(yfull)[:,lowfreqmask==1,:]
                        y[:,1,lowfreqmask==1,:] = torch.imag(yfull)[:,lowfreqmask==1,:]
    #                     highmask = mnet_wrapper(mnet,y,budget,imgshape,normalize=True,complete=False,detach=True,device=device)
                        highmask = self.mnet(y).detach()
                    ######### check if highmask is repetitive
                    highmask_b = binarize( torch.sigmoid(highmask) )
                    diffcount = 0
                    for i in range(len(highmask)-1):
                        for j in range(i+1,len(highmask)):
                            if (highmask_b[i,:] - highmask_b[j,:]).abs().sum() != 0:
                                diffcount += 1
                    if diffcount < (len(highmask)//2+len(highmask)%2) * ((len(highmask)//2+len(highmask)%2)-1) /2.: ## initialize highmask as random mask
                        highmask = torch.zeros_like(highmask)
                        for ind in range(len(highmask)):
                            highmask[ind,:] = mask_naiveRand(xstar.shape[1]-self.corefreq,fix=0,other=self.budget,roll=True)[0].to(self.device)
                        print(iterprog + ' random mask input')
                        randinput = True
                    else:
                        randinput = False

                    flag = None
                    subflag = None
                    alpha_ind = copy.deepcopy(alpha_ind_begin)
                    lr_mb_tmp = copy.deepcopy(self.lr_mb)
                    maxIter_mb_tmp = copy.deepcopy(self.maxIter_mb)
                    while (alpha_ind<len(alpha_grid)) and (alpha_ind>=0):                   
                        highmask_refined,self.unet,loss_aft,loss_bef,mask_sparsity_prenorm = \
                                                mask_backward(highmask,xstar,unet=self.unet,mnet=self.mnet,\
                                                              beta=1.,alpha=alpha_grid[alpha_ind],c=self.c,\
                                                              maxIter=maxIter_mb_tmp, break_limit=np.inf,\
                                                              lr=lr_mb_tmp,lru=self.lr_u,\
                                                              mode='UNET',testmode='UNET',\
                                                              budget=self.budget,normalize=True,\
                                                              verbose=self.verbose,dtyp=self.dtyp,\
                                                              hfen=self.hfen,return_loss_only=False,\
                                                              device=self.device,seed=self.seed)
                        ###############################################################
                        ########## adjust hyperparameters based on feedback: ##########
                        ###############################################################
    #                     print(iterprog + 'mask_sparsity_prenorm = ', mask_sparsity_prenorm)
                        if loss_aft == np.inf:     # converged to degenerate mask, alpha too large, decrease alpha
                            if (flag == 'no_change') or (flag == 'no_improve'):
                                print(f'paradox: first {flag} then degenerate')
                                flag = 'fail'
                                break
                            if mask_sparsity_prenorm > target_sparsity:
                                if subflag == 'down':
                                    print(f'paradox: first {flag}/{subflag} then degenerate/up')
                                    flag = 'fail'
                                    break
                                print(iterprog + ' mask degenerate, alpha_Increase')
                                alpha_ind += 1
                                maxIter_mb_tmp = copy.deepcopy(self.maxIter_mb)
                                subflag = 'up'
                            else:
                                if subflag == 'up':
                                    print(f'paradox: first {flag}/{subflag} then degenerate/down')
                                    flag = 'fail'
                                    break
                                print(iterprog + ' mask degenerate, alpha_Decrease')
                                alpha_ind -= 1
                                lr_mb_tmp *= 2
                                maxIter_mb_tmp += 1
                                subflag = 'down'
                            flag = 'degenerate'
                        elif loss_aft == loss_bef: # no change of mask happened, alpha too large, decrease alpha
                            if (flag == 'degenerate') or (flag == 'no_improve'):
                                print(f'paradox: first {flag} then no change')
                                flag = 'fail'
                                break
                            if mask_sparsity_prenorm > target_sparsity:
                                if subflag == 'down':
                                    print(f'paradox: first {flag}/{subflag} then no_change/up')
                                    flag = 'fail'
                                    break
                                print(iterprog + ' mask no change,  alpha_Increase')
                                alpha_ind += 1
                                maxIter_mb_tmp = copy.deepcopy(self.maxIter_mb)
                                subflag = 'up'
                            else:
                                if subflag == 'up':
                                    print(f'paradox: first {flag}/{subflag} then no_change/down')
                                    flag = 'fail'
                                    break
                                print(iterprog + ' mask no change,  alpha_Decrease')
                                alpha_ind -= 1
                                lr_mb_tmp *= 2
                                maxIter_mb_tmp += 1
                                subflag = 'down'
                            flag = 'no_change'
                        elif loss_aft > loss_bef:# mb did not get better masks, although refined masks are not trivial
                            if mask_sparsity_prenorm > target_sparsity:
                                if subflag == 'down':
                                    print(f'paradox: first {flag}/{subflag} then no_improve/up')
                                    flag = 'fail'
                                    break
                                print(iterprog + ' mb did not improve, alpha_Increase')
                                alpha_ind = max(alpha_ind_begin,alpha_ind) + 1
                                maxIter_mb_tmp = copy.deepcopy(self.maxIter_mb)
                                subflag = 'up'
                            else:
                                if subflag == 'up':
                                    print(f'paradox: first {flag}/{subflag} then no_improve/down')
                                    flag = 'fail'
                                    break
                                alpha_ind -= 1                      
                                lr_mb_tmp *= 2
                                maxIter_mb_tmp += 1
                                subflag = 'down'
                            flag = 'no_improve'
                        else:                      # some change happens to the mask and the mask is not degenerate, proceed
                            print(iterprog + ' mb finished')
                            if loss_aft < loss_bef:
                                flag = 'success'
                            else:
                                flag = 'fail'
                            break      
                        ###################################################################################################
                    if (flag != 'success') and (not randinput):
                        highmask = torch.zeros(xstar.shape[0],xstar.shape[1]-self.corefreq,device=self.device)
                        for ind in range(len(highmask)):
                            highmask[ind,:] = mask_naiveRand(xstar.shape[1]-self.corefreq,fix=0,other=self.budget,roll=True)[0].to(self.device)
                        print(iterprog + ' random mask input for remedy attempt')
                        highmask_refined,_,loss_aft,loss_bef,_ = \
                                            mask_backward(highmask,xstar,unet=copy.deepcopy(self.unet),mnet=self.mnet,\
                                                          beta=1.,alpha=alpha_grid[alpha_ind_begin],c=self.c,\
                                                          maxIter=self.maxIter_mb, break_limit=np.inf,\
                                                          lr=self.lr_mb,lru=self.lr_u,\
                                                          mode='UNET',testmode='UNET',\
                                                          budget=self.budget,normalize=True,\
                                                          verbose=self.verbose,dtyp=self.dtyp,\
                                                          hfen=self.hfen,return_loss_only=False,\
                                                          device=self.device,seed=self.seed)
                        if loss_aft < loss_bef:
                            flag = 'success'
                        else:
                            flag = 'fail'

                    self.loss_after.append(loss_aft)
                    self.loss_before.append(loss_bef)
                    print(iterprog + f' quality of old mnet mask : {loss_bef}')
                    print(iterprog + f' quality of refined  mask : {loss_aft}')

                    if flag == 'success':
                        mask_rand = mask_naiveRand(xstar.shape[1],fix=self.corefreq,other=self.budget,roll=True)[0].to(self.device)
                        mask_rand = mask_rand.repeat(xstar.shape[0],1)
                        randqual  = mask_eval(mask_rand,xstar,mode='UNET',UNET=unet_init,dtyp=self.dtyp,hfen=self.hfen,device=self.device) # use fixed warmed-up unet as the reconstructor, Aug 30                
                        print(iterprog + f' quality of random   mask : {randqual}')
                        loss_rand.append(randqual)  ## check mnet performance: does it beat random sampling?
                    else:
                        print(iterprog + f' quality of random   mask : irrelevant')
                        loss_rand.append(np.nan)
                    ########################################  
                    ## (2) update mnet
                    ########################################  
                    self.mnet.train()
                    if (loss_aft < loss_bef) and (loss_aft < randqual):
                        rep = 0
                        while rep < max(self.maxRep,2*maxIter_mb_tmp):
                            if mnet.in_channels == 1:
                                x_lf      = get_x_f_from_yfull(lowfreqmask,yfull,device=self.device)
                                mask_pred = self.mnet(x_lf.view(batch.size,1,xstar.shape[1],xstar.shape[2]))
                            elif mnet.in_channels == 2:
                                y = torch.zeros((yfull.shape[0],2,yfull.shape[1],yfull.shape[2]),dtype=self.dtyp,device=self.device)
                                y[:,0,lowfreqmask==1,:] = torch.real(yfull)[:,lowfreqmask==1,:]
                                y[:,1,lowfreqmask==1,:] = torch.imag(yfull)[:,lowfreqmask==1,:]
                                mask_pred = self.mnet(y)
                            train_loss = criterion_mnet(mask_pred,highmask_refined)
                            optimizer_m.zero_grad()
                            # optimizer step wrt unet parameters?
                            train_loss.backward()
                            optimizer_m.step()
                            rep += 1
                        self.mnet.eval()
                        print(iterprog+' is a VALID step!\n')
                    else:
                        print(iterprog+' is an invalid step!\n')

                ########################################
                ## Validation after each epoch
                # use mnet to generate mask for validation set
                    if (batchind == batch_nums//2) or (batchind==(batch_nums-1)) or ((epoch_count==0) and (batchind==0)):
    #                 if (global_step%validate_every == 0) or (batchind==(batch_nums-1)):
                        self.validate(valfulls,val_yfulls=val_yfulls,epoch=epoch_count)
                ########################################                     
                    if save_cp and ( (self.global_step%10==0) or (batchind==(batch_nums-1)) ):
                        self.save(epoch=epoch_count,batchind=batchind)
                    self.global_step += 1
                    batchind += 1               
                batchind = 0           
                if save_cp:
                    self.save(epoch=epoch_count)
                epoch_count += 1
        except KeyboardInterrupt: # need debug
            print('Keyboard Interrupted! Exit~')
            if save_cp:
                self.save(epoch=epoch_count,batchind=batchind)
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    #     return mnet, unet

    
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=2,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lrb', '--learning-rate-backward', metavar='LRB', type=float, nargs='?', default=5e-3,
                        help='Learning rate for maskbackward', dest='lrb')
    parser.add_argument('-lrn', '--learning-rate-mnet', metavar='LRN', type=float, nargs='?', default=5e-4,
                        help='Learning rate for mnet', dest='lrn')
    parser.add_argument('-lru', '--learning-rate-unet', metavar='LRU', type=float, nargs='?', default=5e-4,
                        help='Learning rate for unet', dest='lru')

    parser.add_argument('-es','--epoch-start',metavar='ES',type=int,nargs='?',default=0,
                        help='starting epoch count',dest='epoch_start')
    parser.add_argument('-bis','--batchind-start',metavar='BIS',type=int,nargs='?',default=0,
                        help='starting batchind',dest='batchind_start')
    
    parser.add_argument('-mp', '--mnet-path', type=str, default=None,
                        help='path file for a mnet', dest='mnetpath')
    parser.add_argument('-up', '--unet-path', type=str, default='/home/huangz78/checkpoints/unet_1_True_8frand.pt',
                        help='path file for a unet', dest='unetpath')
    parser.add_argument('-hp', '--history-path', type=str, default=None,
                        help='path file for npz file recording training history', dest='histpath')
    parser.add_argument('-uip', '--unet-init-path', type=str, default='/home/huangz78/checkpoints/unet_1_True_8frand.pt',
                        help='path file for an initial unet', dest='unet_init_path')
    
    parser.add_argument('-mbit', '--mb-iter-max', type=int, default=20,
                        help='maximum interation for maskbackward function', dest='maxItermb')
    parser.add_argument('-mnrep', '--mn-iter-rep', type=int, default=60,
                        help='inside one batch, updating mnet this many times', dest='mnRep')   
#     parser.add_argument('-valfreq', '--validate-every', type=int, default=100,
#                         help='do validation every # steps', dest='validate_every')
    
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=8,
                        help='number of observed low frequencies', dest='base_freq')
    parser.add_argument('-bg','--budget',metavar='BG',type=int,nargs='?',default=32,
                        help='number of high frequencies to sample', dest='budget')
    
    parser.add_argument('-alpha', '--alpha-param', type=float, default=2e-5,
                        help='magnitude for l1 penalty in loss function', dest='alpha')    
    parser.add_argument('-c', '--c-param', type=float, default=5e-4,
                        help='magnitude for consistency penalty in loss function', dest='c')
    
    parser.add_argument('-ngpu', '--num-gpu', type=int, default=0,
                        help='number of GPUs', dest='ngpu')
    parser.add_argument('-uc', '--unet-channels', type=int, default=1,
                        help='number of Unet input channcels', dest='n_channels')
    parser.add_argument('-skip', '--unet-skip', type=str, default='True',
                        help='switch for ResNet structure in Unet', dest='skip')
    
    parser.add_argument('-sd', '--seed', type=int, default=0,
                        help='random seed', dest='seed')
    # alpha 2e-5, c 1e-3   ---> unet n_channel: 1, 8 fold
    # alpha 4e-6, c 1e-3   ---> unet n_channel: 1, 4 fold
    
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
      
    if args.skip == 'False':
        args.skip = False
    elif args.skip == 'True':
        args.skip = True

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    ### load a mnet
    mnet = MNet(beta=1,in_channels=2,out_size=320-args.base_freq, imgsize=(320,320),poolk=3).to(device)
    if args.mnetpath is not None:
        mnetpath = args.mnetpath
        checkpoint = torch.load(mnetpath)
        mnet.load_state_dict(checkpoint['model_state_dict'])
        print('MNet loaded successfully from: ' + mnetpath)
    else:
        mnet.apply(nn_weights_init)
        print('Mnet is randomly initialized!')
    mnet.eval()
    
    ### load a unet for maskbackward
    unet = UNet(n_channels=args.n_channels,n_classes=1,bilinear=(not args.skip),skip=args.skip).to(device)
    unetpath = args.unetpath
    if unetpath is not None:
        checkpoint = torch.load(unetpath)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print('Unet loaded successfully from: ' + unetpath )
    else:
        unet.apply(nn_weights_init)
        print('Unet is randomly initalized!')
    unet.train()
    print('nn\'s are ready')
    
    unetinitpath = args.unet_init_path
    if unetinitpath is not None:
        unet_init = UNet(n_channels=args.n_channels,n_classes=1,bilinear=(not args.skip),skip=args.skip).to(device)
        checkpoint = torch.load(unetinitpath)
        unet_init.load_state_dict(checkpoint['model_state_dict'])
        print('Unet_init loaded successfully from: ' + unetinitpath )
    else:
        unet_init = None
        
    # load training data
    train_dir = '/mnt/shared_a/data/fastMRI/knee_singlecoil_train.npz'
    train_xfull = torch.tensor(np.load(train_dir)['data']).to(torch.float)
    train_yfull = None

    for ind in range(train_xfull.shape[0]):
        train_xfull[ind,:,:] = train_xfull[ind,:,:]/torch.max(train_xfull[ind,:,:].abs())
    print('train data size:', train_xfull.shape)
    
    # load validation data 
    val_dir = '/mnt/shared_a/data/fastMRI/knee_singlecoil_val.npz'
    val_xfull = torch.tensor(np.load(val_dir)['data']).to(torch.float)
    val_yfull = None

    for ind in range(val_xfull.shape[0]):
        val_xfull[ind,:,:] = val_xfull[ind,:,:]/torch.max(val_xfull[ind,:,:].abs())
    print('validation data size:', val_xfull.shape)
    
    acceleration_fold = str(int(train_xfull.shape[1]/(args.base_freq+args.budget)))
    print(f'corefreq = {args.base_freq}, budget = {args.budget}, this is a {acceleration_fold}-fold training!')
    
    print(args)
    
    alternating_trainer(mnet=mnet, unet=unet,
                         maxIter_mb=args.maxItermb,
                         alpha=args.alpha,
                         c=args.c,
                         maxRep=args.mnRep,
                         lr_mb=args.lrb,
                         lr_mn=args.lrn,
                         lr_u=args.lru,
                         epochs=args.epochs,
                         batchsize=args.batchsize,
                         valbatchsize=5,
                         corefreq=args.base_freq,
                         budget=args.budget,
                         verbose=False,
                         hfen=False,
                         dtyp=torch.float,
                         count_start=(args.epoch_start,args.batchind_start),
                         dir_hist=args.histpath,
                         dir_checkpoint='/home/huangz78/checkpoints/leo/mri/',
                         device=device,
                         seed:int=args.seed)
    
    alternating_trainer.run(train_xfull,val_xfull,train_yfulls=train_yfull,val_yfulls=val_yfull,
                            unet_init=unet_init,save_cp=False)
    

    print('\n ~Training concluded!')