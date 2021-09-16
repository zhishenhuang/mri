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

sys.path.insert(0,'/home/huangz78/mri/unet/')
from unet_model import UNet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def alternating_update_with_unetRecon(mnet,unet,trainfulls,valfulls,train_yfulls=None,val_yfulls=None,\
                                      maxIter_mb=20,alpha=2.8*1e-5,c=0.05, maxRep=5,
                                      lr_mb=1e-2,lr_mn=1e-4,lr_u=5e-5,\
                                      epoch=1,batchsize=5,valbatchsize=10,\
                                      corefreq=24,budget=56,\
                                      verbose=False,hfen=False,dtyp=torch.float,\
#                                       validate_every=30,\
                                      save_cp=False,count_start=(0,0),histpath=None,device='cpu',\
                                      seed=0):
    '''
    alpha: magnitude of l1 penalty for high-frequency mask
    mnet : the input mnet must match corefreq exactly
    '''
    
    if val_yfulls is None:
        val_yfulls = torch.fft.fftshift(F.fftn(valfulls,dim=(1,2),norm='ortho'),dim=(1,2)) # y is ROLLED!
    else:
        val_yfulls = torch.fft.fftshift(val_yfulls,dim=(1,2))
    unet_init = copy.deepcopy(unet)
    DTyp = torch.cfloat if dtyp==torch.float else torch.cdouble
    dir_checkpoint = '/home/huangz78/checkpoints/'
    criterion_mnet = nn.BCEWithLogitsLoss()
    optimizer_m = optim.RMSprop(mnet.parameters(), lr=lr_mn, weight_decay=0, momentum=0)
    binarize = ThresholdBinarizeMask().apply
    acceleration_fold = str(int(trainfulls.shape[1]/(corefreq+budget)))
    
    # shuffling the training data
    shuffle_inds = torch.randperm(trainfulls.shape[0])
    trainfulls = trainfulls[shuffle_inds,:,:]
    
    global_step = 0
    epoch_count = count_start[0]
    batchind    = count_start[1]
    batch_nums  = int(np.ceil(trainfulls.shape[0]/batchsize))
    
    if histpath is None:
        loss_before = list([]); loss_after = list([]); loss_rand = list([]); loss_val = list([])
    else:
        histRec     = np.load(histpath)
        loss_before = list(histRec['loss_before'])
        loss_after  = list(histRec['loss_after'])
        loss_rand   = list(histRec['loss_rand'])
        loss_val    = list(histRec['loss_val'])
        print('training history file successfully loaded from the path: ', histpath)
    
    alpha_grid = 10**torch.arange(-5.7,-3.71,.2)
    alpha_ind_begin = torch.argmin((alpha-alpha_grid).abs())
    alpha_ind = copy.deepcopy(alpha_ind_begin)
    print('input alpha = ',alpha)
    print('alpha in grid begins with ', alpha_grid[alpha_ind])
    target_sparsity = (corefreq + budget)/trainfulls.shape[1]
    
    # training loop
    try:
        lowfreqmask = mask_naiveRand(trainfulls.shape[1],fix=corefreq,other=0,roll=True)[0].to(device)
        while epoch_count<epoch:
            while batchind<batch_nums:
                iterprog  = f'[{global_step+1}][{epoch_count+1}/{epoch}][{min(batchsize*(batchind+1),trainfulls.shape[0])}/{trainfulls.shape[0]}]'
                
                batch = torch.arange(batchsize*batchind, min(batchsize*(batchind+1),trainfulls.shape[0]))
                xstar = trainfulls[batch,:,:].to(device)
                if train_yfulls is None:
                    yfull = torch.fft.fftshift(F.fftn(xstar,dim=(1,2),norm='ortho'),dim=(1,2)) # y is ROLLED!
                else:
                    yfull = torch.fft.fftshift(train_yfulls[batch,:,:],dim=(1,2))
                ########################################  
                ## (1) mask_backward
                ######################################## 
                imgshape = (xstar.shape[1],xstar.shape[2])
                if mnet.in_channels == 1: ######### initialize highmask as output from mnet
                    x_lf     = get_x_f_from_yfull(lowfreqmask,yfull,device=device)
#                     highmask = mnet_wrapper(mnet,x_lf,budget,imgshape,normalize=True,complete=False,detach=True,device=device)
                    highmask =  mnet(x_lf.view(batch.size,1,xstar.shape[1],xstar.shape[2])).detach()
                elif mnet.in_channels == 2:
                    y = torch.zeros((yfull.shape[0],2,yfull.shape[1],yfull.shape[2]),dtype=torch.float,device=device)
                    y[:,0,lowfreqmask==1,:] = torch.real(yfull)[:,lowfreqmask==1,:]
                    y[:,1,lowfreqmask==1,:] = torch.imag(yfull)[:,lowfreqmask==1,:]
#                     highmask = mnet_wrapper(mnet,y,budget,imgshape,normalize=True,complete=False,detach=True,device=device)
                    highmask = mnet(y).detach()
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
                        highmask[ind,:] = mask_naiveRand(xstar.shape[1]-corefreq,fix=0,other=budget,roll=True)[0].to(device)
                    print(iterprog + ' random mask input')
                    randinput = True
                else:
                    randinput = False
                        
                flag = None
                subflag = None
                alpha_ind = copy.deepcopy(alpha_ind_begin)
                lr_mb_tmp = copy.deepcopy(lr_mb)
                maxIter_mb_tmp = copy.deepcopy(maxIter_mb)
                while (alpha_ind<len(alpha_grid)) and (alpha_ind>=0):                   
                    highmask_refined,unet,loss_aft,loss_bef,mask_sparsity_prenorm = mask_backward(highmask,xstar,unet=unet,mnet=mnet,\
                                                                          beta=1.,alpha=alpha_grid[alpha_ind],c=c,\
                                                                          maxIter=maxIter_mb_tmp, break_limit=np.inf,\
                                                                          lr=lr_mb_tmp,lru=lr_u,\
                                                                          mode='UNET',testmode='UNET',\
                                                                          budget=budget,normalize=True,\
                                                                          verbose=verbose,dtyp=torch.float,\
                                                                          hfen=hfen,return_loss_only=False,\
                                                                          device=device,seed=seed)
                    ######### adjust hyperparameters based on feedback:
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
                            maxIter_mb_tmp = copy.deepcopy(maxIter_mb)
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
                            maxIter_mb_tmp = copy.deepcopy(maxIter_mb)
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
                            maxIter_mb_tmp = copy.deepcopy(maxIter_mb)
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
                    highmask = torch.zeros(xstar.shape[0],xstar.shape[1]-corefreq,device=device)
                    for ind in range(len(highmask)):
                        highmask[ind,:] = mask_naiveRand(xstar.shape[1]-corefreq,fix=0,other=budget,roll=True)[0].to(device)
                    print(iterprog + ' random mask input for remedy attempt')
                    highmask_refined,unet,loss_aft,loss_bef,mask_sparsity_prenorm = mask_backward(highmask,xstar,unet=unet,mnet=mnet,\
                                                                          beta=1.,alpha=alpha_grid[alpha_ind_begin],c=c,\
                                                                          maxIter=maxIter_mb, break_limit=np.inf,\
                                                                          lr=lr_mb,lru=lr_u,\
                                                                          mode='UNET',testmode='UNET',\
                                                                          budget=budget,normalize=True,\
                                                                          verbose=verbose,dtyp=torch.float,\
                                                                          hfen=hfen,return_loss_only=False,\
                                                                          device=device,seed=seed)
                    if loss_aft < loss_bef:
                        flag = 'success'
                    else:
                        flag = 'fail'
                        
                loss_after.append(loss_aft)
                loss_before.append(loss_bef)
                print(iterprog + f' quality of old mnet mask : {loss_bef}')
                print(iterprog + f' quality of refined  mask : {loss_aft}')
                
                if flag == 'success':
                    mask_rand = mask_naiveRand(xstar.shape[1],fix=corefreq,other=budget,roll=True)[0].to(device)
                    mask_rand = mask_rand.repeat(xstar.shape[0],1)
                    randqual  = mask_eval(mask_rand,xstar,mode='UNET',UNET=unet_init,dtyp=dtyp,hfen=hfen,device=device) # use fixed warmed-up unet as the reconstructor, Aug 30                
                    print(iterprog + f' quality of random   mask : {randqual}')
                    loss_rand.append(randqual)  ## check mnet performance: does it beat random sampling?
                else:
                    print(iterprog + f' quality of random   mask : irrelevant')
                    loss_rand.append(np.nan)
                ########################################  
                ## (2) update mnet
                ########################################  
                mnet.train()
                if (loss_aft < loss_bef) and (loss_aft < randqual):
                    rep = 0
                    while rep < max(maxRep,2*maxIter_mb_tmp):
                        if mnet.in_channels == 1:
                            x_lf      = get_x_f_from_yfull(lowfreqmask,yfull,device=device)
                            mask_pred = mnet(x_lf.view(batch.size,1,xstar.shape[1],xstar.shape[2]))
                        elif mnet.in_channels == 2:
                            y = torch.zeros((yfull.shape[0],2,yfull.shape[1],yfull.shape[2]),dtype=torch.float,device=device)
                            y[:,0,lowfreqmask==1,:] = torch.real(yfull)[:,lowfreqmask==1,:]
                            y[:,1,lowfreqmask==1,:] = torch.imag(yfull)[:,lowfreqmask==1,:]
                            mask_pred = mnet(y)
                        train_loss = criterion_mnet(mask_pred,highmask_refined)
                        optimizer_m.zero_grad()
                        # optimizer step wrt unet parameters?
                        train_loss.backward()
                        optimizer_m.step()
                        rep += 1
                    mnet.eval()
                    print(iterprog+' is a VALID step!\n')
                else:
                    print(iterprog+' is an invalid step!\n')
                
            ########################################
            ## Validation after each epoch
            # use mnet to generate mask for validation set
                if (batchind == batch_nums//2) or (batchind==(batch_nums-1)) or ((epoch_count==0) and (batchind==0)):
#                 if (global_step%validate_every == 0) or (batchind==(batch_nums-1)):
                    with torch.no_grad():
                        valerr = 0
                        valbatchind   = 0
                        valbatch_nums = int(np.ceil(valfulls.shape[0]/valbatchsize))
                        while valbatchind < valbatch_nums:
                            batch = np.arange(valbatchsize*valbatchind, min(valbatchsize*(valbatchind+1),valfulls.shape[0]))
                            xstar = valfulls[batch,:,:].to(device)
                            yfull = torch.fft.fftshift(val_yfulls[batch,:,:],dim=(1,2)).to(device)
    #                         lowfreqmask = mask_naiveRand(xstar.shape[1],fix=corefreq,other=0,roll=True)[0].to(device)
                            imgshape = (xstar.shape[1],xstar.shape[2])
                            if mnet.in_channels == 1:
                                x_lf     = get_x_f_from_yfull(lowfreqmask,yfull)
                                mask_val = mnet_wrapper(mnet,x_lf,budget,imgshape,normalize=True,detach=True,device=device)
                            elif mnet.in_channels == 2:
                                y = torch.zeros((yfull.shape[0],2,yfull.shape[1],yfull.shape[2]),dtype=torch.float,device=device)
                                y[:,0,lowfreqmask==1,:] = torch.real(yfull)[:,lowfreqmask==1,:]
                                y[:,1,lowfreqmask==1,:] = torch.imag(yfull)[:,lowfreqmask==1,:]
                                mask_val = mnet_wrapper(mnet,y,budget,imgshape,normalize=True,detach=True,device=device)
                            valerr = valerr + mask_eval(mask_val,xstar,mode='UNET',UNET=unet,dtyp=dtyp,hfen=hfen,device=device) # evaluation the equality of mnet masks
                            valbatchind += 1
                            if mnet.in_channels == 1:
                                del xstar,yfull, x_lf
                            elif mnet.in_channels == 2:
                                del xstar, yfull, y
                        loss_val.append(valerr/valbatch_nums)
                        print(f'\n [{global_step+1}][{epoch_count+1}/{epoch}] validation error: {valerr/valbatch_nums} \n')
                        if device != 'cpu':
                            torch.cuda.empty_cache()
            ########################################                     
                if save_cp and ( (global_step%10==0) or (batchind==(batch_nums-1)) ):
                    torch.save({'model_state_dict': mnet.state_dict()}, dir_checkpoint + 'mnet_split_trained_cf_'+ str(corefreq)+'_bg_'+str(budget)+ '_unet_in_chan_' + str(unet.n_channels) +'.pt')
                    torch.save({'model_state_dict': unet.state_dict()}, dir_checkpoint + 'unet_split_trained_cf_'+ str(corefreq)+'_bg_'+str(budget)+ '_unet_in_chan_' + str(unet.n_channels) +'.pt')
                    print(f'\t Checkpoint saved at epoch {epoch_count+1}, iter {global_step + 1}, batchind {batchind+1}!')
                    filepath = '/home/huangz78/checkpoints/alternating_update_error_track_'+acceleration_fold+'fold_'+ 'unet_in_chan_' + str(unet.n_channels) + '.npz'
                    np.savez(filepath,loss_rand=loss_rand,loss_after=loss_after,loss_before=loss_before,loss_val=loss_val)
                global_step += 1
                batchind += 1               
            batchind = 0
            epoch_count += 1
    except KeyboardInterrupt: # need debug
        print('Keyboard Interrupted! Exit~')
        if save_cp:
            torch.save({'model_state_dict': mnet.state_dict()}, dir_checkpoint + 'mnet_split_trained_cf_'+ str(corefreq)+'_bg_'+str(budget) + '_unet_in_chan_' + str(unet.n_channels) +'.pt')
            torch.save({'model_state_dict': unet.state_dict()}, dir_checkpoint + 'unet_split_trained_cf_'+ str(corefreq)+'_bg_'+str(budget) + '_unet_in_chan_' + str(unet.n_channels) +'.pt')
            print(f'\t Checkpoint saved at Python epoch {epoch_count}, Python batchind {batchind}!')
            filepath = '/home/huangz78/checkpoints/alternating_update_error_track_'+acceleration_fold+'fold_'+ 'unet_in_chan_' + str(unet.n_channels) +'.npz'
            np.savez(filepath,loss_rand=loss_rand,loss_after=loss_after,loss_before=loss_before,loss_val=loss_val)
            print('Model is saved after interrupt~')
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
        mnet.apply(mnet_weights_init)
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
        unet.apply(mnet_weights_init)
        print('Unet is randomly initalized!')
    unet.train()
    print('nn\'s are ready')
        
    # load training data
    train_dir = '/mnt/shared_a/data/fastMRI/knee_singlecoil_train.npz'
    train_xfull = torch.tensor(np.load(train_dir)['data']).to(torch.float)
    train_yfull = None
#     train_dir = '/home/huangz78/data/traindata_x.npz'
#     train_xfull = torch.tensor(np.load(train_dir)['xfull'])
#     train_dir = '/home/huangz78/data/traindata_y.npz'
#     train_yfull = torch.tensor(np.load(train_dir)['yfull'])
#     print('train data fft size:', train_yfull.shape)
    for ind in range(train_xfull.shape[0]):
        train_xfull[ind,:,:] = train_xfull[ind,:,:]/torch.max(train_xfull[ind,:,:].abs())
    print('train data size:', train_xfull.shape)
    
    # load validation data 
    val_dir = '/mnt/shared_a/data/fastMRI/knee_singlecoil_val.npz'
    val_xfull = torch.tensor(np.load(val_dir)['data']).to(torch.float)
    val_yfull = None
#     val_dir = '/home/huangz78/data/testdata_x.npz'
#     val_xfull = torch.tensor(np.load(val_dir)['xfull'])
#     val_dir = '/home/huangz78/data/testdata_y.npz'
#     val_yfull = torch.tensor(np.load(val_dir)['yfull'])
#     print('validation data fft size:', val_yfull.shape)
    for ind in range(val_xfull.shape[0]):
        val_xfull[ind,:,:] = val_xfull[ind,:,:]/torch.max(val_xfull[ind,:,:].abs())
    print('validation data size:', val_xfull.shape)
    
    acceleration_fold = str(int(train_xfull.shape[1]/(args.base_freq+args.budget)))
    print(f'corefreq = {args.base_freq}, budget = {args.budget}, this is a {acceleration_fold}-fold training!')
    
    print(args)
    
    alternating_update_with_unetRecon(mnet,unet,train_xfull,val_xfull,train_yfulls=train_yfull,val_yfulls=val_yfull,\
                                  maxIter_mb=args.maxItermb,maxRep=args.mnRep,\
                                  alpha=args.alpha,c=args.c,\
                                  lr_mb=args.lrb,lr_mn=args.lrn,lr_u=args.lru,\
                                  corefreq=args.base_freq,budget=args.budget,\
                                  epoch=args.epochs,batchsize=args.batchsize,\
#                                   validate_every=args.validate_every,\
                                  verbose=False,save_cp=True,count_start=(args.epoch_start,args.batchind_start),\
                                  histpath=args.histpath,hfen=False,device=device,\
                                  seed=args.seed)
    print('\n ~Training concluded!')