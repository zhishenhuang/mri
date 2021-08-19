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
# from mask_backward_v1 import mask_backward, mask_eval
from mask_backward_v3 import mask_backward, mask_eval
sys.path.insert(0,'/home/huangz78/mri/unet/')
from unet_model import UNet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def alternating_update_with_unetRecon(mnet,unet,trainfulls,valfulls,train_yfulls=None,val_yfulls=None,\
                                      maxIter_mb=20,alpha=2.8*1e-5,c=0.05, lr_mb=1e-2,\
                                      maxRep=5,lr_mn=1e-4,\
                                      epoch=1,batchsize=5,valbatchsize=5,\
                                      corefreq=24,budget=56,\
                                      verbose=False,hfen=False,dtyp=torch.float,validate_every=30,\
                                      save_cp=False,count_start=(0,0),histpath=None):
    '''
    alpha: magnitude of l1 penalty for high-frequency mask
    mnet : the input mnet must match corefreq exactly
    '''
    
    if val_yfulls is None:
        val_yfull = torch.fft.fftshift(F.fftn(valfulls,dim=(1,2),norm='ortho')) # y is ROLLED!
    else:
        val_yfull = torch.fft.fftshift(val_yfulls,dim=(1,2))

    DTyp = torch.cfloat if dtyp==torch.float else torch.cdouble
    dir_checkpoint = '/home/huangz78/checkpoints/'
    criterion_mnet = nn.BCEWithLogitsLoss()
    optimizer_m = optim.RMSprop(mnet.parameters(), lr=lr_mn, weight_decay=0, momentum=0)
    
    acceleration_fold = str(int(trainfulls.shape[1]/(corefreq+budget)))
    
    # training loop
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
        
    try:
        while epoch_count<epoch:
            while batchind<batch_nums:
                batch = np.arange(batchsize*batchind, min(batchsize*(batchind+1),trainfulls.shape[0]))
                xstar = trainfulls[batch,:,:]
                if train_yfulls is None:
                    yfull = torch.fft.fftshift(F.fftn(xstar,dim=(1,2),norm='ortho')) # y is ROLLED!
                else:
                    yfull = torch.fft.fftshift(train_yfulls[batch,:,:],dim=(1,2))
                lowfreqmask,_,_ = mask_naiveRand(xstar.shape[1],fix=corefreq,other=0,roll=True)

                ########################################  
                ## (1) mask_backward
                ######################################## 
                if (epoch_count == 0): # initialize highmask as random mask
                    highmask = mask_filter(mask_naiveRand(xstar.shape[1],fix=corefreq,other=1.5*budget,roll=True)[0],base=corefreq,roll=True)
                    highmask = highmask.repeat(xstar.shape[0],1)
                else: # initialize highmask as output from mnet
                    if mnet.in_channels == 1:
                        x_lf     = get_x_f_from_yfull(lowfreqmask,yfull)
                        highmask = torch.sigmoid( mnet(x_lf.view(batch.size,1,xstar.shape[1],xstar.shape[2])) )
                    elif mnet.in_channels == 2:
                        y = torch.zeros((yfull.shape[0],2,yfull.shape[1],yfull.shape[2]),dtype=torch.float)
                        y[:,0,lowfreqmask==1,:] = torch.real(yfull)[:,lowfreqmask==1,:]
                        y[:,1,lowfreqmask==1,:] = torch.imag(yfull)[:,lowfreqmask==1,:]
                        highmask = torch.sigmoid( mnet(y) )
                highmask_refined,unet,loss_aft,loss_bef = mask_backward(highmask,xstar,unet=unet,mnet=mnet,\
                                  beta=1.,alpha=alpha,c=c,\
                                  maxIter=maxIter_mb,seed=0,break_limit=np.inf,\
                                  lr=lr_mb,mode='UNET',testmode='UNET',\
                                  budget=budget,normalize=True,\
                                  verbose=verbose,dtyp=torch.float,\
                                  hfen=False,return_loss_only=False)   
                iterprog  = f'[{epoch_count+1}/{epoch}][{min(batchsize*(batchind+1),trainfulls.shape[0])}/{trainfulls.shape[0]}]'
                mask_rand = mask_naiveRand(xstar.shape[1],fix=corefreq,other=budget,roll=True)[0]
                mask_rand = mask_rand.repeat(xstar.shape[0],1)
                randqual  = mask_eval(mask_rand,xstar,mode='UNET',UNET=UNET,dtyp=dtyp,hfen=hfen)
                print(iterprog + f', quality of old mnet mask : {loss_bef}')
                print(iterprog + f', quality of refined  mask : {loss_aft}')
                print(iterprog + f', quality of random   mask : {randqual}')
                loss_rand.append(randqual); loss_after.append(loss_aft); loss_before.append(loss_bef) ## check mnet performance: does it beat random sampling?
                ########################################  
                ## (2) update mnet
                ########################################  
                mnet.train()
    #             unet.eval()
                if (loss_aft < loss_bef) and (loss_aft < randqual):
                    rep = 0
                    while rep < maxRep:
                        if mnet.in_channels == 1:
                            x_lf      = get_x_f_from_yfull(lowfreqmask,yfull)
                            mask_pred = mnet(x_lf.view(batch.size,1,xstar.shape[1],xstar.shape[2]))
                        elif mnet.in_channels == 2:
                            y = torch.zeros((yfull.shape[0],2,yfull.shape[1],yfull.shape[2]),dtype=torch.float)
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
                    print(iterprog+' is a VALID step!')
                else:
                    print(iterprog+' is an invalid step!')
                
            ########################################
            ## Validation after each epoch
            # use mnet to generate mask for validation set
                if global_step%validate_every == 0:
                    valerr = 0
                    valbatchind   = 0
                    valbatch_nums = int(np.ceil(valfulls.shape[0]/valbatchsize))
                    while valbatchind < valbatch_nums:
                        batch = np.arange(valbatchsize*valbatchind, min(valbatchsize*(valbatchind+1),valfulls.shape[0]))
                        xstar = valfulls[batch,:,:]
                        if val_yfulls is None:
                            yfull = torch.fft.fftshift(F.fftn(xstar,dim=(1,2),norm='ortho')) # y is ROLLED!
                        else:
                            yfull = torch.fft.fftshift(val_yfulls[batch,:,:],dim=(1,2))
                        lowfreqmask,_,_ = mask_naiveRand(xstar.shape[1],fix=corefreq,other=0,roll=True)
                        imgshape = (xstar.shape[1],xstar.shape[2])
                        if mnet.in_channels == 1:
                            x_lf     = get_x_f_from_yfull(lowfreqmask,yfull)
                            mask_val = mnet_wrapper(mnet,x_lf,budget,imgshape,normalize=True,detach=True)
                        elif mnet.in_channels == 2:
                            y = torch.zeros((yfull.shape[0],2,yfull.shape[1],yfull.shape[2]),dtype=torch.float)
                            y[:,0,lowfreqmask==1,:] = torch.real(yfull)[:,lowfreqmask==1,:]
                            y[:,1,lowfreqmask==1,:] = torch.imag(yfull)[:,lowfreqmask==1,:]
                            mask_val = mnet_wrapper(mnet,y,budget,imgshape,normalize=True,detach=True)
                        valerr += mask_eval(mask_val,xstar,mode='sigpy',Lambda=1e-4,dtyp=dtyp,hfen=hfen) # evaluation the equality of mnet masks
                        valbatchind += 1
                    loss_val.append(valerr/valbatch_nums)
                    print(f'\n [{global_step+1}][{epoch_count+1}/{epoch}] validation error: {valerr/valbatch_nums} \n')
            ########################################                     
                if save_cp and ( (global_step%10==0) or (batchind==(batch_nums-1)) ):
                    torch.save({'model_state_dict': mnet.state_dict()}, dir_checkpoint + 'mnet_split_trained_cf'+ str(corefreq)+'_bg_'+str(budget) +'.pt')
                    torch.save({'model_state_dict': unet.state_dict()}, dir_checkpoint + 'unet_split_trained_cf'+ str(corefreq)+'_bg_'+str(budget)+'.pt')
                    print(f'\t Checkpoint saved at epoch {epoch_count+1}, iter {global_step + 1}, batchind {batchind+1}!')
                    filepath = '/home/huangz78/checkpoints/alternating_update_error_track_'+acceleration_fold+'fold.npz'
                    np.savez(filepath,loss_rand=loss_rand,loss_after=loss_after,loss_before=loss_before,loss_val=loss_val)
                global_step += 1
                batchind += 1                       
            batchind = 0
            epoch_count += 1
    except KeyboardInterrupt: # need debug
        print('Keyboard Interrupted! Exit~')
        if save_cp:
            torch.save({'model_state_dict': mnet.state_dict()}, dir_checkpoint + 'mnet_split_trained_cf'+ str(corefreq)+'_bg_'+str(budget) +'.pt')
            torch.save({'model_state_dict': unet.state_dict()}, dir_checkpoint + 'unet_split_trained_cf'+ str(corefreq)+'_bg_'+str(budget)+'.pt')
            print(f'\t Checkpoint saved at Python epoch {epoch_count}, Python batchind {batchind}!')
            filepath = '/home/huangz78/checkpoints/alternating_update_error_track_'+acceleration_fold+'fold.npz'
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
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lrb', '--learning-rate-backward', metavar='LRB', type=float, nargs='?', default=1e-2,
                        help='Learning rate for maskbackward', dest='lrb')
    parser.add_argument('-lrn', '--learning-rate-mnet', metavar='LRN', type=float, nargs='?', default=1e-3,
                        help='Learning rate for mnet', dest='lrn')

    parser.add_argument('-es','--epoch-start',metavar='ES',type=int,nargs='?',default=0,
                        help='starting epoch count',dest='epoch_start')
    parser.add_argument('-bis','--batchind-start',metavar='BIS',type=int,nargs='?',default=0,
                        help='starting batchind',dest='batchind_start')
    
    parser.add_argument('-mp', '--mnet-path', type=str, default=None,
                        help='path file for a mnet', dest='mnetpath')
    parser.add_argument('-up', '--unet-path', type=str, default='/home/huangz78/checkpoints/unet_1_True.pth',
                        help='path file for a unet', dest='unetpath')
    parser.add_argument('-hp', '--history-path', type=str, default=None,
                        help='path file for npz file recording training history', dest='histpath')
    
    parser.add_argument('-mbit', '--mb-iter-max', type=int, default=30,
                        help='maximum interation for maskbackward function', dest='maxItermb')
    parser.add_argument('-mnrep', '--mn-iter-rep', type=int, default=30,
                        help='inside one batch, updating mnet this many times', dest='mnRep')   
    parser.add_argument('-valfreq', '--validate-every', type=int, default=30,
                        help='do validation every # steps', dest='validate_every')
    
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=8,
                        help='number of observed low frequencies', dest='base_freq')
    parser.add_argument('-bg','--budget',metavar='BG',type=int,nargs='?',default=32,
                        help='number of high frequencies to sample', dest='budget')
    
    parser.add_argument('-alpha', '--alpha-param', type=float, default=1e-4,
                        help='magnitude for l1 penalty in loss function', dest='alpha')    
    parser.add_argument('-c', '--c-param', type=float, default=1e-2,
                        help='magnitude for consistency penalty in loss function', dest='c')
    
    # alpha 1e-4, c 1e-2   ---> 8-fold
    # alpha 1e-3.9, c 1e-1 ---> 4-fold
    
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    print(args)

    
    ### load a mnet
    mnet = MNet(beta=1,in_channels=2,out_size=320-args.base_freq, imgsize=(320,320),poolk=3)
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
    # UNET = UNet(n_channels=1,n_classes=1,bilinear=True,skip=False)
    # unetpath = '/home/huangz78/checkpoints/unet_'+ str(UNET.n_channels) +'.pth'
    # unetpath = '/home/huangz78/checkpoints/unet_1_False.pth'
    UNET = UNet(n_channels=1,n_classes=1,bilinear=False,skip=True)
    unetpath = args.unetpath
    if unetpath is not None:
        checkpoint = torch.load(unetpath)
        UNET.load_state_dict(checkpoint['model_state_dict'])
        print('Unet loaded successfully from: ' + unetpath )
    else:
        UNET.apply(mnet_weights_init)
        print('Unet is randomly initalized!')
    UNET.train()
    print('nn\'s are ready')
        
    # load training data
    train_dir = '/home/huangz78/data/traindata_x.npz'
    # train_sub = np.load(train_dir)['x']
    train_xfull = torch.tensor(np.load(train_dir)['xfull'])
    train_dir = '/home/huangz78/data/traindata_y.npz'
    # train_sub = np.load(train_dir)['x']
    train_yfull = torch.tensor(np.load(train_dir)['yfull'])
    print('train data fft size:', train_yfull.shape)
    print('train data size:', train_xfull.shape)
    
    # load validation data 
    val_dir = '/home/huangz78/data/testdata_x.npz'
    val_xfull = torch.tensor(np.load(val_dir)['xfull'])
    val_dir = '/home/huangz78/data/testdata_y.npz'
    val_yfull = torch.tensor(np.load(val_dir)['yfull'])
    print('validation data fft size:', val_yfull.shape)
    print('validation data size:', val_xfull.shape)
    
    acceleration_fold = str(int(train_xfull.shape[1]/(args.base_freq+args.budget)))
    print(f'corefreq = {args.base_freq}, budget = {args.budget}, this is a {acceleration_fold}-fold training!')
    
    alternating_update_with_unetRecon(mnet,UNET,train_xfull,val_xfull,train_yfulls=train_yfull,val_yfulls=val_yfull,\
                                  maxIter_mb=args.maxItermb,alpha=args.alpha,c=args.c,lr_mb=args.lrb,\
                                  maxRep=args.mnRep,lr_mn=args.lrn,\
                                  corefreq=args.base_freq,budget=args.budget,\
                                  epoch=args.epochs,batchsize=args.batchsize,\
                                  validate_every=args.validate_every,\
                                  verbose=False,save_cp=True,count_start=(args.epoch_start,args.batchind_start),\
                                  histpath=args.histpath)
    print('\n ~Training concluded!')