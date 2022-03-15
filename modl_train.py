import numpy as np
import argparse
import os
import sys
import random
import torch
import torch.fft as F
from importlib import reload
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import *

from unet.unet_model import UNet
from unet.unet_model_fbr import Unet
from unet.unet_model_banding_removal_fbr import UnetModel
from mnet.mnet_v2 import MNet
from MoDL.MoDL import MoDL, MoDL_trainer
import copy
dir_checkpoint = '/mnt/shared_a/checkpoints/leo/recon/'

def prepare_data(mode='mnet',mnet=None, base=8, budget=32,batchsize=5,unet_inchans=2,datatype=torch.float,device=torch.device('cpu')):
    if (mode == 'mnet') or (mode=='rand') or (mode == 'equidist') or (mode == 'lfonly') or (mode == 'prob'):
        train_full = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_train.npz')['data'],dtype=datatype)
        val_full   = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_val.npz')['data'],  dtype=datatype)
        for ind in range(train_full.shape[0]):
            train_full[ind,:,:] = train_full[ind,:,:]/train_full[ind,:,:].abs().max()
        for ind in range(val_full.shape[0]):
            val_full[ind,:,:]   = val_full[ind,:,:]/val_full[ind,:,:].abs().max()    
        train_label  = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
        val_label    = torch.reshape(val_full,(val_full.shape[0],1,val_full.shape[1],val_full.shape[2]))
    
    if mode == 'mnet':
        shuffle_inds = torch.randperm(train_full.shape[0])
        train_full   = train_full[shuffle_inds,:,:]

        shuffle_inds = torch.randperm(val_full.shape[0])
        val_full     = val_full[shuffle_inds,:,:]

        train_label  = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
        val_label    = torch.reshape(val_full,(val_full.shape[0],1,val_full.shape[1],val_full.shape[2]))

        ## create train_in and val_in
        train_in, train_mask = mnet_getinput(mnet,train_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,return_mask=True,device=device)
        del train_full
        val_in, val_mask = mnet_getinput(mnet,val_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,return_mask=True,device=device)
        del val_full, mnet
        
        acceleration_fold = str(int(train_in.shape[2]/(base+budget)))
        print(f'\n   Data successfully prepared with the provided MNet for acceleration fold {acceleration_fold}!\n')
        
    if mode == 'rand':
        if (mode=='rand') or (mode == 'equidist') or (mode == 'lfonly'):   
            train_in, train_mask = base_getinput(train_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,return_mask=True,datatype=datatype,mode=mode)
            val_in, val_mask   = base_getinput(val_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,return_mask=True,datatype=datatype,mode=mode)

        del train_full, val_full         
        print(f'data preparation mode is {mode}')       
        
    return train_in, train_label, train_mask, val_in, val_label, val_mask
            
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)   
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=40,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-tb', '--val-batch-size', metavar='TB', type=int, nargs='?', default=5,
                        help='valbatch size', dest='val_batchsize')
    
    parser.add_argument('-es','--epoch-start',metavar='ES',type=int,nargs='?',default=0,
                        help='starting epoch count',dest='epoch_start')
    parser.add_argument('-bis','--batchind-start',metavar='BIS',type=int,nargs='?',default=0,
                        help='starting batchind',dest='batchind_start')
    
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-lrwd', '--lr-weight-decay', metavar='LRWD', type=float, nargs='?', default=0,
                        help='Learning rate weight decay', dest='lrwd')
    
    parser.add_argument('-m','--mode',metavar='M',type=str,nargs='?',default='mnet',
                        help='training mode', dest='mode')
    
    parser.add_argument('-utype', '--unet-type', type=int, default=2,
                        help='type of unet', dest='utype')
    
    parser.add_argument('-cn', '--channel-num', metavar='CN', type=int, nargs='?', default=64,
                        help='channel number of unet', dest='chans')
    parser.add_argument('-uc', '--uchan-in', metavar='UC', type=int, nargs='?', default=2,
                        help='number of input channel of unet', dest='in_chans')
    parser.add_argument('-s','--skip',type=str,default='False',
                        help='residual network application', dest='skip')
    parser.add_argument('-cgs','--cg-steps',type=int,default=4,
                        help='number of CG steps in MoDL model', dest='cg_steps')
    parser.add_argument('-wssim', '--weight-ssim', metavar='WS', type=float, nargs='?', default=5,
                        help='weight of SSIM loss in training', dest='weight_ssim')
    
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=8,
                        help='number of observed low frequencies', dest='base_freq')
    parser.add_argument('-bg','--budget',metavar='BG',type=int,nargs='?',default=32,
                        help='number of high frequencies to sample', dest='budget')
    
    parser.add_argument('-mp', '--mnet-path', type=str, default='/mnt/shared_a/checkpoints/leo/mri/mnet_v2_split_trained_cf_8_bg_32_unet_in_chan_1_epoch9.pt',
                        help='path file for a mnet', dest='mnetpath') 
    # '/mnt/shared_a/checkpoints/leo/mri/mnet_v2_split_trained_cf_8_bg_32_unet_in_chan_1_epoch9.pt'
    parser.add_argument('-up', '--unet-path', type=str, default=None,
                        help='path file for a unet', dest='unetpath')
    parser.add_argument('-mdp', '--modl-path', type=str, default=None,
                        help='path file for a modl model', dest='modlpath')
    parser.add_argument('-hp', '--history-path', type=str, default=None,
                        help='path file for npz file recording training history', dest='histpath')
    parser.add_argument('-ngpu', '--num-gpu', type=int, default=1,
                        help='number of GPUs', dest='ngpu')
    
    parser.add_argument('-sd', '--seed', type=int, default=0,
                        help='random seed', dest='seed')
    return parser.parse_args()
        
if __name__ == '__main__':  
    args = get_args()
    print(args)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    if args.skip == 'False':
        args.skip = False
    elif args.skip == 'True':
        args.skip = True
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
#     unetpath = args.unetpath if args.unetpath is not None else None
    
    modl = MoDL(in_chans=args.in_chans,out_chans=2,chans=args.chans,\
                num_pool_layers=4,drop_prob=0,unet_path=None,CG_steps=args.cg_steps).to(device)
    if args.modlpath is not None:
        checkpoint = torch.load(args.modlpath)
        modl.load_state_dict(checkpoint['model_state_dict'])
        print(f'MoDL model is loaded successfully from: {args.modlpath}')

    if args.mnetpath is not None:
        mnet = MNet(beta=1,in_chans=2,out_size=320-args.base_freq, imgsize=(320,320),poolk=3)
        checkpoint = torch.load(args.mnetpath)
        mnet.load_state_dict(checkpoint['model_state_dict'])
        print(f'MNet is loaded successfully from: {args.mnetpath}')
        mnet.eval()
    else:
        mnet = None
    
    train_in, train_label, train_mask, val_in, val_label, val_mask = prepare_data(mode=args.mode,mnet=mnet,base=args.base_freq, budget=args.budget,batchsize=args.batchsize,unet_inchans=2,datatype=torch.float,device=device)
    del mnet
    infos = f'base{args.base_freq}_budget{args.budget}'
    trainer = MoDL_trainer(modl,
                           save_dir=dir_checkpoint,
                           lr=args.lr,
                           lr_weight_decay=args.lrwd,
                           lr_s_stepsize=40,
                           lr_s_gamma=.8,
                           patience=5,
                           min_lr=1e-6,
                           reduce_factor=.8,
                           count_start=(args.epoch_start,args.batchind_start),
                           p='fro',
                           weight_ssim=args.weight_ssim,
                           ngpu=args.ngpu,                       
                           hist_dir=args.histpath,
                           batchsize=args.batchsize,
                           valbatchsize=args.val_batchsize,
                           epochs=args.epochs,                           
                           mnetpath=args.mnetpath,
                           mode=args.mode,
                           infos=infos)
    
    trainer.run(train_in,train_label,train_mask, val_in,val_label, val_mask, save_cp=True)
    
    
    
    