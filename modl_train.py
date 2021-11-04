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

# def mnet_getinput(mnet,data,base=8,budget=32,batchsize=10,unet_channels=1,return_mask=False,device=torch.device('cpu')):
#     '''
#     assume the input data has the dimension [img,heg,wid]
#     returned data in the format [NCHW]
#     '''   
#     mnet.eval()
#     mnet.to(device)
#     with torch.no_grad():
#         lowfreqmask = mask_naiveRand(data.shape[1],fix=base,other=0,roll=False)[0]
#         heg,wid  = data.shape[1],data.shape[2]
#         imgshape = (heg,wid)
#         yfull = F.fftn(data,dim=(1,2),norm='ortho')
#         y = torch.zeros_like(yfull)
#         y[:,lowfreqmask==1,:] = yfull[:,lowfreqmask==1,:]    
#         x_ifft = torch.zeros(len(yfull),unet_channels,heg,wid,device=torch.device('cpu'))
#         if return_mask:
#             masks = torch.zeros(len(yfull),heg,device=device)

#         batchind  = 0
#         batchnums = int(np.ceil(data.shape[0]/batchsize))
#         while batchind < batchnums:
#             batch = torch.arange(batchsize*batchind, min(batchsize*(batchind+1),data.shape[0]))
#             yfull_b = yfull[batch,:,:].to(device)
#             y_lf    = y[batch,:,:].to(device)
#             y_in    = torch.zeros(len(batch),2,heg,wid,device=device)
#             y_in[:,0,:,:] = torch.real(y_lf)
#             y_in[:,1,:,:] = torch.imag(y_lf)
#             y_in   = F.fftshift(y_in,dim=(2,3))
#             mask_b = F.ifftshift(mnet_wrapper(mnet,y_in,budget,imgshape,normalize=True,detach=True,device=device),dim=(1))

#             if return_mask:
#                 masks[batch,:] = mask_b        
#             y_mnet_b = torch.zeros_like(yfull_b,device=device)
#             for ind in range(len(mask_b)):
#                 y_mnet_b[:,mask_b[ind,:]==1,:] = yfull_b[:,mask_b[ind,:]==1,:]
#             if   unet_channels == 1:
#                 x_ifft[batch,0,:,:] = torch.abs(F.ifftn(y_mnet_b,dim=(1,2),norm='ortho')).cpu()
#             elif unet_channels == 2:
#                 x_ifft_c = F.ifftn(y_mnet_b,dim=(1,2),norm='ortho')
#                 x_ifft[batch,0,:,:] = torch.real(x_ifft_c).cpu()
#                 x_ifft[batch,1,:,:] = torch.imag(x_ifft_c).cpu()
#             batchind += 1
        
#     if return_mask:
#         return x_ifft, masks
#     else:
#         return x_ifft
    
# def rand_getinput(data,base=8,budget=32,batchsize=5,net_inchans=2,datatype=torch.float):
#     '''
#     assume the input data has the dimension [img,heg,wid]
#     returned data in the format [NCHW]
#     '''   
#     yfull = F.fftn(data,dim=(1,2),norm='ortho')
#     y_lf  = torch.zeros_like(yfull)
#     num_pts,heg,wid = data.shape[0],data.shape[1],data.shape[2]
#     batchind  = 0
#     batchnums = int(np.ceil(num_pts/batchsize))      
#     while batchind < batchnums:
#         batch  = torch.arange(batchind*batchsize,min((batchind+1)*batchsize,num_pts))
#         lfmask = mask_naiveRand(data.shape[1],fix=base,other=budget,roll=False)[0] 
#         batchdata_full = yfull[batch,:,:]
#         batchdata      = torch.zeros_like(batchdata_full)
#         batchdata[:,lfmask==1,:] = batchdata_full[:,lfmask==1,:]
#         y_lf[batch,:,:] = batchdata
#         batchind += 1
    
#     if net_inchans == 2:                
#         x_ifft = F.ifftn(y_lf,dim=(1,2),norm='ortho')
#         x_in   = torch.zeros((num_pts,2,heg,wid),dtype=datatype)
#         x_in[:,0,:,:] = torch.real(x_ifft)
#         x_in[:,1,:,:] = torch.imag(x_ifft)       
#     elif net_inchans == 1:
#         x_ifft = torch.abs(F.ifftn(y_lf,dim=(1,2),norm='ortho'))                
#         x_in   = torch.reshape(x_ifft, (num_pts,1,heg,wid)).to(datatype)    
#     return x_in

# def prepare_data(mode='mnet',mnet=None, base=8, budget=32,batchsize=5,unet_inchans=2,datatype=torch.float,device=torch.device('cpu')):
#     if mode == 'mnet':
#         train_full = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_train.npz')['data'],dtype=datatype)
#         val_full   = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_val.npz')['data'],  dtype=datatype)

#         for ind in range(train_full.shape[0]):
#             train_full[ind,:,:] = train_full[ind,:,:]/train_full[ind,:,:].abs().max()
#         for ind in range(val_full.shape[0]):
#             val_full[ind,:,:]  = val_full[ind,:,:]/val_full[ind,:,:].abs().max()

#         shuffle_inds = torch.randperm(train_full.shape[0])
#         train_full   = train_full[shuffle_inds,:,:]

#         shuffle_inds = torch.randperm(val_full.shape[0])
#         val_full     = val_full[shuffle_inds,:,:]

#         train_label  = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
#         val_label    = torch.reshape(val_full,(val_full.shape[0],1,val_full.shape[1],val_full.shape[2]))

#         ## create train_in and val_in
#         train_in, train_mask = mnet_getinput(mnet,train_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,return_mask=True,device=device)
#         del train_full
#         val_in, val_mask = mnet_getinput(mnet,val_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,return_mask=True,device=device)
#         del val_full, mnet
        
#         acceleration_fold = str(int(train_in.shape[2]/(base+budget)))
#         print(f'\n   Data successfully prepared with the provided MNet for acceleration fold {acceleration_fold}!\n')
        
#     if mode == 'rand':
#         ## train a unet to reconstruct images from random mask
# #         train_full = torch.tensor(np.load('/home/huangz78/data/traindata_x.npz')['xfull'],dtype=datatype)
# #         val_full   = torch.tensor(np.load('/home/huangz78/data/testdata_x.npz')['xfull'], dtype=datatype)         
#         train_full = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_train.npz')['data'],dtype=datatype)
#         val_full   = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_val.npz')['data']  ,dtype=datatype)
#         for ind in range(train_full.shape[0]):
#             train_full[ind,:,:] = train_full[ind,:,:]/train_full[ind,:,:].abs().max()
#         for ind in range(val_full.shape[0]):
#             val_full[ind,:,:]   = val_full[ind,:,:]/val_full[ind,:,:].abs().max()        

#         train_in = rand_getinput(train_full,base=base,budget=budget,batchsize=batchsize,net_inchans=unet_inchans,datatype=datatype)
#         val_in   = rand_getinput(val_full,  base=base,budget=budget,batchsize=batchsize,net_inchans=unet_inchans,datatype=datatype)

#         train_label = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
#         val_label   = torch.reshape(val_full,(val_full.shape[0],1,val_full.shape[1],val_full.shape[2]))
#         del train_full, val_full       
        
#         train_mask = torch.ones(train_in.shape)
#         val_mask   = torch.ones(val_in.shape)

# #     elif mode == 'greedy':
# #         ## train a unet to reconstruct images from greedy mask
# #         assert net.in_chans==1
# #         imgs  = torch.tensor( np.load('/home/huangz78/data/data_gt.npz')['imgdata'] ).permute(2,0,1)
# #         masks = torch.tensor( np.load('/home/huangz78/data/data_gt_greedymask.npz')['mask'].T ) # labels are already rolled
# #         xs    = torch.zeros((imgs.shape[0],1,imgs.shape[1],imgs.shape[2]),dtype=torch.float)

# #         for ind in range(imgs.shape[0]):
# #             imgs[ind,:,:] = imgs[ind,:,:]/torch.max(torch.abs(imgs[ind,:,:]))
# #             y = F.fftshift(F.fftn(imgs[ind,:,:],dim=(0,1),norm='ortho'))
# #             mask = masks[ind,:]
# #             ysub = torch.zeros(y.shape,dtype=y.dtype)
# #             ysub[mask==1,:] = y[mask==1,:]
# #             xs[ind,0,:,:] = torch.abs(F.ifftn(torch.fft.ifftshift(ysub),dim=(0,1),norm='ortho'))

# #         imgNum = imgs.shape[0]
# #         traininds, valinds = train_test_split(np.arange(imgNum),random_state=0,shuffle=True,train_size=round(imgNum*0.8))
# #         np.savez('/home/huangz78/data/inds_rec.npz',traininds=traininds,valinds=valinds)
# #         Heg,Wid,n_train,n_val = imgs.shape[1],imgs.shape[2],len(traininds),len(valinds)

# #         train_full = imgs[traininds,:,:]
# #         train_label= torch.reshape(train_full,(n_train,1,Heg,Wid))
# #         valfull   = imgs[valinds,:,:]
# #         val_label = torch.reshape(valfull,(n_val,1,Heg,Wid))
# #         train_in   = xs[traininds,:,:,:]
# #         val_in    = xs[valinds ,:,:,:]
# #         print('n_train = {}, n_val = {}'.format(n_train,n_val))
# #         del xs, imgs, masks
        
#     return train_in, train_label, train_mask, val_in, val_label, val_mask
            
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
    
    parser.add_argument('-cn', '--channel-num', metavar='CN', type=int, nargs='?', default=32,
                        help='channel number of unet', dest='chans')
    parser.add_argument('-uc', '--uchan-in', metavar='UC', type=int, nargs='?', default=2,
                        help='number of input channel of unet', dest='in_chans')
    parser.add_argument('-s','--skip',type=str,default='False',
                        help='residual network application', dest='skip')
    parser.add_argument('-cgs','--cg-steps',type=int,default=6,
                        help='number of CG steps in MoDL model', dest='cg_steps')
    parser.add_argument('-wssim', '--weight-ssim', metavar='WS', type=float, nargs='?', default=5,
                        help='weight of SSIM loss in training', dest='weight_ssim')
    
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=8,
                        help='number of observed low frequencies', dest='base_freq')
    parser.add_argument('-bg','--budget',metavar='BG',type=int,nargs='?',default=32,
                        help='number of high frequencies to sample', dest='budget')
    
    parser.add_argument('-mp', '--mnet-path', type=str, default='/mnt/shared_a/checkpoints/leo/mri/mnet_v2_split_trained_cf_8_bg_32_unet_in_chan_1_epoch9.pt',
                        help='path file for a mnet', dest='mnetpath')
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
                           mnetpath=args.mnetpath)
    
    trainer.run(train_in,train_label,train_mask, val_in,val_label, val_mask, save_cp=True)
    
    
    
    