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

# import unet_model
from unet.unet_model import UNet
from unet.unet_model_fbr import Unet
from unet.unet_model_banding_removal_fbr import UnetModel
from mnet.mnet_v2 import MNet
import copy
dir_checkpoint = '/mnt/shared_a/checkpoints/leo/recon/'

def prepare_data(mode='mnet',mnet=None, base=8, budget=32,batchsize=5,unet_inchans=2,datatype=torch.float,device=torch.device('cpu')):
    if mode == 'mnet':
        train_full = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_train.npz')['data'],dtype=datatype)
        val_full   = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_val.npz')['data'],  dtype=datatype)

        for ind in range(train_full.shape[0]):
            train_full[ind,:,:] = train_full[ind,:,:]/train_full[ind,:,:].abs().max()
        for ind in range(val_full.shape[0]):
            val_full[ind,:,:]  = val_full[ind,:,:]/val_full[ind,:,:].abs().max()

        shuffle_inds = torch.randperm(train_full.shape[0])
        train_full   = train_full[shuffle_inds,:,:]

        shuffle_inds = torch.randperm(val_full.shape[0])
        val_full     = val_full[shuffle_inds,:,:]

        train_label  = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
        val_label    = torch.reshape(val_full,(val_full.shape[0],1,val_full.shape[1],val_full.shape[2]))

        ## create train_in and val_in
        train_in = mnet_getinput(mnet,train_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,return_mask=False,device=device)
        del train_full
        val_in = mnet_getinput(mnet,val_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,return_mask=False,device=device)
        del val_full, mnet
        
        acceleration_fold = str(int(train_in.shape[2]/(base+budget)))
        print(f'\n   Data successfully prepared with the provided MNet for acceleration fold {acceleration_fold}!\n')
        
    if (mode=='rand') or (mode == 'equidist') or (mode == 'lfonly'):
        ## train a unet to reconstruct images from random mask
#             train_full = torch.tensor(np.load('/home/huangz78/data/traindata_x.npz')['xfull'],dtype=datatype)
#             val_full  = torch.tensor(np.load('/home/huangz78/data/valdata_x.npz')['xfull'] ,dtype=datatype)         
        train_full = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_train.npz')['data'],dtype=datatype)
        val_full  = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_val.npz')['data']  ,dtype=datatype)
        for ind in range(train_full.shape[0]):
            train_full[ind,:,:] = train_full[ind,:,:]/train_full[ind,:,:].abs().max()
        for ind in range(val_full.shape[0]):
            val_full[ind,:,:]   = val_full[ind,:,:]/val_full[ind,:,:].abs().max()        

        train_in = base_getinput(train_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,datatype=datatype,mode=mode)
        val_in   = base_getinput(val_full,base=base,budget=budget,batchsize=batchsize,unet_channels=unet_inchans,datatype=datatype,mode=mode)

        train_label = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
        val_label  = torch.reshape(val_full,(val_full.shape[0],1,val_full.shape[1],val_full.shape[2]))
        del train_full, val_full         
        print(f'data preparation mode is {mode}')
    elif mode == 'greedy':
        ## train a unet to reconstruct images from greedy mask
        assert net.in_chans==1
        imgs  = torch.tensor( np.load('/home/huangz78/data/data_gt.npz')['imgdata'] ).permute(2,0,1)
        masks = torch.tensor( np.load('/home/huangz78/data/data_gt_greedymask.npz')['mask'].T ) # labels are already rolled
        xs    = torch.zeros((imgs.shape[0],1,imgs.shape[1],imgs.shape[2]),dtype=torch.float)

        for ind in range(imgs.shape[0]):
            imgs[ind,:,:] = imgs[ind,:,:]/torch.max(torch.abs(imgs[ind,:,:]))
            y = F.fftshift(F.fftn(imgs[ind,:,:],dim=(0,1),norm='ortho'))
            mask = masks[ind,:]
            ysub = torch.zeros(y.shape,dtype=y.dtype)
            ysub[mask==1,:] = y[mask==1,:]
            xs[ind,0,:,:] = torch.abs(F.ifftn(torch.fft.ifftshift(ysub),dim=(0,1),norm='ortho'))

        imgNum = imgs.shape[0]
        traininds, valinds = train_test_split(np.arange(imgNum),random_state=0,shuffle=True,train_size=round(imgNum*0.8))
        np.savez('/home/huangz78/data/inds_rec.npz',traininds=traininds,valinds=valinds)
        Heg,Wid,n_train,n_val = imgs.shape[1],imgs.shape[2],len(traininds),len(valinds)

        train_full = imgs[traininds,:,:]
        train_label= torch.reshape(train_full,(n_train,1,Heg,Wid))
        valfull    = imgs[valinds,:,:]
        val_label  = torch.reshape(valfull,(n_val,1,Heg,Wid))
        train_in   = xs[traininds,:,:,:]
        val_in     = xs[valinds ,:,:,:]
        print('n_train = {}, n_val = {}'.format(n_train,n_val))
        del xs, imgs, masks
        
    return train_in, train_label, val_in, val_label

class unet_trainer:
    def __init__(self,
                 net: nn.Module,
                 lr:float=1e-3,
                 lr_weight_decay:float=1e-8,
                 lr_s_stepsize:int=40,
                 lr_s_gamma:float=.1,
                 patience:int=5,
                 min_lr:float=5e-6,
                 reduce_factor:float=.8,
                 count_start:tuple=(0,0),
                 p=1,
                 weight_ssim:float=.7,
                 ngpu:int=0,
                 dir_checkpoint: str=dir_checkpoint,
                 dir_hist=None,
                 dir_mnet=None,
                 batchsize:int=5,
                 val_batchsize:int=5,
                 epochs:int=5,
                 modename:str=None
                 ):
        self.ngpu = ngpu
        self.device = torch.device('cuda:0') if ngpu > 0 else torch.device('cpu')
        self.net = net.to(device)
        self.lr = lr
        self.lr_weight_decay = lr_weight_decay
        self.lr_s_stepsize = lr_s_stepsize
        self.lr_s_gamma = lr_s_gamma
        self.patience = patience
        self.min_lr = min_lr
        self.reduce_factor = reduce_factor
        self.count_start = count_start
        self.p = p
        self.weight_ssim = weight_ssim
        self.dir_checkpoint = dir_checkpoint
        self.dir_hist = dir_hist
        self.dir_mnet = dir_mnet
        
        self.batchsize = batchsize
        self.val_batchsize = val_batchsize
        self.epochs = epochs
        self.modename = modename
            
        if self.dir_hist is None:
            self.train_df_loss   = list([]); self.val_df_loss   = list([]);  self.train_loss_epoch = list([])
            self.train_ssim_loss = list([]); self.val_ssim_loss = list([]);  self.val_loss_epoch   = list([])
            self.valloss_old = np.inf
        else:
            histRec = np.load(self.dir_hist)
            self.train_df_loss    = list(histRec['trainloss_df'])
            self.train_ssim_loss  = list(histRec['trainloss_ssim'])
            self.train_loss_epoch = list(histRec['trainloss_epoch'])
            
            self.val_df_loss    = list(histRec['valloss_df'])
            self.val_ssim_loss  = list(histRec['valloss_ssim'])
            self.val_loss_epoch = list(histRec['valloss_epoch'])
            print('training history file successfully loaded from the path: ', self.dir_hist)
            self.valloss_old = self.val_loss[-1]
       
        self.save_model = True
        
    def validate(self,val_in,val_label,epoch=-1):
        valloss = 0
        data_fidelity_loss = 0
        ssim_loss = 0
        self.net.eval()
        n_val  = val_in.shape[0]
        val_batchnums = int(np.ceil(n_val/self.val_batchsize))
        with torch.no_grad():
            for ind in range(val_batchnums):                
                v_b      = torch.arange(ind*self.val_batchsize,min((ind+1)*self.val_batchsize,n_val))
                valin    = val_in[v_b,:,:,:].to(self.device)
                vallabel = val_label[v_b,:,:,:].to(self.device)
                pred     = self.net(valin).detach()
#                     valloss += criterion(pred, vallabel)*len(v_b)
                data_fidelity_loss += lpnorm(pred,vallabel,p=self.p,mode='sum')
                ssim_loss          += -ssim_uniform(pred,vallabel,reduction='mean')                        
            df_loss_epoch   = data_fidelity_loss.item()/n_val
            ssim_loss_epoch = ssim_loss.item()/val_batchnums
            valloss_epoch   = df_loss_epoch  + self.weight_ssim * ssim_loss_epoch
            
            self.val_df_loss.append(df_loss_epoch)
            self.val_ssim_loss.append(ssim_loss_epoch)
            self.val_loss_epoch.append(valloss_epoch)
            if valloss_epoch < self.valloss_old:
                self.valloss_old = copy.deepcopy(valloss_epoch)
                self.save_model = True
            else:
                self.save_model = False
        print(f'\n\t[{epoch+1}/{self.epochs}]  loss/VAL: {valloss_epoch:.4f}, data fidelity loss: {df_loss_epoch:.4f} / 0, ssim loss: {ssim_loss_epoch:.4f} / -1')
        torch.cuda.empty_cache()
        del valin,vallabel,pred
        self.net.train()
        return valloss_epoch
    
    def save(self,epoch=0,batchind=None):
        if self.modename is None:
            recName = self.dir_checkpoint + f'TrainRec_unet_fbr_{str(self.net.in_chans)}_chans_{str(self.net.chans)}_epoch_{str(epoch)}.npz'
        else:
            recName = self.dir_checkpoint + f'TrainRec_unet_fbr_{str(self.net.in_chans)}_chans_{str(self.net.chans)}_{self.modename}_epoch_{str(epoch)}.npz'
        np.savez(recName,trainloss_df=self.train_df_loss, 
                         trainloss_ssim=self.train_ssim_loss, 
                         trainloss_epoch=self.train_loss_epoch,
                         valloss_df=self.val_df_loss, 
                         valloss_ssim=self.val_ssim_loss, 
                         valloss_epoch=self.val_loss_epoch, 
                         mnetpath=self.dir_mnet)
        print(f'\t History saved after epoch {epoch + 1}!')
        if (self.save_model) or (batchind is not None):
            print(f'Training mode: {self.modename}')
            if self.modename is None:
                modelName = self.dir_checkpoint + f'unet_fbr_{str(self.net.in_chans)}_chans_{str(self.net.chans)}_epoch_{str(epoch)}.pt'
            else:
                modelName = self.dir_checkpoint + f'unet_fbr_{str(self.net.in_chans)}_chans_{str(self.net.chans)}_{self.modename}_epoch_{str(epoch)}.pt'
            torch.save({'model_state_dict': self.net.state_dict()}, modelName)  
            if batchind is None:
                print(f'\t Checkpoint saved after epoch {epoch + 1}!')
            else:
                print(f'\t Checkpoint saved at Python epoch {epoch}, batchnum {batchind}!')
                print('Model is saved after interrupt~')
        self.save_model = False
        torch.cuda.empty_cache()
    
    def run(self,
            train_in: torch.Tensor,
            train_label: torch.Tensor,
            val_in: torch.Tensor,
            val_label: torch.Tensor,
            save_cp=True):
        
        optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=self.lr_weight_decay)
        #         optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.lr_weight_decay)
        #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_s_stepsize, gamma=self.lr_s_gamma)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience, verbose=True, min_lr=self.min_lr,factor=self.reduce_factor)
        #         criterion = nn.MSELoss() 
        n_train = train_in.shape[0]

        global_step = 0
        train_batchnums = int(np.ceil(n_train/self.batchsize))            
        _ = self.validate(val_in,val_label)
        try:    
            for epoch in range(self.count_start[0],self.epochs):
                epoch_loss = 0
                batchind   = 0 if epoch!=self.count_start[0] else self.count_start[1]
                while batchind < train_batchnums:        
                    batch = torch.arange(batchind*self.batchsize,min((batchind+1)*self.batchsize,n_train))
                    imgbatch   = train_in[batch,:,:,:].to(self.device)
                    labelbatch = train_label[batch,:,:,:].to(self.device)
                    pred = self.net(imgbatch)
                    
                    data_fidelity_loss = lpnorm(pred,labelbatch,p=self.p,mode='mean')
                    ssim_loss = -ssim_uniform(pred,labelbatch,reduction='mean')
                    loss = data_fidelity_loss  + self.weight_ssim * ssim_loss
                    
                    self.train_df_loss.append(data_fidelity_loss.item())
                    self.train_ssim_loss.append(ssim_loss.item())
                    epoch_loss += loss.item()
                    print(f'[{global_step+1}][{epoch+1}/{self.epochs}][{batchind}/{train_batchnums}]  loss/train: {loss.item():.4f}, data fidelity loss: {data_fidelity_loss.item():.4f}, ssim: {-ssim_loss.item():.4f}')
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
                    optimizer.step()
                    
                    torch.cuda.empty_cache()
                    del imgbatch, labelbatch, pred
                    batchind    += 1
                    global_step += 1
                self.train_loss_epoch.append(epoch_loss/train_batchnums)
                
                valloss_epoch = self.validate(val_in,val_label,epoch=epoch)                
    #             scheduler.step()
                scheduler.step(valloss_epoch)

                if save_cp:  
                    self.save(epoch=epoch) 
        except KeyboardInterrupt:
            print('Keyboard Interrupted! Exit~')
            if save_cp:
                self.save(epoch=epoch,batchind=batchind)
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
            
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)   
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=40,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
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
    
    parser.add_argument('-m','--mode',metavar='M',type=str,nargs='?',default='rand',
                        help='training mode', dest='mode')
    
    parser.add_argument('-utype', '--unet-type', type=int, default=2,
                        help='type of unet', dest='utype')
    
    parser.add_argument('-cn', '--channel-num', metavar='CN', type=int, nargs='?', default=64,
                        help='channel number of unet', dest='chans')
    parser.add_argument('-uc', '--uchan-in', metavar='UC', type=int, nargs='?', default=2,
                        help='number of input channel of unet', dest='in_chans')
    parser.add_argument('-s','--skip',type=str,default='False',
                        help='residual network application', dest='skip')
    
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=8,
                        help='number of observed low frequencies', dest='base_freq')
    parser.add_argument('-bg','--budget',metavar='BG',type=int,nargs='?',default=32,
                        help='number of high frequencies to sample', dest='budget')
    
    parser.add_argument('-mp', '--mnet-path', type=str, default=None,
                        help='path file for a mnet', dest='mnetpath') # '/mnt/shared_a/checkpoints/leo/mri/mnet_v2_split_trained_cf_8_bg_32_unet_in_chan_1_epoch9.pt'
    parser.add_argument('-up', '--unet-path', type=str, default=None,
                        help='path file for a unet', dest='unetpath')
    parser.add_argument('-hp', '--history-path', type=str, default=None,
                        help='path file for npz file recording training history', dest='histpath')
    parser.add_argument('-ngpu', '--num-gpu', type=int, default=1,
                        help='number of GPUs', dest='ngpu')
    
    parser.add_argument('-sd', '--seed', type=int, default=0,
                        help='random seed', dest='seed')
    parser.add_argument('-wssim', '--weight-ssim', metavar='WS', type=float, nargs='?', default=5,
                        help='weight of SSIM loss in training', dest='weight_ssim')
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
    
    if args.utype == 1:
        unet = UNet(in_chans=args.in_chans,n_classes=1,bilinear=(not args.skip),skip=args.skip).to(device)
    elif args.utype == 2: ## Unet from FBR
        unet = Unet(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=4,drop_prob=0).to(device)
    elif args.utype == 3: ## Unet from FBR, res
        unet = UnetModel(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=4,drop_prob=0,variant='res').to(device)
    elif args.utype == 4: ## Unet from FBR, dense
        unet = UnetModel(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=4,drop_prob=0,variant='dense').to(device)
    
    if args.unetpath is not None:
        checkpoint = torch.load(args.unetpath)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print('Unet loaded successfully from: ' + args.unetpath )
    else:
        #         unet.apply(nn_weights_init)
        print('Unet is randomly initalized!')
    unet.train()        
    
    if args.mnetpath is not None:
        mnet = MNet(beta=1,in_chans=2,out_size=320-args.base_freq, imgsize=(320,320),poolk=3)
        checkpoint = torch.load(args.mnetpath)
        mnet.load_state_dict(checkpoint['model_state_dict'])
        print('MNet loaded successfully from: ' + args.mnetpath)
        mnet.eval()
    else:
        mnet = None
    
    device = torch.device('cuda:0') if args.ngpu > 0 else torch.device('cpu')
    
    train_in, train_label, val_in, val_label = prepare_data(mode=args.mode,mnet=mnet, base=args.base_freq, budget=args.budget,batchsize=args.batchsize,unet_inchans=unet.in_chans,datatype=torch.float,device=device)
    del mnet
    trainer = unet_trainer(unet,
                           lr=args.lr,
                           lr_weight_decay=args.lrwd,
                           lr_s_stepsize=40,
                           lr_s_gamma=.8,
                           patience=5,
                           min_lr=1e-6,
                           reduce_factor=.8,
                           count_start=((args.epoch_start,args.batchind_start)),
                           p='fro',
                           weight_ssim=args.weight_ssim,
                           ngpu=args.ngpu,
                           dir_checkpoint=dir_checkpoint,
                           dir_hist=args.histpath,
                           dir_mnet=args.mnetpath,
                           batchsize=args.batchsize,
                           val_batchsize=args.val_batchsize,
                           epochs=args.epochs,
                           modename=args.mode)
    
    trainer.run(train_in,train_label,val_in,val_label,save_cp=True)
    
    
    
    