import torch
import argparse
import sys,os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.optim as optim
from torch.nn import functional as Func
from typing import List
import random
import loupe_env.line_sampler
from loupe_env.line_sampler import *
from loupe_env.loupe_wrap import *

import utils
from utils import *
from unet.unet_model import UNet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


from sklearn.model_selection import train_test_split

    
def sigmoid_binarize(M,threshold=0.5):
    sigmoid = nn.Sigmoid()
    mask = sigmoid(M)
    mask_pred = torch.ones_like(mask)
    for ind in range(M.shape[0]):
        mask_pred[ind,mask[ind,:]<=threshold] = 0
    return mask_pred

def normalize_data(batch):
    for img in range(len(batch)):
        batch[img,:,:] /= torch.max(torch.abs(batch[img,:,:]))
    return batch

class loupe_trainer:
    def __init__(self, model:nn.Module,
                        slope:float=5.,
                        sparsity:float=.25,
                        preselect:bool=True,
                        preselect_num:int=24,
                        lrm:float=1e-3,
                        lru:float=1e-4,
                        weight_ssim:float=.7,
                        weight_decay:float=0,
                        momentum:float=0,
                        epochs:int=1,
                        batchsize:int=5,
                        val_batchsize:int=5,
                        count_start:tuple=(0,0),
                        dir_checkpoint:str='/mnt/shared_a/checkpoints/leo/mri/',
                        dir_hist:str=None,
                        device=torch.device('cpu')):
        self.model = model
        self.slope = slope
        self.sparsity = sparsity
        self.preselect = preselect
        self.preselect_num = preselect_num
        self.lrm = lrm
        self.lru = lru
        self.weight_ssim = weight_ssim
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.epochs = epochs
        self.batchsize = batchsize
        self.val_batchsize = val_batchsize
        self.epoch_start = count_start[0]
        self.batchind_start = count_start[1]
        self.dir_checkpoint = dir_checkpoint
        self.device = device
        self.acceler_fold = int(1./sparsity)
        if dir_hist is None:
            self.loss_train = []; self.l2_train = []; self.ssim_train = []
            self.loss_val   = []; self.l2_val   = []; self.ssim_val   = []
        else:
            histRec         = np.load(dir_hist)
            self.loss_train = list(histRec['loss_train'])
            self.l2_train   = list(histRec['l2_train'])
            self.ssim_train = list(histRec['ssim_train'])
            self.l2_val     = list(histRec['l2_val'])
            self.ssim_val   = list(histRec['ssim_val'])
            self.loss_val   = list(histRec['loss_val'])
            print('training history file successfully loaded from the path: ', dir_hist)
    def empty_cache(self):
        torch.cuda.empty_cache()
        torch.backends.cuda.cufft_plan_cache.clear()
        
    def validate(self, valdata,epoch=0):
        valbatchind = 0
        n_val = valdata.shape[0]
        val_batchnums = int(np.ceil(n_val/self.val_batchsize))
        self.model.eval()
        l2loss = 0
        ssim   = 0
        valloss = 0
        with torch.no_grad():
            while (valbatchind < val_batchnums):
                batch = np.arange(self.val_batchsize*valbatchind, min(self.val_batchsize*(valbatchind+1),n_val))
                databatch = normalize_data(valdata[batch,:,:]) if len(batch)>1 else normalize_data(valdata[batch,:,:].unsqueeze(0))
                xstar = databatch.unsqueeze(1).to(device)
                ystar = F.fftn(xstar,dim=(2,3),norm='ortho')
                x_recon,_ = self.model(ystar)
                x_recon   = x_recon.detach()
                l2loss += lpnorm(x_recon,xstar,p='fro',mode='sum')
                ssim   += ssim_uniform(x_recon,xstar,reduction='mean')
                valbatchind += 1
        
        df_loss_epoch   = l2loss.item()/n_val
        ssim_loss_epoch = -ssim.item()/val_batchnums
        valloss_epoch   = df_loss_epoch  + self.weight_ssim * ssim_loss_epoch
        progress_str = f'[{epoch+1}/{self.epochs}]'
        print('\n' + progress_str + f' L2 loss/VAL: {df_loss_epoch}, SSIM/VAL: {-ssim_loss_epoch}, loss/VAL: {valloss_epoch}')
        self.l2_val.append(df_loss_epoch)
        self.ssim_val.append(-ssim_loss_epoch)
        self.loss_val.append(valloss_epoch)               
    
    def save(self,epoch=0):
        try:
            os.mkdir(self.dir_checkpoint)
            print('Created checkpoint directory')
        except OSError:
            pass
        torch.save({'model_state_dict': self.model.state_dict()}, self.dir_checkpoint + f'loupe_{self.acceler_fold}fold_base_{self.preselect_num}_epoch_{epoch}.pt')
        histfilename = self.dir_checkpoint + f'loupe_{self.acceler_fold}fold_base_{self.preselect_num}_epoch_{epoch}_history.npz'
        np.savez(histfilename, loss_train=self.loss_train, l2_train=self.l2_train,ssim_train=self.ssim_train, l2_val=self.l2_val,ssim_val=self.ssim_val,loss_val=self.loss_val)
        print(f'\t Checkpoint for Loupe saved after epoch {epoch + 1}!' + '\n')
        
    def run(self, traindata, valdata, save_cp=False):
        '''
        train and test data are assumed to have the shape [size, heg, wid]
        '''
        shape   = traindata.shape[1:3]
        batch_nums  = int(np.ceil(traindata.shape[0]/self.batchsize))
        optimizer = optim.RMSprop([{'params': loupe.samplers.parameters()},
                                   {'params': loupe.unet.parameters(),'lr':self.lru}
                                  ], lr=self.lrm, weight_decay=self.weight_decay,eps=1e-10)
        epoch = self.epoch_start 
        batchind = self.batchind_start
        try:
            while epoch<self.epochs:
                while batchind<batch_nums:
                    self.model.train()
                    batch = np.arange(self.batchsize*batchind, min(self.batchsize*(batchind+1),traindata.shape[0]))
                    databatch = normalize_data(traindata[batch,:,:]) if len(batch)>1 else normalize_data(traindata[batch,:,:].unsqueeze(0))
                    xstar = databatch.unsqueeze(1).to(device)
                    ystar = F.fftn(xstar,dim=(2,3),norm='ortho')
                    x_recon,_ = self.model(ystar)

                    l2loss     = lpnorm(x_recon,xstar,p='fro',mode='mean')
                    ssimloss   = -ssim_uniform(x_recon,xstar,reduction='mean')
                    loss_train = l2loss + self.weight_ssim * ssimloss
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()
                    progress_str = f'[{epoch+1}/{self.epochs}][{min(self.batchsize*(batchind+1),traindata.shape[0])}/{traindata.shape[0]}]'
                    print(progress_str + f' L2loss/train: {l2loss.item()}, SSIM/train: {-ssimloss.item()}, loss/train: {loss_train.item()}')
                    self.l2_train.append(l2loss.item())
                    self.ssim_train.append(-ssimloss.item())
                    self.loss_train.append(loss_train.item())
                    batchind += 1   

                # validation eval
                self.validate(valdata,epoch=epoch)
                self.empty_cache()
                # saving models
                if save_cp:
                    self.save(epoch=epoch)
                epoch += 1
                batchind = 0
        except KeyboardInterrupt: # need debug
            print('Keyboard Interrupted! Exit ~')
            self.save(epoch=epoch)
            print('Model is saved after keyboard interruption~')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)   
    
def get_args():
    parser = argparse.ArgumentParser(description='Train the Loupe model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=40,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-bt', '--batch-size-train', metavar='BT', type=int, nargs='?', default=10,
                        help='Batch size train', dest='batchsize')
    parser.add_argument('-bv', '--batch-size-val', metavar='BV', type=int, nargs='?', default=5,
                        help='Batch size val', dest='val_batchsize')
    
    parser.add_argument('-lrm', '--learning-rate-mask', metavar='LRM', type=float, nargs='?', default=1e-4,
                        help='Learning rate mask', dest='lrm')
    parser.add_argument('-lru', '--learning-rate-unet', metavar='LRU', type=float, nargs='?', default=1e-5,
                        help='Learning rate unet', dest='lru')
    
    parser.add_argument('-skip', '--unet-skip', type=str, default='False',
                        help='skip connections in structure of unet', dest='unet_skip')
    parser.add_argument('-s', '--sparsity', type=float, default=.125,
                        help='sparsity', dest='sparsity')
    parser.add_argument('-core', '--preselect-num', type=int, default=8,
                        help='preselected number of low frequencies', dest='preselect_num')
    parser.add_argument('-sl', '--slope', type=float, default=1,
                        help='slope used in the sigmoid function', dest='slope')
    parser.add_argument('-wssim', '--weight-ssim', metavar='WS', type=float, nargs='?', default=0,
                        help='weight of SSIM loss in training', dest='weight_ssim')
    
    parser.add_argument('-mp', '--model-path', type=str, default=None,
                        help='path file for a loupe model', dest='modelpath')
    parser.add_argument('-hp', '--history-path', type=str, default=None,
                        help='path file for npz file recording training history', dest='histpath')
    
    parser.add_argument('-es','--epoch-start',metavar='ES',type=int,nargs='?',default=0,
                        help='starting epoch count',dest='epoch_start')
    parser.add_argument('-bis','--batchind-start',metavar='BIS',type=int,nargs='?',default=0,
                        help='starting batchind',dest='batchind_start')
    
    parser.add_argument('-save', '--save-model', type=str, default='True',
                        help='whether to save model', dest='save_cp')
    parser.add_argument('-ngpu','--number-gpu',metavar='NGPU',type=int,nargs='?',default=0,
                        help='number of GPUs',dest='ngpu')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    
    if args.unet_skip == 'True':
        args.unet_skip = True
    elif args.unet_skip == 'False':
        args.unet_skip = False
        
    if args.save_cp == 'True':
        args.save_cp = True
    elif args.save_cp == 'False':
        args.save_cp = False
    
    ### loading data
    traindata  = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_train.npz')['data'],dtype=torch.float)
    valdata    = torch.tensor(np.load('/mnt/shared_a/fastMRI/knee_singlecoil_val.npz')['data'],dtype=torch.float)
    
    ### loading LOUPE model
    preselect  = True if args.preselect_num > 0 else False
    device = torch.device('cuda:0') if args.ngpu > 0 else torch.device('cpu')
    n_channels = 1
    shape = [traindata.shape[1],traindata.shape[2]]
    loupe = LOUPE(in_chans=n_channels,unet_skip=args.unet_skip,shape=shape,slope=args.slope,sparsity=args.sparsity,\
                  preselect=preselect,preselect_num=args.preselect_num).to(device)
    if args.modelpath is not None:
        checkpoint = torch.load(modelpath)
        loupe.load_state_dict(checkpoint['model_state_dict'])
        print('LOUPE model successfully loaded from the path: ', args.modelpath)
    else:
        print('LOUPE model is randomly initialized~')
    loupe.train()
    
    ### training    
    trainer = loupe_trainer(loupe,
                            slope=args.slope,
                            sparsity=args.sparsity,
                            preselect=preselect, 
                            preselect_num=args.preselect_num,
                            lrm=args.lrm,
                            lru=args.lru,
                            weight_decay=0,
                            weight_ssim=args.weight_ssim,
                            epochs=args.epochs,
                            batchsize=args.batchsize,
                            val_batchsize=args.val_batchsize,
                            count_start=(args.epoch_start,args.batchind_start),
                            dir_checkpoint='/mnt/shared_a/checkpoints/leo/mri/',
                            dir_hist=args.histpath,
                            device=device)
    
    trainer.run(traindata, valdata, save_cp=args.save_cp)