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
from utils import kplot
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
                        weight_decay:float=0,
                        momentum:float=0,
                        epochs:int=1,
                        batchsize:int=5,
                        val_batchsize:int=5,
                        count_start:tuple=(0,0),
                        dir_checkpoint:str='/mnt/shared_a/checkpoints/mri/',
                        dir_hist:str=None):
        self.model = model
        self.slope = slope
        self.sparsity = sparsity
        self.preselect = preselect
        self.preselect_num = preselect_num
        self.lrm = lrm
        self.lru = lru
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.epochs = epochs
        self.batchsize = batchsize
        self.val_batchsize = val_batchsize
        self.epoch_start = count_start[0]
        self.batchind_start = count_start[1]
        self.dir_checkpoint = dir_checkpoint
        
        
        if histpath is None:
            self.train_loss = []; self.val_loss = []
        else:
            histRec    = np.load(histpath)
            self.train_loss = list(histRec['loss_train'])
            self.val_loss   = list(histRec['loss_val'])
            print('training history file successfully loaded from the path: ', histpath)
        
    def validate(self, valdata):
        valbatchind = 0
        valbatch_nums = int(np.ceil(valdata.shape[0]/self.val_batchsize))
        self.model.eval()
        criterion = nn.L1Loss()
        loss_val = 0
        while (valbatchind < valbatch_nums):
            batch = np.arange(batchsize_val*valbatchind, min(batchsize_val*(valbatchind+1),valdata.shape[0]))
            databatch = normalize_data(valdata[batch,:,:]) if len(batch)>1 else normalize_data(valdata[batch,:,:].unsqueeze(0))
            xstar = databatch.unsqueeze(1)
            ystar = F.fftn(xstar,dim=(2,3),norm='ortho')
            x_recon,_ = self.model(ystar)
            x_recon   = x_recon.detach()
            loss_val += self.criterion(x_recon,xstar)
            valbatchind += 1
        progress_str = f'[{epoch+1}/{epochs}]'
        print('\n' + progress_str + f' validation loss: {loss_val.item()/valbatch_nums}')
        self.val_loss.append(loss_val.item()/valbatch_nums)
    
    def save(self,epoch=0):
        try:
            os.mkdir(self.dir_checkpoint)
            print('Created checkpoint directory')
        except OSError:
            pass
        info_str = f'spar_{sparsity}_base_{preselect_num}'
        torch.save({'model_state_dict': self.model.state_dict()}, self.dir_checkpoint + f'loupe_{info_str}_epoch_{epoch}.pt')
        np.savez(self.dir_checkpoint + f'loupe_{info_str}_epoch_{epoch}_history.npz', loss_train=self.train_loss, loss_val=self.val_loss)
        print(f'\t Checkpoint for Loupe saved after epoch {epoch + 1}!' + '\n')
        
    def run(self, traindata, valdata, save_cp=False):
        '''
        train and test data are assumed to have the shape [size, heg, wid]
        '''
        shape   = traindata.shape[1:3]
        batch_nums  = int(np.ceil(traindata.shape[0]/self.batchsize))
        optimizer = optim.RMSprop([
                    {'params': loupe.samplers.parameters()},
                    {'params': loupe.unet.parameters(),'lr':lru}
                ], lr=lrm, weight_decay=weight_decay, momentum=momentum,eps=1e-10)
        criterion = nn.L1Loss()
        epoch = self.epoch_start 
        batchind = self.batchind_start
        try:
            while epoch<self.epochs:
                while batchind<batch_nums:
                    self.model.train()
                    batch = np.arange(self.batchsize*batchind, min(self.batchsize*(batchind+1),traindata.shape[0]))
                    databatch = normalize_data(traindata[batch,:,:]) if len(batch)>1 else normalize_data(traindata[batch,:,:].unsqueeze(0))
                    xstar = traindata[batch,:,:].unsqueeze(1)
                    ystar = F.fftn(xstar,dim=(2,3),norm='ortho')
                    x_recon,_ = self.model(ystar)

                    loss_train = criterion(x_recon,xstar)
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()
                    progress_str = f'[{epoch+1}/{epochs}][{min(batchsize_train*(batchind+1),traindata.shape[0])}/{traindata.shape[0]}]'
                    print(progress_str + f' training loss: {loss_train.item()}')
                    self.train_loss.append(loss_train.item())
                    batchind += 1   

                # validation eval
                self.validate(valdata)

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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-bt', '--batch-size-train', metavar='BT', type=int, nargs='?', default=5,
                        help='Batch size train', dest='batchsize')
    parser.add_argument('-bv', '--batch-size-val', metavar='BV', type=int, nargs='?', default=5,
                        help='Batch size val', dest='val_batchsize')
    
    parser.add_argument('-lrm', '--learning-rate-mask', metavar='LRM', type=float, nargs='?', default=1e-3,
                        help='Learning rate mask', dest='lrm')
    parser.add_argument('-lru', '--learning-rate-unet', metavar='LRU', type=float, nargs='?', default=1e-4,
                        help='Learning rate unet', dest='lru')
    
    parser.add_argument('-skip', '--unet-skip', type=str, default='True',
                        help='skip connections in structure of unet', dest='unet_skip')
    parser.add_argument('-s', '--sparsity', type=float, default=.25,
                        help='sparsity', dest='sparsity')
    parser.add_argument('-core', '--preselect-num', type=int, default=24,
                        help='preselected number of low frequencies', dest='preselect_num')
    parser.add_argument('-sl', '--slope', type=float, default=1,
                        help='slope used in the sigmoid function', dest='slope')
    
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
    traindata  = np.load('/mnt/shared_a/fastMRI/knee_singlecoil_train.npz')['data']
    valdata    = np.load('/mnt/shared_a/fastMRI/knee_singlecoil_val.npz')['data']
    
    ### loading LOUPE model
    preselect  = True if args.preselect_num > 0 else False
    
    n_channels = 1
    loupe = LOUPE(n_channels=n_channels,unet_skip=args.unet_skip,shape=shape,slope=args.slope,sparsity=args.sparsity,\
                  preselect=preselect,preselect_num=args.preselect_num)
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
                            preselect=preselect, preselect_num=args.preselect_num,
                            lrm=args.lrm,
                            lru=args.lru,
                            weight_decay=0,
                            momentum=0,
                            epochs=args.epochs,
                            batchsize=args.batchsize,
                            val_batchsize=args.val_batchsize,
                            count_start=(args.epoch_start,args.batchind_start),
                            dir_checkpoint:str='/mnt/shared_a/checkpoints/mri/',
                            dir_hist=args.histpath)
    
    trainer.run(traindata, valdata, save_cp=False)