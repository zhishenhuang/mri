import sys
import torch
import torch.nn as nn
import numpy as np
import random
import copy
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import RMSprop, Adam
sys.path.insert(0,'/home/leo/mri/unet/')
from unet_model_fbr import Unet
import utils

def CG(output, tol ,L, smap, mask, alised_image):
    return utils.CG.apply(output, tol, L, smap, mask, alised_image)

class MoDL(nn.Module):
    def __init__(self,
                 in_chans: int = 2,
                 out_chans: int = 2,
                 chans: int = 32,
                 num_pool_layers: int = 4,
                 drop_prob: float = 0.0,
                 unet_path: Path = None,
                 CG_steps: int = 5,
                 CG_tol: float = 5e-5, 
                 CG_L: float = 1.,
    ):
        super(MoDL, self).__init__()
        self.model = Unet(in_chans=in_chans,out_chans=out_chans,num_pool_layers=num_pool_layers,drop_prob=drop_prob)
        self.CG_steps = CG_steps
        self.CG_tol   = CG_tol
        self.CG_L     = CG_L
        if unet_path is not None:
            checkpoint = torch.load(unet_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Unet in MoDL model is initialized with {args.model_path}')
                                  
    def forward(self,data,smap,mask):
        '''
        data input is assumed to have the data format NCHW, with C=2
        '''
        MoDL_in = copy.deepcopy(data)
        for cg_step in range(self.CG_steps):
            tmpout  = self.model(MoDL_in)
            tmpout2 = CG(tmpout, tol=self.CG_tol, L=self.CG_L, smap=smap, mask=mask, alised_image=data)
            MoDL_in = tmpout2
            
        return MoDL_in
                 
class MoDL_trainer:
    def __init__(self,
                 model: nn.Module,
                 model_dir: Path,
                 mnet_dir: Path = None):       
        self.model = model
        self.model_dir = model_dir
        self.mnet_dir = mnet_dir
        
        self.epochs = epochs
        self.val_loss = []
        self.train_loss_epoch = []
        self.train_loss = []
    
    def empty_cache(self):
        torch.cuda.empty_cache()
        torch.backends.cuda.cufft_plan_cache.clear()
    
    def save_model(self,epoch=0):
        recName = dir_checkpoint + f'Hist_MoDL_{str(self.model.in_chans)}_chans_{str(self.model.chans)}_epoch_{str(epoch)}.npz'
        np.savez(recName,\
                 trainloss=self.train_loss,trainloss_epoch=self.train_loss_epoch,\
                 valloss_epoch=self.val_loss,mnetpath=self.mnetpath)
        print(f'\t History saved after epoch {epoch + 1}!')
        
        modelName = dir_checkpoint + f'MoDL_{str(self.model.in_chans)}_chans_{str(self.model.chans)}_epoch_{str(epoch)}.pt'
        torch.save({'model_state_dict': self.model.state_dict()}, modelName)                
        print(f'\t Checkpoint saved after epoch {epoch + 1}!')        
        
    def validate(self,data,labels,masks,batchsize=5):
        valloss = 0
        n_val = data.shape[0]
        batchnums = int(np.ceil(n_val/valbatchsize))
        self.model.eval()
        with torch.no_grad():
            batchind = 0
            while batchind < batchnums:
                batch = torch.arange(batchind*batchsize,min((batchind+1)*batchsize,n_val))
                databatch  = data[batch,:,:,:].to(device)
                labelbatch = labels[batch,:,:,:].to(device)
                maskbatch  = masks[batch,:,:].to(device)
                smap = torch.ones(labelbatch.shape) # single coil setting, needs to be changed for multi-coil setting
                
                MoDL_res = self.model(databatch,smap,maskbatch)
                
                loss = lpnorm(MoDL_res, labelbatch, mode='mean')
                valloss += loss.item()
                batchind += 1 
            self.val_loss.append(valloss/batchnums)
            print(f' [{epoch+1}/{epochs}] loss/VAL: {valloss/batchnums}')                    
                
    def train(self,traindata,trainlabels,trainmasks,valdata,vallabels,valmasks,\
              epochs=1,\
              lr=1e-4, lr_weight_decay=0,\
              batchsize=3, valbatchsize=5,\
              count_start=(0,0),
              ngpu=0):
        '''
        Assume the data is already shuffled
        '''
        n_train = traindata.shape[0]       
        device  = torch.device('cpu') if ngpu==0 else torch.device('cuda:0')
        global_step = 0
        batchnums = int(np.ceil(n_train/batchsize))       
        optimizer = RMSprop(self.model.parameters(), lr=lr, weight_decay=lr_weight_decay)
        
        self.model.train()
        for epoch in range(count_start[0],epochs):
            trainloss_epoch = 0
            batchind = 0 if epoch!=count_start[0] else count_start[1]
            while batchind < batchnums:
                batch = torch.arange(batchind*batchsize,min((batchind+1)*batchsize,n_train))
                databatch  = traindata[batch,:,:,:].to(device)
                labelbatch = trainlabels[batch,:,:,:].to(device)
                maskbatch  = trainmasks[batch,:,:].to(device)
                smap = torch.ones(labelbatch.shape) # single coil setting, needs to be changed for multi-coil setting
                
                MoDL_out = self.model(databatch,smap,maskbatch)
                
                optimizer.zero_grad()
                loss = lpnorm(MoDL_out, labelbatch)
                loss.backward()
                optimizer.step()
                trainloss_epoch += loss.item()
                self.train_loss.append(loss.item())
                print(f'[{global_step}][{epoch+1}/{epochs}][{batchind+1}/{batchnums}] loss/train: {loss.item()}')
                global_step += 1
                batchind += 1           
            self.train_loss_epoch.append(trainloss_epoch/train_batchnums)  
            print(f'[{global_step}][{epoch+1}/{epochs}] loss/train: {trainloss_epoch/train_batchnums}')          
            self.empty_cache()           
            self.validate(valdata,vallabels,valmasks,batchsize=valbatchsize)
            self.savemodel(epoch)
                
                
                
                
                
                
                
                