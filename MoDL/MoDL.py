import sys
import torch
import torch.nn as nn
import numpy as np
import random
import copy
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import RMSprop, Adam

from unet.unet_model_fbr import Unet
import MoDL.utils_MoDL as utils

from utils import lpnorm

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
                 CG_steps: int = 6,
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
        data input: format NCHW, with C=2, be image space data
        smap input: format [N,num_coils,C,H,W]
        mask input: format NCHW
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
                 save_dir: Path,
                 epochs=1,
                 lr=1e-4, 
                 lr_weight_decay=0,
                 batchsize=3, 
                 valbatchsize=5,
                 count_start=(0,0),
                 ngpu=0,
                 hist_dir:str=None,
                 mnetpath:str=None): 
        
        self.model = model
        self.save_dir = save_dir
        
        self.epochs = epochs
        self.lr = lr
        self.lr_weight_decay = lr_weight_decay
        self.batchsize = batchsize
        self.valbatchsize = valbatchsize
        self.count_start = count_start
        self.device = torch.device('cuda:0') if ((torch.cuda.is_available()) and (ngpu>0)) else torch.device('cpu')
        
        if hist_dir is None:
            self.val_loss         = []
            self.train_loss_epoch = []
            self.train_loss       = []
        else:
            histRec = np.load(hist_dir)
            self.val_loss         = list(histRec[valloss_epoch])
            self.train_loss_epoch = list(histRec[trainloss_epoch])
            self.train_loss       = list(histRec[trainloss])
    
    def empty_cache(self):
        torch.cuda.empty_cache()
        torch.backends.cuda.cufft_plan_cache.clear()
    
    def save_model(self,epoch=0):
        recName = self.save_dir + f'Hist_MoDL_{str(self.model.in_chans)}_chans_{str(self.model.chans)}_epoch_{str(epoch)}.npz'
        np.savez(recName,\
                 trainloss=self.train_loss,trainloss_epoch=self.train_loss_epoch,\
                 valloss_epoch=self.val_loss,mnetpath=self.mnetpath)
        print(f'\t History saved after epoch {epoch + 1}!')
        
        modelName = self.save_dir + f'MoDL_{str(self.model.in_chans)}_chans_{str(self.model.chans)}_epoch_{str(epoch)}.pt'
        torch.save({'model_state_dict': self.model.state_dict()}, modelName)                
        print(f'\t Checkpoint saved after epoch {epoch + 1}!')        
        
    def validate(self,data,labels,masks,epoch=-1):
        valloss = 0
        n_val = data.shape[0]
        batchnums = int(np.ceil(n_val/self.valbatchsize))
        self.model.eval()
        with torch.no_grad():
            batchind = 0
            while batchind < batchnums:
                batch      = torch.arange(batchind*self.valbatchsize,min((batchind+1)*self.valbatchsize,n_val))
                databatch  = data[batch,:,:,:].to(self.device)
                labelbatch = labels[batch,:,:,:].to(self.device)
                maskbatch  = masks[batch,:,:].to(self.device)
                smap       = torch.ones(databatch.shape,device=self.device).unsqueeze(1) # single coil setting, needs to be changed for multi-coil setting
                
                MoDL_res = self.model(databatch,smap,maskbatch)
                loss = lpnorm(MoDL_res[:,0:1,:,:], labelbatch, mode='mean')
                valloss += loss.item()
                batchind += 1 
            self.val_loss.append(valloss/batchnums)
            print(f' [{epoch+1}/{self.epochs}] loss/VAL: {valloss/batchnums:.4f}')                    
                
    def run(self,traindata,trainlabels,trainmasks,valdata,vallabels,valmasks,save_cp=True):
        '''
        Assume the data is already shuffled
        '''
        
        if len(trainmasks.shape) == 2: # input mask is in the format of NH
            trainmasks_in = copy.deepcopy(trainmasks)
            trainmasks    = torch.zeros(trainmasks.shape[0],traindata.shape[2],traindata.shape[3])
            for ind in range(len(trainmasks)):
                mask_tmp = trainmasks_in[ind].unsqueeze(1).repeat(1,traindata.shape[3])
                trainmasks[ind] = mask_tmp
            trainmasks = trainmasks.unsqueeze(1).repeat(1,2,1,1)
        
        if len(valmasks.shape) == 2: # input mask is in the format of NH
            valmasks_in = copy.deepcopy(valmasks)
            valmasks    = torch.zeros(valmasks_in.shape[0],valdata.shape[2],valdata.shape[3])
            for ind in range(len(valmasks)):
                mask_tmp = valmasks_in[ind].unsqueeze(1).repeat(1,valdata.shape[3])
                valmasks[ind] = mask_tmp
            valmasks = valmasks.unsqueeze(1).repeat(1,2,1,1)
        
        n_train = traindata.shape[0]       
        global_step = 0
        batchnums = int(np.ceil(n_train/self.batchsize))       
        optimizer = RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.lr_weight_decay)
        
        self.model.train()
        self.validate(valdata,vallabels,valmasks)
        for epoch in range(self.count_start[0],self.epochs):
            trainloss_epoch = 0
            batchind = 0 if epoch!=self.count_start[0] else self.count_start[1]
            while batchind < batchnums:
                batch          = torch.arange(batchind*self.batchsize,min((batchind+1)*self.batchsize,n_train))
                databatch      = traindata[batch,:,:,:].to(self.device)
                labelbatch     = trainlabels[batch,:,:,:].to(self.device)
                label          = torch.zeros(batch.shape[0],2,databatch.shape[2],databatch.shape[3],device=self.device)
                label[:,0,:,:] = labelbatch[:,0,:,:]
                maskbatch      = trainmasks[batch].to(self.device)
                smap           = torch.ones(databatch.shape,device=self.device).unsqueeze(1) # single coil setting, needs to be changed for multi-coil setting
                MoDL_out = self.model(databatch,smap,maskbatch)
                
                optimizer.zero_grad()
                loss = lpnorm(MoDL_out, label, mode='no_normalization')
                loss.backward()
                optimizer.step()
                trainloss_epoch += loss.item()
                self.train_loss.append(loss.item())
                with torch.no_grad():
                    l2loss_batch = lpnorm(MoDL_out[:,0:1,:,:],labelbatch,mode='mean')
                print(f'[{global_step}][{epoch+1}/{self.epochs}][{batchind+1}/{batchnums}] loss/train: {loss.item():.4f}, l2loss: {l2loss_batch.item():.4f}')
                global_step += 1
                batchind += 1           
            self.train_loss_epoch.append(trainloss_epoch/train_batchnums)  
            print(f'[{global_step}][{epoch+1}/{self.epochs}] loss/train: {trainloss_epoch/train_batchnums}')          
            self.empty_cache()           
            self.validate(valdata,vallabels,valmasks,epoch=epoch)
            if save_cp:
                self.savemodel(epoch=epoch)
                
                
                
                
                
                
                
                