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

sys.path.insert(0,'/home/huangz78/mri/unet/')
import unet_model
from unet_model import UNet

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

def loupeTrain(traindata,valdata,\
               slope=5, sparsity=.25, preselect=True, preselect_num=24,\
               unet_skip=True, n_channels=1,\
               lrm=1e-3, lru=1e-4, weight_decay=0, momentum=0,\
               epochs=1, batchsize_train=5, batchsize_val=2, count_start=(0,0),\
               modelpath=None,histpath=None,\
               save_cp=True):
    
    '''
    train and test data are assumed to have the shape [size, heg, wid]
    '''
    
    # load sampler
    shape   = traindata.shape[1:3]
    
    sampler = None
    # load unet
    UNET = UNet(n_channels=n_channels,n_classes=1,bilinear=(not unet_skip),skip=unet_skip)
    unetpath = '/home/huangz78/checkpoints/unet_1_' + str(unet_skip) + '.pth'
    checkpoint = torch.load(unetpath)
    UNET.load_state_dict(checkpoint['model_state_dict'])

    loupe = LOUPE(n_channels=n_channels,unet_skip=unet_skip,shape=shape,slope=slope,sparsity=sparsity,\
                  preselect=preselect,preselect_num=preselect_num,\
                  sampler=sampler,unet=UNET)
    if modelpath is not None:
        checkpoint = torch.load(modelpath)
        loupe.load_state_dict(checkpoint['model_state_dict'])
        print('loupe model successfully loaded from the path: ', modelpath)
    loupe.train()
    
    # training
    if histpath is None:
        train_loss = []; val_loss = []
    else:
        histRec    = np.load(histpath)
        train_loss = list(histRec['loss_train'])
        val_loss   = list(histRec['loss_val'])
        print('training history file successfully loaded from the path: ', histpath)
    epoch_count = count_start[0];  batchind = count_start[1]
    batch_nums  = int(np.ceil(traindata.shape[0]/batchsize_train))
    
    optimizer = optim.RMSprop([
                    {'params': loupe.samplers.parameters()},
                    {'params': loupe.unet.parameters(),'lr':lru}
                ], lr=lrm, weight_decay=weight_decay, momentum=momentum,eps=1e-10)
    criterion = nn.L1Loss()
    try:
        while epoch_count<epochs:
            while batchind<batch_nums:
                batch = np.arange(batchsize_train*batchind, min(batchsize_train*(batchind+1),traindata.shape[0]))
                databatch = normalize_data(traindata[batch,:,:]) if len(batch)>1 else normalize_data(traindata[batch,:,:].view(-1,shape[0],shape[1]))
                xstar = traindata[batch,:,:].view(batch.shape[0],-1,shape[0],shape[1])
                ystar = F.fftn(xstar,dim=(2,3),norm='ortho')
                x_recon,_ = loupe(ystar)

                loss_train = criterion(x_recon,xstar)
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                progress_str = f'[{epoch_count+1}/{epochs}][{min(batchsize_train*(batchind+1),traindata.shape[0])}/{traindata.shape[0]}]'
                print(progress_str + f' training loss: {loss_train.item()}')
                train_loss.append(loss_train.item())
                batchind += 1   

            # validation eval
            valbatchind = 0
            valbatch_nums = int(np.ceil(valdata.shape[0]/batchsize_val))
            loupe.eval()
            loss_val = 0
            while (valbatchind < valbatch_nums):
                batch = np.arange(batchsize_val*valbatchind, min(batchsize_val*(valbatchind+1),valdata.shape[0]))
                databatch = normalize_data(valdata[batch,:,:]) if len(batch)>1 else normalize_data(valdata[batch,:,:].view(-1,shape[0],shape[1]))
                xstar = databatch.view(batch.shape[0],-1,shape[0],shape[1])
                ystar = F.fftn(xstar,dim=(2,3),norm='ortho')
                x_recon,_ = loupe(ystar)
                x_recon = x_recon.detach()
                loss_val += criterion(x_recon,xstar)
                valbatchind += 1
            progress_str = f'[{epoch_count+1}/{epochs}]'
            print('\n' + progress_str + f' validation loss: {loss_val.item()/valbatch_nums}')
            val_loss.append(loss_val.item()/valbatch_nums)
            loupe.train()

            # saving models
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                try:
                    os.mkdir(dir_checkpoint)
                    print('Created checkpoint directory')
                except OSError:
                    pass
                torch.save({'model_state_dict': loupe.state_dict()}, dir_checkpoint + 'loupe_model.pt')
                np.savez(dir_checkpoint+'loupe_history.npz',loss_train=train_loss, loss_val=val_loss)
                print(f'\t Checkpoint for Loupe saved after epoch {epoch_count + 1}!' + '\n')
            epoch_count += 1
            batchind = 0
    except KeyboardInterrupt: # need debug
        print('Keyboard Interrupted! Exit ~')
        torch.save({'model_state_dict': loupe.state_dict()}, dir_checkpoint + 'loupe_model.pt')
        np.savez(dir_checkpoint+'loupe_history.npz',loss_train=train_loss, loss_val=val_loss)
        print('Model is saved after keyboard interruption~')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            
def get_args():
    parser = argparse.ArgumentParser(description='Train the Loupe model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-bt', '--batch-size-train', metavar='BT', type=int, nargs='?', default=5,
                        help='Batch size train', dest='batchsize_train')
    parser.add_argument('-bv', '--batch-size-val', metavar='BV', type=int, nargs='?', default=5,
                        help='Batch size val', dest='batchsize_val')
    parser.add_argument('-lrm', '--learning-rate-mask', metavar='LRM', type=float, nargs='?', default=5e-4,
                        help='Learning rate mask', dest='lrm')
    parser.add_argument('-lru', '--learning-rate-unet', metavar='LRU', type=float, nargs='?', default=1e-4,
                        help='Learning rate unet', dest='lru')
    parser.add_argument('-skip', '--unet-skip', type=str, default='True',
                        help='skip connections in structure of unet', dest='unet_skip')
    parser.add_argument('-s', '--sparsity', type=float, default=.25,
                        help='sparsity', dest='sparsity')
    parser.add_argument('-core', '--preselect-num', type=int, default=24,
                        help='preselected number of low frequencies', dest='preselect_num')
    parser.add_argument('-sl', '--slope', type=float, default=5,
                        help='slope used in the sigmoid function', dest='slope')
    parser.add_argument('-mp', '--model-path', type=str, default=None,
                        help='path file for a loupe model', dest='modelpath')
    parser.add_argument('-es','--epoch-start',metavar='ES',type=int,nargs='?',default=0,
                        help='starting epoch count',dest='epoch_start')
    parser.add_argument('-bis','--batchind-start',metavar='BIS',type=int,nargs='?',default=0,
                        help='starting batchind',dest='batchind_start')
    parser.add_argument('-hp', '--history-path', type=str, default=None,
                        help='path file for npz file recording training history', dest='histpath')
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
    
    traindata  = np.load('/home/huangz78/data/traindata_x.npz')
    dtyp       = torch.float
    trainxfull = torch.tensor(traindata['xfull'],dtype=dtyp)
    testdata   = np.load('/home/huangz78/data/testdata_x.npz')
    testxfull  = torch.tensor(testdata['xfull'],dtype=dtyp)

    preselect  = True if args.preselect_num > 0 else False
   
    loupeTrain(trainxfull,testxfull[0:20],\
               slope=args.slope, sparsity=args.sparsity, preselect=preselect, preselect_num=args.preselect_num,\
               unet_skip=args.unet_skip, n_channels=1,\
               lrm=args.lrm, lru=args.lru, weight_decay=0, momentum=0,\
               epochs=args.epochs, batchsize_train=args.batchsize_train, batchsize_val=args.batchsize_val, count_start=(args.epoch_start,args.batchind_start),\
               modelpath=args.modelpath,histpath=args.histpath,\
               save_cp=args.save_cp)
