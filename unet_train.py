import numpy as np
import argparse
import os
import sys
import random
import torch
import torch.fft as F
from importlib import reload
from torch.nn.functional import relu
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import *
sys.path.insert(0,'/home/huangz78/mri/unet/')
import unet_model
from unet_model import UNet


def mnet_getinput(mnet,data,base=8,budget=32,batchsize=10,unet_channels=1,return_mask=False):
    '''
    assume the input data has the dimension [img,heg,wid]
    '''    
    lowfreqmask = mask_naiveRand(data.shape[1],fix=base,other=0,roll=True)[0]
    heg,wid  = data.shape[1],data.shape[2]
    imgshape =(heg,wid)
    yfull = F.fftn(data,dim=(1,2),norm='ortho')
    y = torch.zeros_like(yfull)
    y[:,lowfreqmask==1,:] = yfull[:,lowfreqmask==1,:]    
    x_ifft = torch.zeros(len(yfull),unet_channels,heg,wid)
    if return_mask:
        masks = torch.zeros(len(yfull),heg)
    
    batchind  = 0
    batchnums = int(np.ceil(data.shape[0]/batchsize))
    while batchind < batchnums:
        batch = torch.arange(batchsize*batchind, min(batchsize*(batchind+1),data.shape[0]))
        yfull_b = yfull[batch,:,:]
        y_b   = y[batch,:,:]
        y_in  = torch.zeros(len(batch),2,heg,wid)
        y_in[:,0,:,:] = torch.real(y_b)
        y_in[:,1,:,:] = torch.imag(y_b)
#         breakpoint()
        mask_b = mnet_wrapper(mnet,y_in,budget,imgshape,normalize=True,detach=True)
        if return_mask:
            masks[batch,:] = mask_b        
        y_mnet_b = torch.zeros_like(yfull_b)
        for ind in range(len(mask_b)):
            y_mnet_b[:,mask_b[ind,:]==1,:] = yfull_b[:,mask_b[ind,:]==1,:]
        if   unet_channels == 1:
            x_ifft[batch,0,:,:] = torch.abs(F.ifftn(y_mnet_b,dim=(1,2),norm='ortho'))
        elif unet_channels == 2:
            x_ifft[batch,:,:,:] = F.ifftn(y_mnet_b,dim=(1,2),norm='ortho')
        batchind += 1
        
    if return_mask:
        return x_ifft, masks
    else:
        return x_ifft
    

def train_net(net, mnet=None,\
              epochs=5,batchsize=5,test_batchsize=10,\
              lr=0.001,lr_weight_decay=1e-8,lr_momentum=0.9,\
              lr_s_stepsize=10,lr_s_gamma=0.8,\
              save_cp=False,datatype=torch.float,\
              base=8,budget=32,\
              histpath=None,
              mode='rand',device='cpu'):
    print('mode = ',mode)
    net.train()
    
    if histpath is None:
        train_loss = list([]); test_loss = list([]); train_loss_epoch = list([])
    else:
        histRec    = np.load(histpath)
        train_loss = list(histRec['trainloss'])
        test_loss  = list(histRec['testloss'])
        train_loss_epoch = list(histRec['trainloss_epoch'])
        print('training history file successfully loaded from the path: ', histpath)
    
    try:
        if mode == 'mnet':
            train_full  = torch.tensor(np.load('/mnt/shared_a/data/fastMRI/knee_singlecoil_train.npz')['data'],dtype=datatype)
            test_full   = torch.tensor(np.load('/mnt/shared_a/data/fastMRI/knee_singlecoil_val.npz')['data'],dtype=datatype)
            train_label = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
            test_label  = torch.reshape(test_full,(test_full.shape[0],1,test_full.shape[1],test_full.shape[2]))
            del train_full,test_full
            
            ## create mnet masks
            
            ## create train_in and test_in
            
        if mode == 'rand':
            ## train a unet to reconstruct images from random mask
#             train_full = torch.tensor(np.load('/home/huangz78/data/traindata_x.npz')['xfull'],dtype=datatype)
#             test_full  = torch.tensor(np.load('/home/huangz78/data/testdata_x.npz')['xfull'],dtype=datatype)
            
            traindata = torch.tensor(np.load('/mnt/shared_a/data/fastMRI/knee_singlecoil_train.npz')['data'],dtype=datatype)
            testdata  = torch.tensor(np.load('/mnt/shared_a/data/fastMRI/knee_singlecoil_val.npz')['data'],dtype=datatype)
            traininds = np.arange(traindata.shape[0])
            testinds  = np.arange(testdata.shape[0])  
#             traininds = np.random.choice(traindata.shape[0],1000,replace=False)
#             testinds  = np.random.choice(testdata.shape[0],100,replace=False)            
            train_full = traindata[traininds,:,:]
            test_full  = testdata[testinds ,:,:]
            for ind in range(train_full.shape[0]):
                train_full[ind,:,:] = train_full[ind,:,:]/train_full[ind,:,:].abs().max()
            for ind in range(test_full.shape[0]):
                test_full[ind,:,:] = test_full[ind,:,:]/test_full[ind,:,:].abs().max()
            
            
            Heg,Wid,n_train,n_test = train_full.shape[1],train_full.shape[2],train_full.shape[0],test_full.shape[0]
            cdatatype = torch.cfloat if datatype==torch.float else torch.cdouble
            
            yfull_train = F.fftn(train_full,dim=(1,2),norm='ortho')
            y_train = torch.zeros_like(yfull_train)
            batchind = 0
            train_batchnums = int(np.ceil(n_train/batchsize))      
            while batchind < train_batchnums:
                batch = torch.arange(batchind*batchsize,min((batchind+1)*batchsize,n_train))
                lfmask = mask_naiveRand(train_full.shape[1],fix=base,other=budget,roll=False)[0] 
                batchdata_full = yfull_train[batch,:,:]
                batchdata      = torch.zeros_like(batchdata_full)
                batchdata[:,lfmask==1,:] = batchdata_full[:,lfmask==1,:]
                y_train[batch,:,:] = batchdata
                batchind += 1
                
            yfull_test = F.fftn(test_full,dim=(1,2),norm='ortho')
            y_test = torch.zeros_like(yfull_test)
            batchind = 0
            test_batchnums = int(np.ceil(n_test/test_batchsize))
            while batchind < test_batchnums:
                t_b = torch.arange(batchind*test_batchsize,min((batchind+1)*test_batchsize,n_test))
                lfmask = mask_naiveRand(test_full.shape[1],fix=base,other=budget,roll=False)[0]
                batchdata_full = yfull_test[t_b,:,:]
                batchdata      = torch.zeros_like(batchdata_full)
                batchdata[:,lfmask==1,:] = batchdata_full[:,lfmask==1,:]           
                y_test[t_b,:,:] = batchdata
                batchind += 1
                
            if net.n_channels == 2:                
                train_ifft = F.ifftn(y_train,dim=(1,2),norm='ortho')
                train_in   = torch.zeros((n_train,2,Heg,Wid),dtype=datatype)
                train_in[:,0,:,:] = torch.real(train_ifft)
                train_in[:,1,:,:] = torch.imag(train_ifft)
                
                test_ifft = F.ifftn(y_test,dim=(1,2),norm='ortho')
                test_in   = torch.zeros((n_test,2,Heg,Wid),dtype=datatype)
                test_in[:,0,:,:] = torch.real(test_ifft)
                test_in[:,1,:,:] = torch.imag(test_ifft)              
            elif net.n_channels == 1:
                train_ifft = torch.abs(F.ifftn(y_train,dim=(1,2),norm='ortho'))                
                test_ifft  = torch.abs(F.ifftn(y_test ,dim=(1,2),norm='ortho'))
    
                train_in = torch.reshape(train_ifft, (n_train,1,Heg,Wid)).to(datatype)
                test_in  = torch.reshape(test_ifft , (n_test ,1,Heg,Wid)).to(datatype)
            
            train_label = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
            test_label  = torch.reshape(test_full,(test_full.shape[0],1,test_full.shape[1],test_full.shape[2]))
            del train_full, test_full
            print('n_train = {}, n_test = {}'.format(n_train,n_test))          
                
        elif mode == 'greedy':
            ## train a unet to reconstruct images from greedy mask
            assert net.n_channels==1
            imgs  = torch.tensor( np.load('/home/huangz78/data/data_gt.npz')['imgdata'] ).permute(2,0,1)
            masks = torch.tensor( np.load('/home/huangz78/data/data_gt_greedymask.npz')['mask'].T ) # labels are already rolled
            xs    = torch.zeros((imgs.shape[0],1,imgs.shape[1],imgs.shape[2]),dtype=torch.float)

            for ind in range(imgs.shape[0]):
                imgs[ind,:,:] = imgs[ind,:,:]/torch.max(torch.abs(imgs[ind,:,:]))
                y = torch.fft.fftshift(F.fftn(imgs[ind,:,:],dim=(0,1),norm='ortho'))
                mask = masks[ind,:]
                ysub = torch.zeros(y.shape,dtype=y.dtype)
                ysub[mask==1,:] = y[mask==1,:]
                xs[ind,0,:,:] = torch.abs(F.ifftn(torch.fft.ifftshift(ysub),dim=(0,1),norm='ortho'))

            imgNum = imgs.shape[0]
            traininds, testinds = train_test_split(np.arange(imgNum),random_state=0,shuffle=True,train_size=round(imgNum*0.8))
            np.savez('/home/huangz78/data/inds_rec.npz',traininds=traininds,testinds=testinds)
            Heg,Wid,n_train,n_test = imgs.shape[1],imgs.shape[2],len(traininds),len(testinds)

            train_full = imgs[traininds,:,:]
            train_label= torch.reshape(train_full,(n_train,1,Heg,Wid))
            testfull   = imgs[testinds,:,:]
            test_label = torch.reshape(testfull,(n_test,1,Heg,Wid))
            train_in   = xs[traininds,:,:,:]
            test_in    = xs[testinds ,:,:,:]
            print('n_train = {}, n_test = {}'.format(n_train,n_test))

        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=lr_weight_decay, momentum=lr_momentum)
        # optimizer = optim.Adam/SGD
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_s_stepsize, gamma=lr_s_gamma)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
        criterion = nn.MSELoss() 

        
        global_step = 0
        train_batchnums = int(np.ceil(n_train/batchsize))
        test_batchnums  = int(np.ceil(n_test/test_batchsize))
        
        ############################################################
        # val before training
        if len(train_loss_epoch)==0:
            testloss = 0
            net.eval()
            with torch.no_grad():
                for ind in range(test_batchnums):                
                    t_b       = torch.arange(ind*test_batchsize,min((ind+1)*test_batchsize,n_test)).to(device)
                    testin    = test_in[t_b,:,:,:].to(device)
                    testlabel = test_label[t_b,:,:,:].to(device)
                    pred      = net(testin).detach()
                    testloss += criterion(pred, testlabel)*len(t_b)
                test_loss.append(testloss.item()/n_test)
            print(f'\n\t[0/{epochs}]  loss/VAL: {testloss.item()/n_test}')
            net.train()
        ############################################################
        acceleration_fold = str(int(train_in.shape[2]/(args.base_freq+args.budget)))
        for epoch in range(epochs):
            epoch_loss = 0        
            batchind   = 0
            while batchind < train_batchnums:        
                batch = torch.arange(batchind*batchsize,min((batchind+1)*batchsize,n_train)).to(device)
                imgbatch   = train_in[batch,:,:,:].to(device)
                labelbatch = train_label[batch,:,:,:].to(device)

                pred = net(imgbatch)
                loss = criterion(pred, labelbatch)
                train_loss.append(loss.item())
                epoch_loss += loss.item()*len(batch)
                print(f'[{global_step+1}][{epoch+1}/{epochs}][{batchind}/{train_batchnums}]  loss/train: {loss.item()}')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                
                batchind    += 1
                global_step += 1
            
            train_loss_epoch.append(epoch_loss/n_train)
            
            testloss = 0
            net.eval()
            with torch.no_grad():
                for ind in range(test_batchnums):                
                    t_b       = torch.arange(ind*test_batchsize,min((ind+1)*test_batchsize,n_test)).to(device)
                    testin    = test_in[t_b,:,:,:].to(device)
                    testlabel = test_label[t_b,:,:,:].to(device)
                    pred      = net(testin).detach()
                    testloss += criterion(pred, testlabel)*len(t_b)
                test_loss.append(testloss.item()/n_test)
            print(f'\n\t[{epoch+1}/{epochs}]  loss/VAL: {testloss.item()/n_test}')
            net.train()
            scheduler.step()
            
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                np.savez('/home/huangz78/checkpoints/TrainRec_unet_'+ str(net.n_channels) + '_' +str(net.skip) +'_'+ acceleration_fold +'f' + mode+'.npz',trainloss=train_loss,testloss=test_loss,trainloss_epoch=train_loss_epoch)
                torch.save({'model_state_dict': net.state_dict()}, dir_checkpoint + 'unet_' + str(net.n_channels) + '_' + str(net.skip) +'_' + acceleration_fold +'f' + mode +'.pt')
    #                         }, dir_checkpoint + f'CP_epoch{epoch + 1}.pt')
                print(f'\t Checkpoint saved after epoch {epoch + 1}!')
                
    except KeyboardInterrupt:
        print('Keyboard Interrupted! Exit~')
        dir_checkpoint = '/home/huangz78/checkpoints/'
        torch.save({'model_state_dict': net.state_dict()}, dir_checkpoint + 'unet_' + str(net.n_channels) + '_' + str(net.skip) + '_' + acceleration_fold +'f' + mode +'.pt')
        print(f'\t Checkpoint saved at Python epoch {epoch}, batchnum {batchind}!')
        np.savez('/home/huangz78/checkpoints/TrainRec_unet_'+ str(net.n_channels) + '_' +str(net.skip)+'_'+ acceleration_fold +'f' + mode+'.npz',trainloss=train_loss,testloss=test_loss,trainloss_epoch=train_loss_epoch)
        print('Model is saved after interrupt~')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)   
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=30,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-m','--mode',metavar='M',type=str,nargs='?',default='rand',
                        help='training mode', dest='mode')
    parser.add_argument('-cn', '--channel-num', metavar='CN', type=int, nargs='?', default=1,
                        help='channel number of unet', dest='n_channels')
    parser.add_argument('-s','--skip',type=str,default='False',
                        help='residual network application', dest='skip')
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=8,
                        help='number of observed low frequencies', dest='base_freq')
    parser.add_argument('-bg','--budget',metavar='BG',type=int,nargs='?',default=32,
                        help='number of high frequencies to sample', dest='budget')
    
    parser.add_argument('-up', '--unet-path', type=str, default=None,
                        help='path file for a unet', dest='unetpath')
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
    
    unet = UNet(n_channels=args.n_channels,n_classes=1,bilinear=(not args.skip),skip=args.skip).to(device)
    if args.unetpath is not None:
        checkpoint = torch.load(args.unetpath)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print('Unet loaded successfully from: ' + args.unetpath )
    train_net(unet,epochs=args.epochs,batchsize=args.batchsize,\
              lr=args.lr,lr_weight_decay=0,lr_momentum=0,\
              base=args.base_freq,budget=args.budget,\
              save_cp=True,mode=args.mode,histpath=args.histpath,\
              device=device)