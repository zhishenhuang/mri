import numpy as np
import argparse
import os
import sys
import random
import torch
import torch.fft as FFT
from importlib import reload
from torch.nn.functional import relu
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import *
sys.path.insert(0,'/home/leo/mri/unet/')
# import unet_model
from unet.unet_model import UNet
from unet.unet_model_fbr import Unet
from unet.unet_model_banding_removal_fbr import UnetModel
from mnet import MNet
import copy
dir_checkpoint = '/mnt/DataA/checkpoints/leo/'

def lpnorm(x,xstar,p='fro'):
    '''
    x and xstar are both assumed to be in the format NCHW
    '''
    assert(x.shape==xstar.shape)
    numerator   = torch.norm(x-xstar,p=p,dim=(2,3))
    denominator = torch.norm(xstar  ,p=p,dim=(2,3))
    error = torch.sum( torch.div(numerator,denominator) )
    return error


def mnet_getinput(mnet,data,base=8,budget=32,batchsize=10,unet_channels=1,return_mask=False,device='cpu'):
    '''
    assume the input data has the dimension [img,heg,wid]
    returned data in the format [NCHW]
    '''   
    mnet.eval()
    mnet.to(device)
    with torch.no_grad():
        lowfreqmask = mask_naiveRand(data.shape[1],fix=base,other=0,roll=False)[0]
        heg,wid  = data.shape[1],data.shape[2]
        imgshape = (heg,wid)
        yfull = FFT.fftn(data,dim=(1,2),norm='ortho')
        y = torch.zeros_like(yfull)
        y[:,lowfreqmask==1,:] = yfull[:,lowfreqmask==1,:]    
        x_ifft = torch.zeros(len(yfull),unet_channels,heg,wid,device='cpu')
        if return_mask:
            masks = torch.zeros(len(yfull),heg,device=device)

        batchind  = 0
        batchnums = int(np.ceil(data.shape[0]/batchsize))
        while batchind < batchnums:
            batch = torch.arange(batchsize*batchind, min(batchsize*(batchind+1),data.shape[0]))
            yfull_b = yfull[batch,:,:].to(device)
            y_lf    = y[batch,:,:].to(device)
            y_in    = torch.zeros(len(batch),2,heg,wid,device=device)
            y_in[:,0,:,:] = torch.real(y_lf)
            y_in[:,1,:,:] = torch.imag(y_lf)
            y_in   = FFT.fftshift(y_in,dim=(2,3))
            mask_b = FFT.ifftshift(mnet_wrapper(mnet,y_in,budget,imgshape,normalize=True,detach=True,device=device),dim=(1))

            if return_mask:
                masks[batch,:] = mask_b        
            y_mnet_b = torch.zeros_like(yfull_b,device=device)
            for ind in range(len(mask_b)):
                y_mnet_b[:,mask_b[ind,:]==1,:] = yfull_b[:,mask_b[ind,:]==1,:]
            if   unet_channels == 1:
                x_ifft[batch,0,:,:] = torch.abs(FFT.ifftn(y_mnet_b,dim=(1,2),norm='ortho')).cpu()
            elif unet_channels == 2:
                x_ifft_c = FFT.ifftn(y_mnet_b,dim=(1,2),norm='ortho')
                x_ifft[batch,0,:,:] = torch.real(x_ifft_c).cpu()
                x_ifft[batch,1,:,:] = torch.imag(x_ifft_c).cpu()
            batchind += 1
        
    if return_mask:
        return x_ifft, masks
    else:
        return x_ifft
    
def rand_getinput(data,base=8,budget=32,batchsize=5,datatype=torch.float):
    '''
    assume the input data has the dimension [img,heg,wid]
    returned data in the format [NCHW]
    '''   
    yfull = FFT.fftn(data,dim=(1,2),norm='ortho')
    y_lf  = torch.zeros_like(yfull)
    num_pts,heg,wid = data.shape[0],data.shape[1],data.shape[2]
    batchind = 0
    batchnums = int(np.ceil(num_pts/batchsize))      
    while batchind < batchnums:
        batch = torch.arange(batchind*batchsize,min((batchind+1)*batchsize,num_pts))
        lfmask = mask_naiveRand(data.shape[1],fix=base,other=budget,roll=False)[0] 
        batchdata_full = yfull[batch,:,:]
        batchdata      = torch.zeros_like(batchdata_full)
        batchdata[:,lfmask==1,:] = batchdata_full[:,lfmask==1,:]
        y_lf[batch,:,:] = batchdata
        batchind += 1
    
    if net.in_chans == 2:                
        x_ifft = FFT.ifftn(y_lf,dim=(1,2),norm='ortho')
        x_in   = torch.zeros((num_pts,2,heg,wid),dtype=datatype)
        x_in[:,0,:,:] = torch.real(x_ifft)
        x_in[:,1,:,:] = torch.imag(x_ifft)       
    elif net.in_chans == 1:
        x_ifft = torch.abs(FFT.ifftn(y_lf,dim=(1,2),norm='ortho'))                
        x_in   = torch.reshape(x_ifft, (num_pts,1,heg,wid)).to(datatype)
    
    return x_in
    

def train_net(net,\
              epochs=5,batchsize=5,test_batchsize=10,\
              lr=0.001, lr_weight_decay=1e-8, \
              lr_s_stepsize=40, lr_s_gamma=0.1, patience=5, min_lr=5e-6,reduce_factor=.8,\
              save_cp=False, datatype=torch.float,\
              mnet=None, base=8, budget=32,\
              histpath=None,mnetpath=None,\
              count_start=(0,0),\
              p=1,\
              mode='rand',\
              device='cpu'):
    print('mode = ',mode)
    net.to(device)
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
            train_full  = torch.tensor(np.load('/mnt/DataA/knee_singlecoil_train.npz')['data'],dtype=datatype)
            test_full   = torch.tensor(np.load('/mnt/DataA/knee_singlecoil_val.npz')['data'],  dtype=datatype)
            
            for ind in range(train_full.shape[0]):
                train_full[ind,:,:] = train_full[ind,:,:]/train_full[ind,:,:].abs().max()
            for ind in range(test_full.shape[0]):
                test_full[ind,:,:]  = test_full[ind,:,:]/test_full[ind,:,:].abs().max()
            
            shuffle_inds = torch.randperm(train_full.shape[0])
            train_full   = train_full[shuffle_inds,:,:]
            
            shuffle_inds = torch.randperm(test_full.shape[0])
            test_full    = test_full[shuffle_inds,:,:]
            
            train_label  = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
            test_label   = torch.reshape(test_full,(test_full.shape[0],1,test_full.shape[1],test_full.shape[2]))
            
            ## create train_in and test_in
            train_in = mnet_getinput(mnet,train_full,base=base,budget=budget,batchsize=batchsize,unet_channels=net.in_chans,return_mask=False,device=device)
            del train_full
            test_in = mnet_getinput(mnet,test_full,base=base,budget=budget,batchsize=batchsize,unet_channels=net.in_chans,return_mask=False,device=device)
            del test_full, mnet
            print('\n   Data successfully prepared with the provided MNet!\n')
            
        if mode == 'rand':
            ## train a unet to reconstruct images from random mask
#             train_full = torch.tensor(np.load('/home/huangz78/data/traindata_x.npz')['xfull'],dtype=datatype)
#             test_full  = torch.tensor(np.load('/home/huangz78/data/testdata_x.npz')['xfull'] ,dtype=datatype)         
            train_full = torch.tensor(np.load('/mnt/shared_a/data/fastMRI/knee_singlecoil_train.npz')['data'],dtype=datatype)
            test_full  = torch.tensor(np.load('/mnt/shared_a/data/fastMRI/knee_singlecoil_val.npz')['data']  ,dtype=datatype)
            for ind in range(train_full.shape[0]):
                train_full[ind,:,:] = train_full[ind,:,:]/train_full[ind,:,:].abs().max()
            for ind in range(test_full.shape[0]):
                test_full[ind,:,:] = test_full[ind,:,:]/test_full[ind,:,:].abs().max()        
            
            train_in = rand_getinput(train_full,base=base,budget=budget,batchsize=batchsize,datatype=datatype)
            test_in  = rand_getinput(test_full ,base=base,budget=budget,batchsize=batchsize,datatype=datatype)
                   
            train_label = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
            test_label  = torch.reshape(test_full,(test_full.shape[0],1,test_full.shape[1],test_full.shape[2]))
            del train_full, test_full         
                
        elif mode == 'greedy':
            ## train a unet to reconstruct images from greedy mask
            assert net.in_chans==1
            imgs  = torch.tensor( np.load('/home/huangz78/data/data_gt.npz')['imgdata'] ).permute(2,0,1)
            masks = torch.tensor( np.load('/home/huangz78/data/data_gt_greedymask.npz')['mask'].T ) # labels are already rolled
            xs    = torch.zeros((imgs.shape[0],1,imgs.shape[1],imgs.shape[2]),dtype=torch.float)

            for ind in range(imgs.shape[0]):
                imgs[ind,:,:] = imgs[ind,:,:]/torch.max(torch.abs(imgs[ind,:,:]))
                y = FFT.fftshift(FFT.fftn(imgs[ind,:,:],dim=(0,1),norm='ortho'))
                mask = masks[ind,:]
                ysub = torch.zeros(y.shape,dtype=y.dtype)
                ysub[mask==1,:] = y[mask==1,:]
                xs[ind,0,:,:] = torch.abs(FFT.ifftn(torch.fft.ifftshift(ysub),dim=(0,1),norm='ortho'))

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
        
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=lr_weight_decay)
#         optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=lr_weight_decay)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_s_stepsize, gamma=lr_s_gamma)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, verbose=True, min_lr=min_lr,factor=reduce_factor)
#         criterion = nn.MSELoss() 

        n_train = train_in.shape[0]
        n_test  = test_in.shape[0]
        
        global_step = 0
        train_batchnums = int(np.ceil(n_train/batchsize))
        test_batchnums  = int(np.ceil(n_test/test_batchsize))
        testloss_old = np.inf
        ############################################################
        # val before training
        if len(train_loss_epoch)==0:
            testloss = 0
            net.eval()
            with torch.no_grad():
                for ind in range(test_batchnums):                
                    t_b       = torch.arange(ind*test_batchsize,min((ind+1)*test_batchsize,n_test))
                    testin    = test_in[t_b,:,:,:].to(device)
                    testlabel = test_label[t_b,:,:,:].to(device)
                    pred      = net(testin).detach()
#                     testloss += criterion(pred, testlabel)*len(t_b)
                    testloss += lpnorm(pred, testlabel,p=p)
                test_loss.append(testloss.item()/n_test)
                if testloss.item()/n_test < testloss_old:
                    testloss_old = testloss.item()/n_test
            print(f'\n\t[0/{epochs}]  loss/VAL: {testloss.item()/n_test}')
            torch.cuda.empty_cache()
            del testin,testlabel,pred
            net.train()
        ############################################################
        acceleration_fold = str(int(train_in.shape[2]/(args.base_freq+args.budget)))
        
        for epoch in range(count_start[0],epochs):
            epoch_loss = 0
            batchind   = 0 if epoch!=count_start[0] else count_start[1]
            while batchind < train_batchnums:        
                batch = torch.arange(batchind*batchsize,min((batchind+1)*batchsize,n_train))
                imgbatch   = train_in[batch,:,:,:].to(device)
                labelbatch = train_label[batch,:,:,:].to(device)

                pred = net(imgbatch)
#                 loss = criterion(pred, labelbatch)                
#                 train_loss.append(loss.item())
#                 epoch_loss += loss.item()*len(batch)
                loss = lpnorm(pred,labelbatch,p=p)
                train_loss.append(loss.item()/len(batch))
                epoch_loss += loss.item()
                print(f'[{global_step+1}][{epoch+1}/{epochs}][{batchind}/{train_batchnums}]  loss/train: {loss.item()/len(batch)}')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                torch.cuda.empty_cache()
                del imgbatch, labelbatch, pred
                batchind    += 1
                global_step += 1
                
            train_loss_epoch.append(epoch_loss/n_train)
            
            testloss = 0
            net.eval()
            with torch.no_grad():
                for ind in range(test_batchnums):                
                    t_b       = torch.arange(ind*test_batchsize,min((ind+1)*test_batchsize,n_test))
                    testin    = test_in[t_b,:,:,:].to(device)
                    testlabel = test_label[t_b,:,:,:].to(device)
                    pred      = net(testin).detach()
#                     testloss += criterion(pred, testlabel)*len(t_b)
                    testloss += lpnorm(pred, testlabel,p=p)
                testloss_epoch = testloss.item()/n_test
                test_loss.append(testloss_epoch)
                if testloss_epoch < testloss_old:
                    testloss_old = copy.deepcopy(testloss_epoch)
                    save_flag = True
                else:
                    save_flag = False
                torch.cuda.empty_cache()
                del testin, testlabel, pred
                    
            print(f'\n\t[{epoch+1}/{epochs}]  loss/VAL: {testloss_epoch}')
            net.train()
#             scheduler.step()
            scheduler.step(testloss_epoch)
            
            if save_cp :                
#                 recName   = dir_checkpoint + 'TrainRec_unet_'+ str(net.in_chans) + '_' +str(net.skip) +'_'+ acceleration_fold +'f' + mode + '_epoch_' + str(epoch) + '.npz' 
                recName = dir_checkpoint + f'TrainRec_unet_fbr_{str(net.in_chans)}_chans_{str(net.chans)}_epoch_{str(epoch)}.npz'
                np.savez(recName,trainloss=train_loss,testloss=test_loss,trainloss_epoch=train_loss_epoch,mnetpath=mnetpath)
                print(f'\t History saved after epoch {epoch + 1}!')
                if save_flag:
#                     modelName = dir_checkpoint + 'unet_' + str(net.in_chans) + '_' + str(net.skip) +'_' + acceleration_fold +'f' + mode + '_epoch_' + str(epoch) +'.pt'
                    modelName = dir_checkpoint + f'unet_fbr_{str(net.in_chans)}_chans_{str(net.chans)}_epoch_{str(epoch)}.pt'
                    torch.save({'model_state_dict': net.state_dict()}, modelName)                
                    print(f'\t Checkpoint saved after epoch {epoch + 1}!')
                save_flag = False
            torch.cuda.empty_cache()   
    except KeyboardInterrupt:
        print('Keyboard Interrupted! Exit~')
#         modelName = dir_checkpoint + 'unet_' + str(net.in_chans) + '_' + str(net.skip) +'_' + acceleration_fold +'f' + mode + '.pt'
#         recName   = dir_checkpoint + 'TrainRec_unet_'+ str(net.in_chans) + '_' + str(net.skip) +'_'+ acceleration_fold +'f' + mode +'.npz'
        modelName = dir_checkpoint + f'unet_fbr_{str(net.in_chans)}_chans_{str(net.chans)}_epoch_{str(epoch)}.pt'
        recName = dir_checkpoint + f'TrainRec_unet_fbr_{str(net.in_chans)}_chans_{str(net.chans)}.npz'
        np.savez(recName,trainloss=train_loss,testloss=test_loss,trainloss_epoch=train_loss_epoch,mnetpath=mnetpath)
        torch.save({'model_state_dict': net.state_dict()}, modelName)
        print(f'\t Checkpoint saved at Python epoch {epoch}, batchnum {batchind}!')
        print('Model is saved after interrupt~')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)   
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-tb', '--test-batch-size', metavar='TB', type=int, nargs='?', default=5,
                        help='Testbatch size', dest='test_batchsize')
    
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
    
    parser.add_argument('-cn', '--channel-num', metavar='CN', type=int, nargs='?', default=128,
                        help='channel number of unet', dest='chans')
    parser.add_argument('-uc', '--uchan-in', metavar='UC', type=int, nargs='?', default=2,
                        help='number of input channel of unet', dest='in_chans')
    parser.add_argument('-s','--skip',type=str,default='False',
                        help='residual network application', dest='skip')
    
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=8,
                        help='number of observed low frequencies', dest='base_freq')
    parser.add_argument('-bg','--budget',metavar='BG',type=int,nargs='?',default=32,
                        help='number of high frequencies to sample', dest='budget')
    
    parser.add_argument('-mp', '--mnet-path', type=str, default='/mnt/DataA/checkpoints/leo/mri/mnet_split_trained_cf_8_bg_32_unet_in_chan_1_epoch_9.pt',
                        help='path file for a mnet', dest='mnetpath')
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
        mnet = MNet(beta=1,in_channels=2,out_size=320-args.base_freq, imgsize=(320,320),poolk=3)
        checkpoint = torch.load(args.mnetpath)
        mnet.load_state_dict(checkpoint['model_state_dict'])
        print('MNet loaded successfully from: ' + args.mnetpath)
        mnet.eval()
    else:
        mnet = None
    
    train_net(unet,\
              epochs=args.epochs,batchsize=args.batchsize,test_batchsize=args.test_batchsize,\
              lr=args.lr, lr_weight_decay=args.lrwd,\
              lr_s_stepsize=40, lr_s_gamma=0.8, patience=5, min_lr=1e-6,reduce_factor=.8,\
              count_start=(args.epoch_start,args.batchind_start),\
              p='fro',\
              mnet=mnet,\
              base=args.base_freq,budget=args.budget,\
              save_cp=True,mode=args.mode,histpath=args.histpath,mnetpath=args.mnetpath,\
              device=device)