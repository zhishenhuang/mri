import numpy as np
import argparse
import os
import sys
import torch
import torch.fft as F
from importlib import reload
from torch.nn.functional import relu
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from sklearn.model_selection import train_test_split

sys.path.insert(0,'/home/huangz78/mri/unet/')
import unet_model_orig
from unet_model_orig import UNet
# import unet_model
# from unet_model import UNet

def train_net(net, train_in, test_full, \
              epochs=5,batch_size=5,\
              lr=0.001,lr_weight_decay=1e-8,lr_momentum=0.9,\
              lr_s_stepsize=10,lr_s_gamma=0.8,\
              save_cp=False,datatype=torch.float,mode='rand',t_bs=10.0):
    print('mode = ',mode)
    net.train()
    try:
        if mode == 'rand':
            train_full   = torch.tensor(np.load('/home/huangz78/data/traindata_x.npz')['xfull'],dtype=datatype)
            train_label  = torch.reshape(train_full,(train_full.shape[0],1,train_full.shape[1],train_full.shape[2]))
            test_full    = torch.tensor(np.load('/home/huangz78/data/testdata_x.npz')['xfull'],dtype=datatype)
            test_label   = torch.reshape(test_full,(test_full.shape[0],1,test_full.shape[1],test_full.shape[2]))
            if net.n_channels == 2:
                cdatatype = torch.cfloat if datatype==torch.float else torch.cdouble
                train_dir = '/home/huangz78/data/traindata_y.npz'
                train_sub  = torch.tensor(np.load(train_dir)['y'],dtype=cdatatype)                
                test_dir  = '/home/huangz78/data/testdata_y.npz'
                test_sub   = torch.tensor(np.load(test_dir)['y'],dtype=cdatatype)
                Heg,Wid,n_train,n_test = train_sub.shape[1],train_sub.shape[2],train_sub.shape[0],test_sub.shape[0]
                
                train_ifft = F.ifftn(train_sub,dim=(1,2),norm='ortho')
                train_in   = torch.zeros((n_train,2,Heg,Wid),dtype=datatype)
                train_in[:,0,:,:] = torch.real(train_ifft)
                train_in[:,1,:,:] = torch.imag(train_ifft)
                
                test_ifft  = F.ifftn(test_sub,dim=(1,2),norm='ortho')
                test_in    = torch.zeros((n_test,2,Heg,Wid),dtype=datatype)
                test_in[:,0,:,:] = torch.real(test_ifft)
                test_in[:,1,:,:] = torch.imag(test_ifft)              
            elif net.n_channels == 1:
                train_dir  = '/home/huangz78/data/traindata_x.npz' # randmask training
                train_sub  = torch.tensor(np.load(train_dir)['x'],dtype=datatype)                
                test_dir   = '/home/huangz78/data/testdata_x.npz'
                test_sub   = torch.tensor(np.load(test_dir)['x'],dtype=datatype)
                Heg,Wid,n_train,n_test = train_sub.shape[1],train_sub.shape[2],train_sub.shape[0],test_sub.shape[0]
                
                train_in   = torch.reshape(train_sub, (n_test,1,Heg,Wid))
                test_in    = torch.reshape(test_sub , (n_test,1,Heg,Wid))
                
            print('n_train = {}, n_test = {}'.format(n_train,n_test))          
                
        elif mode == 'greedy':
            assert net.n_channels==1
            imgs  = torch.tensor( np.load('/home/huangz78/data/data_gt.npz')['imgdata'] ).permute(2,0,1)
            masks = torch.tensor( np.load('/home/huangz78/data/data_gt_greedymask.npz')['mask'].T ) # labels are already rolled
            xs     = torch.zeros((imgs.shape[0],1,imgs.shape[1],imgs.shape[2]),dtype=torch.float)

            for ind in range(imgs.shape[0]):
                imgs[ind,:,:] = imgs[ind,:,:]/torch.max(torch.abs(imgs[ind,:,:]))
                y = torch.fft.fftshift(F.fftn(imgs[ind,:,:],dim=(0,1),norm='ortho'))
                mask = masks[ind,:]
                ysub = torch.zeros(y.shape,dtype=y.dtype)
                ysub[mask==1,:] = y[mask==1,:]
                xs[ind,0,:,:] = torch.abs(F.ifftn(torch.fft.ifftshift(ysub),dim=(0,1),norm='ortho'))

            imgNum = imgs.shape[0]
            traininds, testinds = train_test_split(np.arange(imgNum),random_state=0,shuffle=True,train_size=round(imgNum*0.8))
            test_total  = testinds.size
            np.savez('/home/huangz78/data/inds_rec.npz',traininds=traininds,testinds=testinds)
            train_full  = imgs[traininds,:,:]
            testfull    = imgs[testinds[0:test_total//2],:,:]

            train_sub   = xs[traininds,:,:,:]
            testsub     = xs[testinds[0:test_total//2],:,:,:]
            print('training data shape: ', train_sub.shape)
            Heg,Wid,n_train,n_test = train_sub.shape[2],train_sub.shape[3],train_sub.shape[0],testsub.shape[0]

            testfull = torch.reshape(testfull,(n_test,1,Heg,Wid))
            print('n_train = {}, n_test = {}'.format(n_train,n_test))

        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=lr_weight_decay, momentum=lr_momentum)
        # optimizer = optim.Adam/SGD
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_s_stepsize, gamma=lr_s_gamma)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
        criterion = nn.L1Loss() 

        train_loss = list([]); test_loss = list([]); train_loss_epoch = list([])
        global_step = 0
        train_batchnums = int(np.ceil(n_train/batch_size))
        test_batchnums  = int(np.ceil(n_test/t_bs))
        for epoch in range(epochs):
            epoch_loss = 0        
            batchind   = 0
            while batchind < train_batchnums:        
                batch = np.arange(batchind*batch_size,min((batchind+1)*batch_size,n_train))
                imgbatch   = train_in[batch,:,:,:]
                labelbatch = train_label[batch,:,:,:]

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
                    t_b       = np.arange(ind*t_bs,min((ind+1)*t_bs,n_test))
                    testin    = test_in[t_b,:,:,:]
                    testlabel = test_label[t_b,:,:,:]
                    pred      = net(testin).detach()
                    testloss += criterion(pred, testlabel)*len(t_b)
                test_loss.append(testloss.item()/n_test)
            print(f'\n\t[{epoch+1}/{epochs}]  loss/VAL: {testloss.item()/n_test}')
            net.train()
            scheduler.step()
            
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                np.savez('/home/huangz78/checkpoints/unet_'+ str(net.n_channels) + '_' +'TrainRec_'+str(net.skip)+'.npz',trainloss=train_loss,testloss=test_loss,trainloss_epoch=train_loss_epoch)
                torch.save({'model_state_dict': net.state_dict()}, dir_checkpoint + 'unet_' + str(net.n_channels) + '_' + str(net.skip) +'.pt')
    #                         }, dir_checkpoint + f'CP_epoch{epoch + 1}.pt')
                print(f'\t Checkpoint saved after epoch {epoch + 1}!')
                
    except KeyboardInterrupt:
        print('Keyboard Interrupted! Exit~')
        dir_checkpoint = '/home/huangz78/checkpoints/'
        torch.save({'model_state_dict': net.state_dict()}, dir_checkpoint + 'unet_' + str(net.n_channels) + '_' + str(net.skip) +'.pt')
        print(f'\t Checkpoint saved at Python epoch {epoch}, batchnum {batchind}!')
        np.savez('/home/huangz78/checkpoints/unet_'+ str(net.n_channels) + '_' +'TrainRec_'+str(net.skip)+'.npz',trainloss=train_loss,testloss=test_loss,trainloss_epoch=train_loss_epoch)
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
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=5e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-m','--mode',metavar='M',type=str,nargs='?',default='rand',
                        help='training mode', dest='mode')
    parser.add_argument('-cn', '--channel-num', metavar='CN', type=int, nargs='?', default=1,
                        help='channel number of unet', dest='n_channels')
    parser.add_argument('-s','--skip',type=bool,default=False,
                        help='residual network application', dest='skip')

    return parser.parse_args()
        
if __name__ == '__main__':
    args = get_args()
    print(args)
    
    unet = UNet(n_channels=args.n_channels,n_classes=1,bilinear=(not args.skip),skip=args.skip)
    train_net(unet,epochs=args.epochs,batch_size=args.batchsize,\
              lr=args.lr,lr_weight_decay=0,lr_momentum=0,\
              save_cp=True,mode=args.mode)