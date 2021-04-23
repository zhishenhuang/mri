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

sys.path.insert(0,'/home/huangz78/mri/unet/')
import unet_model
from unet_model import UNet

def train_net(net,epochs=5,batch_size=5,\
              lr=0.001,lr_weight_decay=1e-8,lr_momentum=0.9,\
              lr_s_stepsize=10,lr_s_gamma=0.5,\
              save_cp=False,datatype=torch.float):
    if net.n_channels == 2:
        train_dir = '/home/huangz78/data/traindata_y.npz'
        train_sub = np.load(train_dir)['y']
        train_full = np.load(train_dir)['yfull']
    elif net.n_channels == 1:
        train_dir = '/home/huangz78/data/traindata_x.npz'
        train_sub = np.load(train_dir)['x']
        train_full = np.load(train_dir)['xfull']
    
#     test_sub = np.copy(train_sub[0:2,:,:])
#     test_full = np.copy(train_full[0:2,:,:])

    test_dir = '/home/huangz78/data/testdata_x.npz'
    test_sub  = torch.tensor(np.load(test_dir)['x'])     ; test_sub  = test_sub[0:10,:,:]
    test_full = torch.tensor(np.load(test_dir)['xfull']) ; test_full = test_full[0:10,:,:]      
    Heg,Wid,n_train,n_test = train_sub.shape[1],train_sub.shape[2],train_sub.shape[0],test_sub.shape[0]
    
    print('n_train = {}, n_test = {}'.format(n_train,n_test))
   
    if net.n_channels == 2:
        testsub  = torch.zeros((n_test,2,Heg,Wid),dtype=datatype)
        testsub[:,0,:,:] = torch.real(test_sub)
        testsub[:,1,:,:] = torch.imag(test_sub)

        testfull  = torch.zeros((n_test,2,Heg,Wid),dtype=datatype)
        testfull[:,0,:,:] = torch.real(test_full)
        testfull[:,1,:,:] = torch.imag(test_full)
    elif net.n_channels == 1:
        testsub = torch.reshape(test_sub,(n_test,1,Heg,Wid)).to(datatype)
        testfull = torch.reshape(test_full,(n_test,1,Heg,Wid)).to(datatype)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=lr_weight_decay, momentum=lr_momentum)
    # optimizer = optim.Adam/SGD
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_s_stepsize, gamma=lr_s_gamma)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.MSELoss()
    
    train_loss = list([]); test_loss = list([])
    global_step = 1
    for epoch in range(epochs):
        epoch_loss = 0        
        batch_init = 0
        while batch_init < n_train:        
            batch = np.arange(batch_init,min(batch_init+batch_size,n_train))
            
            imgbatch_tmp = torch.tensor(train_sub[batch,:,:],dtype=datatype)
            if net.n_channels == 2:
                imgbatch = torch.zeros((len(batch),2,Heg,Wid),dtype=datatype)
                imgbatch[:,0,:,:] = torch.real(imgbatch_tmp)
                imgbatch[:,1,:,:] = torch.imag(imgbatch_tmp)
            elif net.n_channels == 1:
                imgbatch = torch.reshape(imgbatch_tmp,(len(batch),1,Heg,Wid))
            
            labelbatch_tmp = torch.tensor(train_full[batch,:,:],dtype=datatype)
            if net.n_channels == 2:
                labelbatch = torch.zeros((len(batch),2,Heg,Wid),dtype=datatype)
                labelbatch[:,0,:,:] = torch.real(labelbatch_tmp)
                labelbatch[:,1,:,:] = torch.imag(labelbatch_tmp)
            elif net.n_channels == 1:
                labelbatch = torch.reshape(labelbatch_tmp,(len(batch),1,Heg,Wid))
            
            batch_init += len(batch)
            
            pred = net(imgbatch)
            loss = criterion(pred, labelbatch)
            epoch_loss += loss.item()
#             writer.add_scalar('Loss/train', loss.item(), global_step)
            print('step:{}, loss/train: {}'.format(global_step,loss.item()))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            global_step += 1
            if ( global_step % max(n_train//(10*batch_size),1) )==0:
                pred = net(testsub)
                testloss = criterion(pred, testfull)
                print('step:{}, loss/test/: {}'.format(global_step,testloss))               
        if save_cp:
            dir_checkpoint = '/home/huangz78/mri/checkpoints/'
            try:
                os.mkdir(dir_checkpoint)
                print('Created checkpoint directory')
#                 logging.info('Created checkpoint directory')
            except OSError:
                pass

            torch.save({'model_state_dict': net.state_dict()}, dir_checkpoint + 'unet_' + str(net.n_channels) +'.pth')
#                         }, dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            print(f'\t Checkpoint saved after epoch {epoch + 1}!')
        train_loss.append(epoch_loss)
        pred = net(testsub)
        testloss = criterion(pred, testfull)
        test_loss.append(testloss.item())
        scheduler.step()
        np.savez('/home/huangz78/mri/mri_unet_rec.npz',trainloss=train_loss,testloss=test_loss)

if __name__ == '__main__':
    unet = UNet(n_channels=1,n_classes=1,bilinear=True,skip=False)
    train_net(unet,epochs=50,batch_size=10,\
                                 lr=5e-4,lr_weight_decay=0,lr_momentum=0,\
                                 save_cp=True)