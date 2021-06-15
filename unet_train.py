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
import unet_model
from unet_model import UNet

def train_net(net,epochs=5,batch_size=5,\
              lr=0.001,lr_weight_decay=1e-8,lr_momentum=0.9,\
              lr_s_stepsize=10,lr_s_gamma=0.8,\
              save_cp=False,datatype=torch.float,mode='rand',t_bs=3.0):
    print('mode = ',mode)
    net.train()
    try:
        if mode == 'rand':
            if net.n_channels == 2:
                train_dir = '/home/huangz78/data/traindata_y.npz'
                train_sub = np.load(train_dir)['y']
                train_full = np.load(train_dir)['yfull']
            elif net.n_channels == 1:
                train_dir = '/home/huangz78/data/traindata_x.npz' # randmask training
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
                print('epoch: {}, step:{}, loss/train: {}'.format(epoch+1, global_step,loss.item()))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                global_step += 1
#                 if ( global_step % max(n_train//(3*batch_size),1) )==0:
#                     testloss = 0
#                     net.eval()
#                     for ind in range( int(np.ceil(n_test/t_bs)) ):
#                         t_b = testsub[np.arange(ind*t_bs,min((ind+1)*t_bs,n_test)),:,:,:]     
#                         pred = net(t_b)
#                         testloss += criterion(pred, testfull[np.arange(ind*t_bs,min((ind+1)*t_bs,n_test)),:,:,:])
#                     print('step:{}, loss/test/: {}'.format(global_step,testloss))       
#                     net.train()
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                try:
                    os.mkdir(dir_checkpoint)
                    print('Created checkpoint directory')
    #                 logging.info('Created checkpoint directory')
                except OSError:
                    pass

                torch.save({'model_state_dict': net.state_dict()}, dir_checkpoint + 'unet_' + str(net.n_channels) + '_' + str(net.skip) +'.pth')
    #                         }, dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                print(f'\t Checkpoint saved after epoch {epoch + 1}!')
            train_loss.append(epoch_loss)
            testloss = 0
            net.eval()
            for ind in range( int(np.ceil(n_test/t_bs)) ):
                t_b = testsub[np.arange(ind*t_bs,min((ind+1)*t_bs,n_test)),:,:,:]     
                pred = net(t_b)
                testloss += criterion(pred, testfull[np.arange(ind*t_bs,min((ind+1)*t_bs,n_test)),:,:,:])
    #         pred = net(testsub)
    #         testloss = criterion(pred, testfull)
            test_loss.append(testloss.item())
            net.train()
            scheduler.step()
            np.savez('/home/huangz78/checkpoints/mri_unet_rec_'+str(net.skip)+'.npz',trainloss=train_loss,testloss=test_loss)
    except KeyboardInterrupt:
        print('Keyboard Interrupted! Exit~')
        dir_checkpoint = '/home/huangz78/checkpoints/'
        torch.save({'model_state_dict': net.state_dict()}, dir_checkpoint + 'unet_' + str(net.n_channels) + '_' + str(net.skip) +'.pth')
        print(f'\t Checkpoint saved after epoch {epoch + 1}!')
        np.savez('/home/huangz78/checkpoints/mri_unet_rec_'+str(net.skip)+'.npz',trainloss=train_loss,testloss=test_loss)
        print('Model is saved after interrupt~')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s','--skip',type=bool,default=False,
                        help='residual network application', dest='skip')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=60,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=5e-4,
                        help='Learning rate', dest='lr')

    return parser.parse_args()
        
if __name__ == '__main__':
    args = get_args()
    print(args)
    
    unet = UNet(n_channels=1,n_classes=1,bilinear=(not args.skip),skip=args.skip)
    train_net(unet,epochs=args.epochs,batch_size=args.batchsize,\
              lr=args.lr,lr_weight_decay=0,lr_momentum=0,\
              save_cp=True,mode='greedy')