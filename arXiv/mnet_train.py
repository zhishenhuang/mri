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
import utils
import logging

from sklearn.model_selection import train_test_split
from utils import kplot,mask_naiveRand,mask_filter
sys.path.insert(0,'/home/huangz78/mri/mnet/')
from mnet import MNet
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
    
def sigmoid_binarize(M,threshold=0.5):
    sigmoid = nn.Sigmoid()
    mask = sigmoid(M)
    mask_pred = torch.ones_like(mask)
    for ind in range(M.shape[0]):
        mask_pred[ind,mask[ind,:]<=threshold] = 0
    return mask_pred

def trainMNet(trainimgs,trainlabels,testimgs,testlabels,\
              epochs=20,batchsize=5,\
              lr=0.01,lr_weight_decay=1e-8,opt_momentum=0,positive_weight=6,\
              lr_s_stepsize=5,lr_s_gamma=0.5,\
              model=None,save_cp=True,threshold=0.5,\
              beta=1,poolk=3,datatype=torch.float,print_every=10):
    '''
    trainimgs    : train data, with dimension (#imgs,height,width,layer)
    '''
    try:
        train_shape  = trainimgs.shape; test_shape = testimgs.shape 
        trainimgs    = torch.tensor(trainimgs,dtype=datatype)
        trainlabels  = torch.tensor(trainlabels,dtype=datatype)
        testimgs     = torch.tensor(testimgs,dtype=datatype)
        testlabels   = torch.tensor(testlabels,dtype=datatype)    
        dir_checkpoint = '/home/huangz78/checkpoints/'
        # input images are assumed to be normalized

        if model is None:
            net = MNet(beta=beta,in_channels=train_shape[1],out_size=trainlabels.shape[1],\
                       imgsize=(train_shape[2],train_shape[3]),poolk=poolk)
        else:
            net = model
    #     optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=lr_weight_decay, momentum=opt_momentum)
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=lr_weight_decay, amsgrad=False)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_s_stepsize,factor=lr_s_gamma)
        pos_weight = torch.ones([trainlabels.shape[1]]) * positive_weight # weight assigned to positive labels 
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        test_criterion = nn.BCELoss()
        train_loss_rec    = list([])
        epoch_loss_rec    = np.full((epochs),np.nan)
        precision_train   = list([]); recall_train = list([])
        precision_history = np.full((epochs),np.nan); recall_history = np.full((epochs),np.nan)
        net.train()
        for epoch in range(epochs):
            batch_init = 0; step_count = 0
            while batch_init < train_shape[0]:
                batch = np.arange(batch_init,min(batch_init+batchsize,train_shape[0]))
                imgbatch = trainimgs[batch,:,:,:] # maybe shuffling?
                batchlabels = trainlabels[batch,:]
                mask_pred   = net(imgbatch)
                train_loss  = criterion(mask_pred,batchlabels)
                batch_init += batchsize; step_count += 1
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                train_loss_rec.append(train_loss.item())
                if (step_count%print_every)==0:
                    print('[{}/{}][{}] train batch loss {}'.format(epoch+1,epochs,step_count,train_loss.item()))
                    precision_train.append(\
                        precision_score(torch.flatten(batchlabels),\
                                        torch.flatten(sigmoid_binarize(mask_pred,threshold=threshold))) ) 
                    recall_train.append(\
                        recall_score(torch.flatten(batchlabels),\
                                     torch.flatten(sigmoid_binarize(mask_pred,threshold=threshold))) )
                    print('[{}/{}][{}] precision {}, recall {}'.format(epoch+1,epochs,step_count,precision_train[-1],recall_train[-1]))
            with torch.no_grad():
                net.eval()
                mask_test = sigmoid_binarize(net(testimgs),threshold=threshold)
                test_loss = test_criterion(mask_test,testlabels)
                net.train()
    #             scheduler.step(test_loss)
                epoch_loss_rec[epoch] = test_loss.item()
                precision_history[epoch] = precision_score(torch.flatten(testlabels),torch.flatten(mask_test))
                recall_history[epoch] = recall_score(torch.flatten(testlabels),torch.flatten(mask_test))
                print('\t [{}/{}] validation loss {} '.format(epoch+1,epochs,test_loss.item()))
                print('\t [{}/{}] precision {} '.format(epoch+1,epochs,precision_history[epoch]))
                print('\t [{}/{}] recall    {} '.format(epoch+1,epochs,recall_history[epoch]))
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    print('Created checkpoint directory')
                except OSError:
                    pass
                torch.save({'model_state_dict': net.state_dict()}, dir_checkpoint + 'mnet.pth')
    #                         'optimizer_state_dict': optimizer.state_dict(),
    #                         'epoch': epoch,
    #                         'threshold':threshold
    #                         }, dir_checkpoint + 'mnet.pth')
    #                         }, dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                print(f'\t Checkpoint saved after epoch {epoch + 1}!')
                np.savez(dir_checkpoint+'mnet_train_history.npz', loss_train=train_loss_rec,loss_test=epoch_loss_rec,\
                         precision_train=precision_train,recall_train=recall_train,\
                         precision_test=precision_history,recall_test=recall_history)
    except KeyboardInterrupt: # need debug
        print('Keyboard Interrupted! Exit~')
#         torch.save({'model_state_dict': net.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'epoch': epoch-1
#                     }, dir_checkpoint + 'mnet.pth')
#         print('Model, optimizer and epoch count saved after interrupt~')
        torch.save({'model_state_dict': net.state_dict()}, dir_checkpoint + 'mnet.pth')
        print(f'\t Checkpoint saved after epoch {epoch + 1}!')
        np.savez(dir_checkpoint+'mnet_train_history.npz',loss_train=train_loss_rec, loss_test=epoch_loss_rec,\
                         precision_train=precision_train,recall_train=recall_train,\
                         precision_test=precision_history,recall_test=recall_history)
        print('Model is saved after interrupt~')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=60,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-tg','--threshold',type=float,nargs='?',default=.5,
                        help='threshold for binarinize output', dest='threshold')
    parser.add_argument('-m','--check-point',metavar='CR',type=str,nargs='?',default=None,
                        help='Path of checkpoint to load', dest='model_path')
    parser.add_argument('-beta','--beta',metavar='Beta',type=float,nargs='?',default=1,
                        help='Beta for Sigmoid function', dest='beta')
    parser.add_argument('-pw','--positive-weight',type=float,nargs='?',default=1,
                        help='weight for positive rows',dest='positive_weight')
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=24,
                        help='number of observed low frequencies', dest='base_freq')
    parser.add_argument('-mode','--train-mode',type=str,nargs='?',default='y',
                        help='training mode: use x or y as input',dest='mode')
    # parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
    #                     help='Downscaling factor of the images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    
    imgs = torch.tensor( np.load('/home/huangz78/data/data_gt.npz')['imgdata'] ).permute(2,0,1)
    base = args.base_freq
    mask_lf,_,_ = mask_naiveRand(imgs.shape[1],fix=base,other=0,roll=True)

    yfulls = torch.zeros((imgs.shape[0],2,imgs.shape[1],imgs.shape[2]),dtype=torch.float)
    ys     = torch.zeros((imgs.shape[0],2,imgs.shape[1],imgs.shape[2]),dtype=torch.float)
    xs     = torch.zeros((imgs.shape[0],1,imgs.shape[1],imgs.shape[2]),dtype=torch.float)
    for ind in range(imgs.shape[0]):
        imgs[ind,:,:] = imgs[ind,:,:]/torch.max(torch.abs(imgs[ind,:,:]))
        y = torch.fft.fftshift(F.fftn(imgs[ind,:,:],dim=(0,1),norm='ortho'))
        ysub = torch.zeros(y.shape,dtype=y.dtype)
        ysub[mask_lf==1,:] = y[mask_lf==1,:]
        xs[ind,0,:,:] = torch.abs(F.ifftn(torch.fft.ifftshift(ysub),dim=(0,1),norm='ortho')) 

        yfulls[ind,0,:,:] = torch.real(y)
        yfulls[ind,1,:,:] = torch.imag(y)
        ys[ind,:,mask_lf==1,:] = yfulls[ind,:,mask_lf==1,:]

    labels = torch.tensor( np.load('/home/huangz78/data/data_gt_greedymask.npz')['mask'].T ) # labels are already rolled
    
    imgNum = imgs.shape[0]
    traininds, testinds = train_test_split(np.arange(imgNum),random_state=0,shuffle=True,train_size=round(imgNum*0.8))
    test_total  = testinds.size
    
    trainlabels = mask_filter(labels[traininds,:],base=base)
    vallabels   = mask_filter(labels[testinds[0:test_total//2],:],base=base)
    
    if args.mode == 'y':
        traindata   = ys[traininds,:,:,:]
        valdata     = ys[testinds[0:test_total//2],:,:,:]
        print('training data shape: ', traindata.shape)
    elif args.mode == 'x':        
        traindata   = xs[traininds,:,:,:]
        valdata     = xs[testinds[0:test_total//2],:,:,:]
        print('training data shape: ', traindata.shape)
        
    if args.model_path is not None:
        net = MNet(beta=1,in_channels=traindata.shape[1],out_size=trainlabels.shape[1],\
                   imgsize=(traindata.shape[2],traindata.shape[3]),poolk=3)
#         optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=0, momentum=0)
        checkpoint = torch.load(args.model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         epoch_init = checkpoint['epoch']
        net.train()
        print(f'Model loaded from {args.model_path}')
        model = net
#         model = [net,optimizer,epoch_init]
    else:
        model = None

    trainMNet(traindata, trainlabels, valdata, vallabels,model=model, \
              epochs=args.epochs, batchsize=args.batchsize, \
              positive_weight=args.positive_weight,\
              lr=args.lr, lr_weight_decay=0, opt_momentum=0,\
              lr_s_stepsize=2, lr_s_gamma=0.8,\
              threshold=args.threshold, beta=args.beta, save_cp=True)

