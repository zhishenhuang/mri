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

def sigmoid_binarize(M,threshold=0.6):
    sigmoid = nn.Sigmoid()
    mask = sigmoid(M)
    mask_pred = torch.ones_like(mask)
    mask_pred[mask<=threshold] = 0
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
        trainimgs    = torch.tensor(trainimgs,dtype=datatype).view(train_shape[0],-1,train_shape[1],train_shape[2])
        trainlabels  = torch.tensor(trainlabels,dtype=datatype)
        testimgs     = torch.tensor(testimgs,dtype=datatype).view(test_shape[0],-1,test_shape[1],test_shape[2])
        testlabels   = torch.tensor(testlabels ,dtype=datatype)

        train_shape = trainimgs.shape
        dir_checkpoint = '/home/huangz78/checkpoints/'
        # add normalization for images here

        if model is None:
            net = MNet(beta=beta,in_channels=train_shape[1],out_size=trainlabels.shape[1],\
                       imgsize=(train_shape[2],train_shape[3]),poolk=poolk)
            optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=lr_weight_decay, momentum=opt_momentum)
            epoch_init = 0
        else:
            net = model[0]
            optimizer  = model[1]
            epoch_init = model[2] + 1
    #     criterion = nn.MSELoss()
        pos_weight = torch.ones([trainlabels.shape[1]]) * positive_weight # weight assigned to positive labels 
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        test_criterion = nn.BCELoss()
        sigmoid = nn.Sigmoid()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_s_stepsize,factor=lr_s_gamma)

        epoch_loss        = np.full((epochs),np.nan)
        precision_history = np.full((epochs),np.nan)
        recall_history    = np.full((epochs),np.nan)
        for epoch in range(epoch_init,epoch_init + epochs):
            batch_init = 0; step_count = 1
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
                if (step_count%print_every)==0:
                    with torch.no_grad():
                        net.eval()
                        mask_test = sigmoid_binarize(net(testimgs),threshold=threshold)
                        test_loss = test_criterion(mask_test,testlabels)
                        print('epoch {} global step {}: train batch loss {}, test loss {} '.format(epoch+1,step_count,train_loss.item(),test_loss.item()))
                        net.train()
            with torch.no_grad():
                net.eval()
                mask_test = sigmoid_binarize(net(testimgs),threshold=threshold)
                test_loss = test_criterion(mask_test,testlabels)
                net.train()
                scheduler.step(test_loss)
                epoch_loss[epoch-epoch_init] = test_loss.item()
                precision_history[epoch-epoch_init] = precision_score(torch.flatten(testlabels),torch.flatten(mask_test))
                recall_history[epoch-epoch_init] = recall_score(torch.flatten(testlabels),torch.flatten(mask_test))
                print('\t epoch {} end: test loss {} '.format(epoch+1,test_loss.item()))
                print('\t epoch {} end: precision {} '.format(epoch+1,precision_history[epoch-epoch_init]))
                print('\t epoch {} end: recall    {} '.format(epoch+1,recall_history[epoch-epoch_init]))
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    print('Created checkpoint directory')
                except OSError:
                    pass
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'threshold':threshold
                            }, dir_checkpoint + 'mnet.pth')
    #                         }, dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                print(f'\t Checkpoint saved after epoch {epoch + 1}!')

                np.savez(dir_checkpoint+'epoch_loss.npz', loss=epoch_loss,precision=precision_history,recall=recall_history)
    except KeyboardInterrupt: # need debug
        print('Keyboard Interrupted! Exit~')
        torch.save({'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch-1
                    }, dir_checkpoint + 'mnet.pth')
        print('Model, optimizer and epoch count saved after interrupt~')
        # logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-train','--traindata',type=str,default='/home/huangz78/data/traindata.npz',
                        help='train data path', dest='traindata')
    parser.add_argument('-test','--testdata',type=str,default='/home/huangz78/data/testdata.npz',
                        help='test data path', dest='testdata')
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
    # parser.add_argument('-f', '--load', dest='load', type=str, default=False,
    #                     help='Load model from a .pth file')
    # parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
    #                     help='Downscaling factor of the images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    data_path = '/home/huangz78/data/'
    data_imgs = np.load(data_path+'data_gt.npz')
    data_labels = np.load(data_path+'data_gt_greedymask.npz')
    print(data_imgs.files)
    data    = data_imgs['imgdata']
    labels  = data_labels['mask'].T
    datashape = data.shape
    print('all data shape',datashape)
    print('all labels shape', labels.shape)

    base = 24
    print('base frequencies: ', base)
    mask = torch.tensor( mask_naiveRand(320,fix=base,other=0,roll=False)[0] ,dtype=torch.float )
    data_under = np.zeros((datashape[2],datashape[0],datashape[1]))
    for ind in range(data.shape[2]):
        img = data[:,:,ind]
        img = img/np.max(np.abs(img)) # normalize all images
        yfull = F.fftn(torch.tensor(img,dtype=torch.float),dim=(0,1),norm='ortho')
        ypart = torch.tensordot(torch.diag(mask).to(torch.cfloat) , yfull, dims=([1],[0]))
        data_under[ind,:,:] = torch.abs(F.ifftn(ypart,dim=(0,1),norm='ortho'))

    imgNum = datashape[2]
    traininds, testinds = train_test_split(np.arange(imgNum),random_state=0,shuffle=True,train_size=round(imgNum*0.8))
    test_total   = testinds.size
    traindata    = data_under[traininds,:,:]
    trainlabels  = mask_filter(labels[traininds,:],base=base)
    valdata      = data_under[testinds[0:test_total//2],:,:]
    vallabels    = mask_filter(labels[testinds[0:test_total//2],:],base=base)


    if args.model_path is not None:
        net = MNet()
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=0, momentum=0)
        checkpoint = torch.load(args.model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch']
        print(f'Model loaded from {args.model_path}')
        model = [net,optimizer,epoch_init]
    else:
        model = None

    trainMNet(traindata, trainlabels, valdata, vallabels,
              epochs=args.epochs, batchsize=args.batchsize, \
              positive_weight=args.positive_weight,\
              lr=args.lr, lr_weight_decay=0, opt_momentum=0,\
              lr_s_stepsize=2, lr_s_gamma=0.5,\
              threshold=args.threshold, beta=args.beta, save_cp=True)


    
