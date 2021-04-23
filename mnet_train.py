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

from mnet import MNet
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def trainMNet(trainimg,trainlabel,testimg,testlabel,epochs=20,batchsize=5,lr=0.01,\
              model=None,save_cp=True,threshold=0.6,\
              resnet=False,beta=1,poolk=3):
    
    train_shape = trainimg.shape; test_shape = testimg.shape # (#imgs,height,width,layer)
    trainimgs  = torch.tensor(trainimg).to(torch.float).view(train_shape[0],-1,train_shape[1],train_shape[2])
    trainlabel = torch.tensor(trainlabel).to(torch.float)
    testimgs   = torch.tensor(testimg).to(torch.float).view(test_shape[0],-1,test_shape[1],test_shape[2])
    testlabel  = torch.tensor(testlabel).to(torch.float)
    
    train_shape = trainimgs.shape
    
    # add normalization for images here
    
    if model is None:
        net = MNet(resnet=resnet,beta=beta,in_channels=train_shape[1],out_size=trainlabel.shape[1],imgsize=(train_shape[2],train_shape[3]),poolk=poolk)
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        epoch_init = 0
    else:
        net = model[0]
        optimizer = model[1]
        epoch_init = model[2] + 1

#     writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
#     criterion = nn.MSELoss()
    pos_weight = torch.ones([trainlabel.shape[1]]) * 7
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    dir_checkpoint = '/home/huangz78/mri/checkpoints/'
    sigmoid = nn.Sigmoid()
    
    epoch_loss = np.full((epochs),np.nan)
    precision_history = np.full((epochs),np.nan)
    recall_history = np.full((epochs),np.nan)
    for epoch in range(epoch_init,epoch_init + epochs):
        batch_init = 0; batch_count = 1
        while batch_init < train_shape[0]:
            batch = np.arange(batch_init,min(batch_init+batchsize,train_shape[0]))
            imgbatch = trainimgs[batch,:,:,:] # maybe shuffling?
            batchlabel = trainlabel[batch,:]
            mask_pred  = net(imgbatch)
            train_loss = criterion(mask_pred,batchlabel)
            if (batch_count%5)==0:
                mask_test = sigmoid(net(testimgs))
                mask_test_bool = torch.ones(mask_test.shape)
                mask_test_bool[mask_test<=threshold] = 0
                test_loss = test_criterion(mask_test_bool,testlabel)
                print('epoch {} batch {}: train batch loss {}, test loss {} '.format(epoch+1,batch_count,train_loss.item(),test_loss.item()))
            batch_init += batchsize; batch_count += 1
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        mask_test = sigmoid(net(testimgs))
        mask_test_bool = torch.ones(mask_test.shape)
        mask_test_bool[mask_test<=threshold] = 0
        test_loss = test_criterion(mask_test_bool,testlabel)
        scheduler.step(test_loss)
        epoch_loss[epoch-epoch_init] = test_loss.item()
        precision_history[epoch-epoch_init] = precision_score(torch.flatten(testlabel),torch.flatten(mask_test_bool))
        recall_history[epoch-epoch_init] = recall_score(torch.flatten(testlabel),torch.flatten(mask_test_bool))
        print('\t epoch {} end: test loss {} '.format(epoch+1,test_loss.item()))
        print('\t epoch {} end: precision {} '.format(epoch+1,precision_history[epoch-epoch_init]))
        print('\t epoch {} end: recall    {} '.format(epoch+1,recall_history[epoch-epoch_init]))
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                print('Created checkpoint directory')
#                 logging.info('Created checkpoint directory')
            except OSError:
                pass

            torch.save({'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'threshold':threshold
                        }, dir_checkpoint + 'mnet.pth')
#                         }, dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            print(f'\t Checkpoint saved after epoch {epoch + 1}!')
    
            np.savetxt(dir_checkpoint+'epoch_loss.txt', (epoch_loss,precision_history,recall_history), delimiter=',',header='loss,precision,recall')
#             logging.info(f'Checkpoint {epoch + 1} saved !')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-train','--traindata',type=str,default='/home/huangz78/data/traindata.npz',
                        help='train data path', dest='traindata')
    parser.add_argument('-test','--testdata',type=str,default='/home/huangz78/data/testdata.npz',
                        help='test data path', dest='testdata')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-tg','--threshold',type=float,nargs='?',default=.6,
                        help='threshold for binarinize output', dest='threshold')
    parser.add_argument('-m','--check-point',metavar='CR',type=str,nargs='?',default=None,
                        help='Path of checkpoint to load', dest='model_path')
    parser.add_argument('-beta','--beta',metavar='Beta',type=float,nargs='?',default=1,
                        help='Beta for Sigmoid function', dest='beta')
    parser.add_argument('-res','--resnet',default=False,
                        help='ResNet for MNet',action='store_true', dest='resnet')
    # parser.add_argument('-f', '--load', dest='load', type=str, default=False,
    #                     help='Load model from a .pth file')
    # parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
    #                     help='Downscaling factor of the images')
    # parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
    #                     help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    try:
        traindata = np.load(args.traindata)
        trainimg = traindata['ifftimgs']; trainlabel = traindata['labels']
        testdata = np.load(args.testdata)
        testimg = testdata['ifftimgs']; testlabel = testdata['labels']

        # breakpoint()
        if args.model_path is not None:
            net = MNet()
            optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

            checkpoint = torch.load(args.model_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_init = checkpoint['epoch']
            print(f'Model loaded from {args.model_path}')
            model = [net,optimizer,epoch_init]
        else:
            model = None

        trainMNet(trainimg,trainlabel, \
                  testimg,testlabel, \
                model=model, \
                epochs=args.epochs, \
                batchsize=args.batchsize, \
                lr=args.lr, \
                save_cp=True, \
                threshold=args.threshold,\
                beta=args.beta,\
                resnet=args.resnet)

    except KeyboardInterrupt: # need debug
        print('Keyboard Interrupted! Exit~')
    #     torch.save({'model_state_dict': net.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'epoch': epoch-1
    #                 }, dir_checkpoint + 'mnet.pth')
    #     print('Saved interrupt')
    #     # logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
