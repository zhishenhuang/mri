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
from matplotlib import pyplot as plt
import random
import copy

from utils import mask_naiveRand, mask_filter, get_x_f_from_yfull, mnet_wrapper, mask_complete, mask_makebinary, raw_normalize
from mnet import MNet
from mask_backward_new import mask_backward, mask_eval
sys.path.insert(0,'/home/huangz78/mri/unet/')
from unet_model import UNet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def alternating_update_with_unetRecon(mnet,unet,trainfulls,testimg,mask_init,mask_init_full=True,\
                                      maxIter_mb=50,evalmode='unet',alpha=2.8*1e-5,c=0.05, Lambda=1e-4,\
                                      lr_mb=1e-4,lr_mn=1e-4,maxRep=5,epoch=1,batchsize=5,\
                                      corefreq=24,budget=24,plot=False,verbose=False,mask_greedy=None,\
                                      change_initmask=True,validate_every=10,dtyp=torch.float,\
                                      save_cp=False):
    '''
    alpha: magnitude of l1 penalty for high-frequency mask
    mnet : the input mnet needs to coordinate exactly with corefreq
    '''
    if mask_init_full:
        fullmask = torch.tensor(mask_init).clone()
        highmask = mask_filter(fullmask,base=corefreq,roll=True)
    else:
        fullmask = mask_complete(torch.tensor(mask_init),trainfulls.shape[1],rolled=True,dtyp=dtyp)
        highmask = torch.tensor(mask_init).clone()
    DTyp = torch.cfloat if dtyp==torch.float else torch.cdouble
    dir_checkpoint = '/home/huangz78/checkpoints/'
    criterion_mnet = nn.BCEWithLogitsLoss()
    optimizer_m = optim.RMSprop(mnet.parameters(), lr=lr_mn, weight_decay=0, momentum=0)
    # optimizer_u = ......
    
    unet_eval = UNet(n_channels=1,n_classes=1,bilinear=True,skip=False)
    unet_eval = copy.deepcopy(unet)
    unet_eval.eval()
    # training loop
    global_step = 0

    randqual = []; mnetqual = []
    randspar = []; mnetspar = []
    if mask_greedy is not None:
        greedyqual = []
        greedyspar = np.sum(mask_greedy[0,:])/trainfulls.shape[1]
    else:
        greedyqual = None; greedyspar = None
    epoch_count = 0
    while epoch_count<epoch:
        for xstar in trainfulls:
            xstar = torch.tensor(xstar,dtype=dtyp)
            yfull = torch.fft.fftshift(F.fftn(xstar,dim=(0,1),norm='ortho')) # y is ROLLED!
            lowfreqmask,_,_ = mask_naiveRand(xstar.shape[0],fix=corefreq,other=0,roll=True)
            x_lf            = get_x_f_from_yfull(lowfreqmask,yfull)
            ########################################  
            ## (1) mask_backward
            ########################################        
            if change_initmask and global_step>0: # option 2: highmask = mask_pred from step (2)
                highmask = mnet(x_lf.view(1,1,xstar.shape[0],xstar.shape[1])).view(-1)
            highmask_refined,unet = mask_backward(highmask,xstar,unet=unet, mnet=mnet,\
                              beta=1.,alpha=alpha,c=c,\
                              maxIter=maxIter_mb,seed=0,break_limit=maxIter_mb*3//5,\
                              lr=lr_mb,mode='UNET',budget=budget,normalize=True,\
                              verbose=verbose,dtyp=torch.float)        
            ########################################  
            ## (2) update mnet
            ########################################        
            mnet.train()
            unet.eval()
            rep = 0
            while rep < maxRep:
                mask_pred  = mnet(x_lf.view(1,1,xstar.shape[0],xstar.shape[1]))
                train_loss = criterion_mnet(mask_pred,highmask_refined.view(mask_pred.shape))
                optimizer_m.zero_grad()
                # optimizer step wrt unet parameters ?
                train_loss.backward()
                optimizer_m.step()
                rep += 1
            mnet.eval()             
            ########################################  
            ## (3) check mnet performance: does it beat random sampling? here 'test' means validation
            ########################################
            if (global_step%validate_every==0) or (global_step==trainfulls.shape[0]-1):
                randqual_tmp = 0; mnetqual_tmp = 0; greedyqual_tmp = 0
                randspar_tmp = 0; mnetspar_tmp = 0
                imgind = 0
                for img in testimg:
                    x_test_lf     = img      
                    mask_test     = mnet_wrapper(mnet,x_test_lf,budget,img.shape,dtyp=dtyp)
                    mask_rand,_,_ = mask_naiveRand(img.shape[0],fix=corefreq,other=mask_test.sum().item()-corefreq,roll=True)        
                    randqual_img  = mask_eval(mask_rand,img,UNET=unet_eval) # do not need to eval rand. mask every validation call
                    mnetqual_img  = mask_eval(mask_test,img,UNET=unet) # UNET = unet_eval               
                    randqual_tmp += randqual_img
                    mnetqual_tmp += mnetqual_img                
                    if verbose:
                        print('Quality of random mask : ', randqual_img) 
                        print('Quality of mnet   mask : ', mnetqual_img)

                    ### compute sampling ratio of generated masks
                    randspar_img  = mask_rand.sum().item()/img.shape[0]
                    mnetspar_img  = mask_test.sum().item()/img.shape[0]
                    randspar_tmp += randspar_img
                    mnetspar_tmp += mnetspar_img
                    if mask_greedy is not None:
                        greedyqual_img = mask_eval(mask_greedy[imgind,:],img,mode='sigpy',Lambda=Lambda) # UNET=unet_eval
                        greedyqual_tmp += greedyqual_img
                        if verbose:
                            print('Quality of greedy mask : ', greedyqual_img)
                            print(f'sparsity of random mask: {randspar_img},mnet mask: {mnetspar_img}, \
                                    greedy mask: {greedyspar}\n')
                    else:
                        if verbose:
                            print(f'sparsity of random mask: {randspar_img},mnet mask: {mnetspar_img}\n')
                    imgind += 1
                randqual.append( randqual_tmp/testimg.shape[0] )
                mnetqual.append( mnetqual_tmp/testimg.shape[0] )
                if mask_greedy is not None:
                    greedyqual.append( greedyqual_tmp/testimg.shape[0] )
                randspar.append( randspar_tmp/testimg.shape[0] )
                mnetspar.append( mnetspar_tmp/testimg.shape[0] )
                if plot:
                    try:
                        visualization(randqual,mnetqual,greedyqual=greedyqual,\
                                 randspar=randspar,mnetspar=mnetspar,greedyspar=greedyspar*np.ones(len(greedyqual)))
                    except Exception:
                        visualization(randqual,mnetqual,randspar=randspar,mnetspar=mnetspar)
                if save_cp:
                    torch.save({'model_state_dict': mnet.state_dict()}, dir_checkpoint + 'mnet_split_trained.pth')
                    torch.save({'model_state_dict': unet.state_dict()}, dir_checkpoint + 'unet_split_trained.pth')
                    print(f'\t Checkpoint saved at epoch {epoch_count}, iter {global_step + 1}!')
                    filepath = '/home/huangz78/checkpoints/alternating_update_error_track.npz'
                    np.savez(filepath,randqual=randqual,mnetqual=mnetqual,greedyqual=greedyqual,randspar=randspar,mnetspar=mnetspar)
            global_step += 1
        epoch_count+= 1


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-train','--traindata',type=str,default='/home/huangz78/data/traindata_x.npz',
                        help='train data path', dest='traindata')
    parser.add_argument('-test','--testdata',type=str,default='/home/huangz78/data/testdata_x.npz',
                        help='test data path', dest='testdata')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=60,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lrb', '--learning-rate-backward', metavar='LRB', type=float, nargs='?', default=1e-4,
                        help='Learning rate for maskbackward', dest='lrb')
    parser.add_argument('-lrn', '--learning-rate-mnet', metavar='LRN', type=float, nargs='?', default=1e-4,
                        help='Learning rate for mnet', dest='lrn')
    parser.add_argument('-tg','--threshold',type=float,nargs='?',default=.5,
                        help='threshold for binarinize output', dest='threshold')
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=24,
                        help='number of observed low frequencies', dest='base_freq')
    parser.add_argument('-bg','--budget',metavar='BG',type=int,nargs='?',default=16,
                        help='number of high frequencies to sample', dest='budget')
    parser.add_argument('-beta','--beta',metavar='Beta',type=float,nargs='?',default=1,
                        help='Beta for Sigmoid function', dest='beta')
    parser.add_argument('-cin','--channel-input',type=int,nargs='?',default=1,
                        help='input channel',dest='cin')
    parser.add_argument('-mp', '--mnet-path', type=str, default='/home/huangz78/checkpoints/mnet.pth',
                        help='path file for a mnet', dest='mnetpath')
    parser.add_argument('-up', '--unet-path', type=str, default='/home/huangz78/checkpoints/unet_1.pth',
                        help='path file for a unet', dest='unetpath')
    parser.add_argument('-mbit', '--mb-iter-max', type=int, default=5,
                        help='maximum interation for maskbackward function', dest='maxItermb')
    parser.add_argument('-mnrep', '--mn-iter-rep', type=int, default=2,
                        help='inside one batch, updating mnet this many times', dest='mnRep')

    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    print(args)
    
    # load a mnet
    mnet = MNet(out_size=320-args.base_freq)
    checkpoint = torch.load(args.mnetpath)
    mnet.load_state_dict(checkpoint['model_state_dict'])
    print('MNet loaded successfully from: ', args.mnetpath)
    
    # load a unet for maskbackward
    UNET =  UNet(n_channels=args.cin,n_classes=1,bilinear=True,skip=False)
    checkpoint = torch.load(args.unetpath)
    UNET.load_state_dict(checkpoint['model_state_dict'])
    print('Unet loaded successfully from : ', arg.unetpath )
    
    train_dir  = args.traindata
    test_dir = args.testdata
    if args.cin == 1:
        train_full = np.load(train_dir)['xfull']
        testimg  = torch.tensor(np.load(test_dir)['x']) 
        print(testimg.shape)
    else:
        train_full = np.load(train_dir)['yfull']
        testimg  = torch.tensor(np.load(test_dir)['y']) 
        print(testimg.shape)
    
    fullmask = torch.fft.fftshift(torch.tensor(np.load(train_dir)['mask'])) # roll the input mask
    
    mask_greedy = np.load('/home/huangz78/data/data_gt_greedymask.npz')
    mask_greedy = mask_greedy['mask'].T # this greedy mask is rolled
    print(mask_greedy.shape)
    
    alternating_update_with_unetRecon(mnet,UNET,train_full,testimg[0:20,:,:],fullmask,\
                                  alpha=3e-4,c=1e-2,Lambda=1e-4,epoch=args.epochs,batchsize=args.batchsize,\
                                  lr_mb=args.lrb,lr_mn=args.lrn,\
                                  maxIter_mb=args.maxItermb,maxRep=args.mnRep,\
                                  corefreq=args.base_freq,budget=args.budget,\
                                  mask_greedy=mask_greedy,change_initmask=True,\
                                  verbose=False,plot=False,validate_every=50,save_cp=True)