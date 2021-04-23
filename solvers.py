import numpy as np
import torch
from torch.nn.functional import relu
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import torch.fft as F

def TVA(X):

    diff1 = X[1:,:,:] - X[0:-1,:,:]
    diff2 = X[:,1:,:] - X[:,0:-1,:]
    tva   = torch.norm(diff1,p=1) + torch.norm(diff2,p=1)
    return tva.item()

def W1(x):
    d_row = torch.zeros_like(x)
    d_row[:-1,:,:] = x[:-1,:,:] - x[1:,:,:]
    d_row[-1 ,:,:] = x[-1,:,:] - x[0 ,:,:]
    return d_row

def W2(x):
    d_col = torch.zeros_like(x)
    d_col[:,:-1,:] = x[:,:-1,:] - x[:,1:,:]
    d_col[:,  -1,:] = x[:,-1,:] - x[:, 0,:]
    return d_col

def W1T(x):
    d_row = torch.zeros_like(x)
    d_row[1:,:,:] = x[1:,:,:] - x[:-1,:,:]
    d_row[0 ,:,:] = x[0 ,:,:] - x[-1 ,:,:]
    return d_row

def W2T(x):
    d_col = torch.zeros_like(x)
    d_col[:,1:,:] = x[:,1:,:] - x[:,:-1,:]
    d_col[:,0 ,:] = x[:,0 ,:] - x[:, -1,:]
    return d_col

def A_admm(X,mask,rho=1,DType=torch.cdouble):
    return torch.real(F.ifftn(torch.tensordot(torch.diag(mask).to(DType),F.fftn(X,dim=(0),norm='ortho'),dims=([1],[0])),dim=(0),norm='ortho')) + \
            rho * (W1T(W1(X)) + W2T(W2(X)))

def A_sfft(X,M,DType=torch.cdouble):
    return torch.real(F.ifftn(torch.tensordot(M.to(DType),F.fftn(X,dim=(0),norm='ortho'),dims=([1],[0])),dim=(0),norm='ortho'))


## TODO: add warmup initialization for conj_grad
def conj_grad(A, y, num_iters=100, verbose=False, eps=1e-8,x=None):
    """
    solve A x = y for x using the conjugate gradient method
    assumption: A is symmetric and positive-definite (not checked)
    Caveat:
        - The operator/matrix A is not preconditioned.
          TODO: To accelerate, preconditionining can help.
    """
    epsilon_sq = eps**2
#     if callable(AA):
#         A = AA
#     else:
#         A = lambda x: A*x
    if x is None:
        x = torch.zeros_like(y)
    r = y - A(x)
    d = r.clone()
    delta_new = (r * r).sum()
    delta_0 = delta_new.clone()
    for i in range(num_iters):
        if verbose:
            print(f'iter {i}, r norm: {r.norm()}')
            # logging.info(f'iter {i}, r norm: {r.norm()}')
        q = A(d)
        alpha = delta_new / (d * q).sum()
        x = x + alpha * d
        #if i % 50:
        r = y - A(x)
        #else:
        #r = r - alpha*q
        delta_old = delta_new.clone()
        delta_new = (r * r).sum()
        beta = delta_new / delta_old
        d = r + beta * d

        #logging.info((x - y).pow(2).sum() + gamma * D(x, ws).pow(2).sum())
        if delta_new < epsilon_sq * delta_0:
            if verbose:
                # logging.info('CG converged at iter {}, ending'.format(i))
                print('CG converged at iter {}, ending'.format(i))
            break

    return x

def TV(X):
    '''
    total variation operator
    _verified
    '''
    imgHeg,imgWid,layers = X.shape[0],X.shape[1],X.shape[2]
    p = torch.zeros(imgHeg,imgWid-1,layers)
    q = torch.zeros(imgHeg-1,imgWid,layers)
    p = X[:,:-1,:] - X[:,1:,:]
    q = X[:-1,:,:] - X[1:,:,:]
    return p,q

def TVadj(p,q,imgHeg,imgWid):
    '''
    Adjoint of total variation operator
    inputs:
    p: m * (n-1)
    q: (m-1) * n
    imgHeg
    imgWid

    _verified
    '''
    P = torch.zeros(imgHeg,imgWid+1,p.shape[2])
    Q = torch.zeros(imgHeg+1,imgWid,p.shape[2])
    P[:,1:-1,:] = p
    Q[1:-1,:,:] = q
    return P[:,1:,:]-P[:,:-1,:] + Q[1:,:,:] - Q[:-1,:,:]

def ADMM_TV(z,mask,maxIter=6,Lambda=10**(-6.5),rho=8e1,isotropic=False,\
                xOrig=None,DType=torch.cdouble,imgInput=False,\
                cgIter=20,cgeps=1e-8,cgverbose=False,verbose=False,x_init=None):
    """
    This function is a solver for the lower-level optimization problem by ADMM method.
    It aims to solve anisotropic-total-variation regularised image reconstruction problems.

    Inputs:
        z        : input image, assumed to be in the frequency domain and to represent actual observation
                   input dimension:(imgHeg,imgWid,layers), where the third dimension is optional
        mask     : mask, should be a vector (hopefully binary, meanwhile continuous value is okay)
        rho      : ADMM parameter governing convergence speed
        Lambda   : the magnitude of TV penalty
        imgInput : indicator for input to be in image space, by default False
        isotropic: default 'False', as 'anisotropic' option is faster
    Note:
        (1) the last dimension of y is the #layers
    """

    imgHeg = z.shape[0];  imgWid = z.shape[1]
    if len(z.shape)==3:
        layers = z.shape[2]
    else:
        layers = 1
        z = torch.reshape(z,(imgHeg,imgWid,layers))

    maxIter = int(maxIter); cgIter = int(cgIter)

    if (xOrig is not None) and (maxIter>0):
        error_nrmse = np.zeros(maxIter)

    if not imgInput:
        x_ifft = torch.real(F.ifftn(z , dim=(0,1),norm='ortho')) # z is the image in the k-space
    else:
        x_ifft = z.clone().detach()                # z is the image in the image space   
    
    if isotropic:
        M = torch.diag(mask)
        z1=y1=torch.zeros(imgHeg,imgWid-1,layers)
        z2=y2=torch.zeros(imgHeg-1,imgWid,layers)
        if x_init is not None:
            x = x_init.clone()
        else:
            x = x_ifft.clone()
        
        ind = 0 
        while ind < maxIter:
            if ((ind+1)%max(maxIter//4,1)==0) and verbose:
                print('ADMM_I iter {0} out of {1}'.format(ind+1,maxIter))
            RHS   = x_ifft + rho * TVadj(z1-(1/rho)*y1,z2-(1/rho)*y2,imgHeg,imgWid)
            AA    = lambda xx: A_sfft(xx,M,DType=DType) + rho*TVadj(*TV(xx),imgHeg,imgWid)
            x     = conj_grad(AA, RHS, num_iters=cgIter, verbose=cgverbose, eps=cgeps,x=x)

            a1,a2 = TV(x)
            p = a1 + y1/rho # p = xi
            q = a2 + y2/rho # q = eta
            factor  = torch.sqrt(p[:-1,:,:]**2 + q[:,:-1,:]**2)
            mag_tmp = 1 - (Lambda/rho)/torch.max(factor,(1/rho)*Lambda*torch.ones_like(factor))
            z1 = torch.zeros_like(p); z2 = torch.zeros_like(q)
            z1[:-1,:,:] = torch.mul(mag_tmp , p[:-1,:,:])
            z1[ -1,:,:] = torch.mul(relu(torch.abs(p[-1,:,:]) - (1/rho)*Lambda) , torch.sign(p[-1,:,:]))
            z2[:,:-1,:] = torch.mul(mag_tmp , q[:,:-1,:])
            z2[:, -1,:] = torch.mul(relu(torch.abs(q[:,-1,:]) - (1/rho)*Lambda) , torch.sign(q[:,-1,:]))

            y1 += rho*(a1 - z1)
            y2 += rho*(a2 - z2)

            # output MSE
            if xOrig is not None:
                error_nrmse[ind] = torch.norm(torch.flatten(x)-torch.flatten(xOrig),'fro') / torch.norm(torch.flatten(xOrig),'fro')
            ind += 1
    else: # anisotropic
        z1 = W1(x_ifft)
        z2 = W2(x_ifft)
        y1=y2 = torch.zeros(z.shape) 
        if x_init is not None:
            x = x_init.clone()
        else:
            x = x_ifft.clone()
            
        ind = 0
        while ind < maxIter:
            if ((ind+1)%max(maxIter//4,1) ==0) and verbose:
                print('ADMM_A iter {0} out of {1}'.format(ind+1,maxIter))
            RHS   = x_ifft + rho * (W1T(z1-y1/rho)+W2T(z2-y2/rho))
            AA    = lambda xx: A_admm(xx,mask,rho=rho,DType=DType)
            x     = conj_grad(AA, RHS, num_iters=cgIter, verbose=cgverbose, eps=cgeps, x=x)
            z1    = torch.mul( relu(torch.abs(W1(x)+y1/rho) - Lambda/rho) , torch.sign(W1(x)+y1/rho) )
            z2    = torch.mul( relu(torch.abs(W2(x)+y2/rho) - Lambda/rho) , torch.sign(W2(x)+y2/rho) )
            y1   += rho*(W1(x)-z1)
            y2   += rho*(W2(x)-z2)
            # output MSE
            if xOrig is not None:
                error_nrmse[ind] = torch.norm(torch.flatten(x)-torch.flatten(xOrig),'fro') / torch.norm(torch.flatten(xOrig),'fro')
            ind += 1
    if xOrig is not None:
        return x, error_nrmse
    else:
        return x