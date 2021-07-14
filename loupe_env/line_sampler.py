import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.fft as F
from torch.autograd import Function
from utils import kplot

def RescaleProbMap(batch_x, sparsity):
    """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity
        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
    """
    batch_size = len(batch_x)
    ret = []
    for i in range(batch_size):
        x = batch_x[i:i+1]
        xbar = torch.mean(x)
        r = sparsity / (xbar)
        beta = (1-sparsity) / (1-xbar)

        # compute adjucement
        le = torch.le(r, 1).float()
        ret.append(le * x * r + (1-le) * (1 - (1 - x) * beta))

    return torch.cat(ret, dim=0)


class ThresholdRandomMaskSigmoidV1(Function):
    def __init__(self):
        """
            Straight through estimator.
            The forward step stochastically binarizes the probability mask.
            The backward step estimate the non differentiable > operator using sigmoid with large slope (10).
        """
        super(ThresholdRandomMaskSigmoidV1, self).__init__()

    @staticmethod
    def forward(ctx, input):
        batch_size = len(input)
        probs = [] 
        results = [] 

        for i in range(batch_size):
            x = input[i:i+1]

            count = 0 
            while True:
                prob = x.new(x.size()).uniform_()
                result = (x > prob).float()

                if torch.isclose(torch.mean(result), torch.mean(x), atol=1e-3):
                    break

                count += 1 

                if count > 1000:
                    print(torch.mean(prob), torch.mean(result), torch.mean(x))
                    assert 0 

            probs.append(prob)
            results.append(result)

        results = torch.cat(results, dim=0)
        probs = torch.cat(probs, dim=0)
        ctx.save_for_backward(input, probs)

        return results  

    @staticmethod
    def backward(ctx, grad_output):
        slope = 10
        input, prob = ctx.saved_tensors

        # derivative of sigmoid function
        current_grad = slope * torch.exp(-slope * (input - prob)) / torch.pow((torch.exp(-slope*(input-prob))+1), 2)

        return current_grad * grad_output

class LineConstrainedProbMask(nn.Module):
    """
    A learnable probablistic mask with the same shape as the kspace measurement.
    The mask is constrinaed to include whole kspace lines in the readout direction
    """
    def __init__(self, shape=[32], slope=5, preselect=False, preselect_num=2):
        super(LineConstrainedProbMask, self).__init__()

        if preselect:
            length = shape[0] - preselect_num 
        else:
            length = shape[0]

        self.preselect_num = preselect_num 
        self.preselect     = preselect 
        self.slope         = slope
        init_tensor        = self._slope_random_uniform(length)
        self.mask          = nn.Parameter(init_tensor)
        
    def _slope_random_uniform(self, shape, eps=1e-2):
        """
            uniform random sampling mask with the same shape as the kspace measurement
        """
        temp = torch.zeros(shape).uniform_(eps, 1-eps)

        # logit with slope factor
        return -torch.log(1./temp-1.) / self.slope

    def forward(self, input, eps=1e-10):
        """
        Args:
            input (torch.Tensor): Input tensor of shape NHWC

        Returns:
            (torch.Tensor): Output tensor of shape NHWC
        """
        logits = self.mask
        mask   = torch.sigmoid(self.slope * logits).view(1, 1, self.mask.shape[0], 1) 
        if self.preselect:
            if self.preselect_num % 2 == 0:
                zeros = torch.zeros(1, 1, self.preselect_num // 2, 1).to(input.device) 
                mask = torch.cat([zeros, mask, zeros], dim=2)
                mask.retain_grad()
            else:
                raise NotImplementedError()

        return mask 

class LOUPESampler(nn.Module):
    """
        LOUPE Sampler
    """
    def __init__(self, shape=[320, 320], slope=5, sparsity=0.25, preselect=False, preselect_num=2):
        """
            shape ([int. int]): Shape of the reconstructed image
            slope (float): Slope for the Loupe probability mask. Larger slopes make the mask converge faster to
                           deterministic state.
            sparsity (float): Predefined sparsity of the learned probability mask. 1 / acceleration_ratio
            line_constrained (bool): Sample kspace measurements column by column
            conjugate_mask (bool): For real image, the corresponding kspace measurements have conjugate symmetry property
                (point reflection). Therefore, the information in the left half of the kspace image is the same as the
                other half. To take advantage of this, we can force the model to only sample right half of the kspace
                (when conjugate_mask is set to True)
            preselect: preselect center regions  
        """
        super().__init__()

#         assert conjugate_mask is False

        # probability mask
        self.gen_mask = LineConstrainedProbMask(shape, slope, preselect=preselect, preselect_num=preselect_num)

        self.rescale  = RescaleProbMap
        self.binarize = ThresholdRandomMaskSigmoidV1.apply # FIXME

        self.preselect = preselect
        self.preselect_num = preselect_num
        self.preselect_num_one_side = preselect_num // 2
        self.shape = shape

    def _mask_neg_entropy(self, mask, eps=1e-10):
        # negative of pixel wise entropy
        entropy = mask * torch.log(mask+eps) + (1-mask) * torch.log(1-mask+eps)
        return entropy

    def forward(self, kspace, sparsity):
        # kspace: NCHW
        # sparsity (float)
        prob_mask = self.gen_mask(kspace)
        if self.preselect:
            rescaled_mask = self.rescale(prob_mask, sparsity - self.preselect_num/self.shape[0])
        else:
            rescaled_mask = self.rescale(prob_mask, sparsity)

        binarized_mask = self.binarize(rescaled_mask)
        binarized_mask[..., :self.preselect_num_one_side , :] = 1  # wrt unrolled masks
        binarized_mask[..., -self.preselect_num_one_side:, :] = 1
        binarized_mask.retain_grad()

#         neg_entropy = self._mask_neg_entropy(rescaled_mask)
#         masked_kspace = binarized_mask * kspace
        binarized_mask = torch.tile(torch.tile(binarized_mask,(1,1,1,self.shape[1])),(kspace.shape[0],1,1,1))
#         kspace[:,:,binarized_mask==0,:] = 0
        masked_kspace  = torch.mul(binarized_mask , kspace) 
#         data_to_vis_sampler = {'prob_mask': transforms.fftshift(prob_mask[0,:,:,0],dim=(0,1)).cpu().detach().numpy(), 
#                                'rescaled_mask': transforms.fftshift(rescaled_mask[0,:,:,0],dim=(0,1)).cpu().detach().numpy(), 
#                                'binarized_mask': transforms.fftshift(binarized_mask[0,:,:,0],dim=(0,1)).cpu().detach().numpy()}
        return masked_kspace, binarized_mask 
#         return masked_kspace, binarized_mask, neg_entropy, data_to_vis_sampler 