import math
import torch
from torch.optim import Optimizer, SGD
from torch.nn import CrossEntropyLoss, MSELoss
import pickle as pkl
import os
from tqdm import tqdm
from torch.nn.functional import softmax
import logging

class Weight_Regularized_AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1]  < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(Weight_Regularized_AdamW, self).__init__(params, defaults)
        self.model = None

    def set_model(self, model):
        self.model = model

    def step(self, lamda, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        reg_params = self.model.reg_params
        if closure is not None:
            loss = closure()

        reg_lambda = lamda #reg_params.get('lambda')

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                #Regularize PART CODE GOES HERE
                if p in reg_params:

                    reg_param=reg_params.get(p)
                    #get omega for this parameter
                    omega=reg_param.get('omega')
                    #initial value when the training start
                    init_val=reg_param.get('init_val')
                    curr_weight_val=p.data

                    #print("Curr Weight",curr_weight_val)
                    #get the difference
                    weight_dif=curr_weight_val.add(-1,init_val)
                    #compute the MAS penalty
                    regulizer=weight_dif.mul(reg_lambda*omega)
                    #print("Regularize", regulizer)
                    del weight_dif
                    del curr_weight_val
                    del omega
                    del init_val
                    #add the MAS regulizer to the gradient
                    # grad.add_(regulizer)
                    p.data.add_(-group['lr'], regulizer)
                    #print("P data",p.data)
                    del regulizer
                #Regularize PART CODE ENDS
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss
