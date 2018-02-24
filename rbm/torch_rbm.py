"""
when comparing with the numpy version, the random method in 
sample and reset_parameters methods should be replaced.
"""
import math
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
    
def calc_mse(v_input, v_probs):
    temp = v_input - v_probs
    return torch.mean(temp * temp)

def calc_cross_entropy(v_input, v_probs):
    cross_entropy = torch.mean(
                v_input *  torch.log(v_probs) +
                (1 - v_input) * torch.log(1 - v_probs)
            )
    return cross_entropy
    
class ReconstructItem(object):
    def __init__(self, v, v_recon):
        self._v = v
        self._v_recon = v_recon
      
    @property  
    def v(self):
        return self._v.numpy()
    
    @property
    def v_recon(self):
        return self._v_recon.numpy()
    
    @property
    def cross_entropy(self):
        return calc_cross_entropy(self._v, self._v_recon)    
    
    @property
    def mse(self):
        return calc_mse(self._v, self._v_recon)

class ProbabilityMappingUnit(object):
    def __init__(self, weight : Variable, bias : Variable):
        self.weight = weight
        self.bias = bias
        
    def propgate(self, input : torch.Tensor):
        probs = 1/(1 + np.exp(-(input.numpy() @ self.weight.data.numpy().T + self.bias.data.numpy()))) # Wv + a
        return torch.Tensor(probs)
#        return torch.sigmoid(input @ self.weight.data.transpose(0, 1) + self.bias.data)
        
    def propgate_variable(self, input : Variable):
        #linear： y = [ W @ x + b for x in input]
        return F.sigmoid(F.linear(input, self.weight, self.bias))
    
    def sample(self, input : torch.Tensor):
        probs = self.propgate(input)
        # 1 to compare with numpy
#        temp = np.random.binomial(1, probs)
#        sample = torch.Tensor(temp)
        # 2
        sample = torch.bernoulli(probs)
        # 3
#        sample = torch.zeros_like(probs)
#        sample[probs > torch.rand(probs.shape)] = 1
        return probs, sample
    
    def sample_variable(self, input : Variable):
        probs = self.propgate_variable(input)
        sample = torch.bernoulli(probs)
        return probs, Variable(sample)
    
class BinaryBinaryRBM(nn.Module):
    def __init__(self, n_visible = 0, n_hidden = 0):     
        super(BinaryBinaryRBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # look out this order, the input data of all DL frameworks is element-index-in-batch
        # For PyTorch Linear op, [ Wx for x in X ] -> X @ W.T
        self.W = Parameter(torch.Tensor(n_hidden, n_visible))
        self.h_bias = Parameter(torch.Tensor(n_hidden)) # a
        self.v_bias = Parameter(torch.Tensor(n_visible)) # b
        
        self.v2h_mapping = ProbabilityMappingUnit(self.W, self.h_bias)
        self.h2v_mapping = ProbabilityMappingUnit(self.W.transpose(0, 1), self.v_bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 4 * math.sqrt(6. / (self.n_hidden + self.n_visible))
        # 1 to be comparable with numpy version
#        size = self.n_visible * self.n_hidden
#        self.W.data[:] = torch.Tensor(np.random.uniform(-stdv, stdv, size).reshape((self.n_visible, self.n_hidden)).T)
        # 2
        self.W.data.uniform_(-stdv, stdv)
        self.v_bias.data.zero_()
        self.h_bias.data.zero_()
        
    def sample_h_given_v(self, v : torch.Tensor):
        return self.v2h_mapping.sample(v)
    
    def gibbs_hvh(self, hb1_sample : torch.Tensor):
        v_probs, v_sample = self.h2v_mapping.sample(hb1_sample)
        h_probs, h_sample = self.v2h_mapping.sample(v_sample)
        return v_probs, v_sample, h_probs, h_sample
                                
    def reconstruct(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.Tensor(v)
        h_probs = self.v2h_mapping.propgate(v)
        v_recon = self.h2v_mapping.propgate(h_probs)
        return ReconstructItem(v, v_recon)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'
        
    def fit(self, data, batch_size, shuffle = False, 
            num_epoch = 50, lr = 0.1, momentum  = 0.9, k = 1):
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
            
    #    optimizer = MomentumOptimizer(rbm.parameters(), lr = lr, momentum = momentum)
        optimizer = optim.SGD(self.parameters(), lr = lr, momentum = momentum)
        
        criterion = CDkLoss(self, k = k)
        
        shuffle = False
        train_data = DataLoader(dataset = torch.Tensor(data),
                            batch_size = batch_size, 
                            shuffle = shuffle)
        
        costs = []
        for epoch in range(num_epoch):
            temp = []
            for batch_data in train_data:
                optimizer.zero_grad()  # Be Careful, grad will be added defaultly
                loss = criterion(batch_data)
                loss.backward()
                optimizer.step()
                temp.append(loss.data[0])
            cost = np.mean(temp)
            costs.append(cost)
            print(epoch, cost)
        return costs

class MulGradFunction(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, tensor, grad):
        """
        Tensor in and Tensor out.
        not Variable!
        """
        ctx.save_for_backward(grad)
        return torch.Tensor([0])
   
    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        if grad_output.data[0] != 1:
            grad = grad.mul(grad_output)
        return Variable(grad), None

mul_grad = MulGradFunction.apply
    
from torch.nn.modules.loss import _Loss

class RBMLoss(_Loss):
    def __init__(self, model):
        super(RBMLoss, self).__init__()
        self.model = model    

class CDkLoss(RBMLoss):
    def __init__(self, model, k = 1):
        super(CDkLoss, self).__init__(model)
        self.k = k
        
    def forward(self, v_input : torch.Tensor): 
        h0_probs, hn_sample = self.model.sample_h_given_v(v_input)
        for n in range(1, self.k+1):
            vn_probs, vn_sample, hn_probs, hn_sample = self.model.gibbs_hvh(hn_sample)
          
        W_grad = (hn_probs.transpose(0, 1) @ vn_sample \
                  - (h0_probs.transpose(0, 1) @ v_input)) / len(v_input)
        v_bias_grad =  torch.sum(vn_sample - v_input, dim = 0) / len(v_input)
        h_bias_grad = torch.sum(hn_probs - h0_probs, dim = 0) / len(v_input)
                
        # here we can used
#        self.model.W.grad = Variable(W_grad)
#        self.model.v_bias.grad = Variable(v_bias_grad)
#        self.model.h_bias.grad = Variable(h_bias_grad)
#        loss = Variable(torch.Tensor([0]))
        # we just to show the usage of autograd, and the power of pytorch.
        loss = mul_grad(self.model.W, Variable(W_grad)) \
            + mul_grad(self.model.v_bias, Variable(v_bias_grad))  \
            + mul_grad(self.model.h_bias, Variable(h_bias_grad))
        loss.data[0] = calc_mse(v_input, vn_probs)
        
        return loss
    
from torch.optim import Optimizer
    
class MomentumOptimizer(Optimizer):
    '''
    Sutskever et. al. SDG is different from pytorch SGD momentum, see the document of pytorch SGD.
    '''
    def __init__(self, params, lr = 0.1, momentum = 0.5):
        defaults = dict(lr = lr, momentum = momentum)
        if momentum < 0 :
            raise ValueError("momentum requires a positive value")
        super(MomentumOptimizer, self).__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'step_buffer' not in param_state:
                        buf = param_state['step_buffer'] = torch.zeros_like(p.data)
                        buf.add_(-lr, d_p)
                    else:
                        buf = param_state['step_buffer']
                        buf.mul_(momentum).add_(-lr, d_p)
                else:
                    buf = -lr * d_p

                p.data.add_(buf)

        return loss