"""
RBM base on pure numpy.
"""
import numpy as np
from numpy import exp
from numpy import random
from .data import DataLoader

def sigmoid(x):
    return 1/(1 + exp(-x))
    
def calc_mse(v_input, v_probs):
    temp = v_input - v_probs
    return np.average(np.multiply(temp, temp))

def calc_cross_entropy(v_input, v_probs):
    cross_entropy = np.average(np.sum(
                np.multiply(v_input,  np.log(v_probs)) +
                np.multiply((1 - v_input), np.log(1 - v_probs))
            ))
    return cross_entropy
    
class ReconstructItem(object):
    def __init__(self, v, v_recon):
        self.v = v
        self.v_recon = v_recon
    
    @property
    def cross_entropy(self):
        return calc_cross_entropy(self.v, self.v_recon)    
    
    @property
    def mse(self):
        return calc_mse(self.v, self.v_recon)

class RBM(object):
    def __init__(self):
        pass        

class ProbabilityMappingUnit(object):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        
    def propgate(self, input):
        probs = sigmoid(input @ self.weight + self.bias) # Wv + a
        return probs
            
    def sample_(self, input):
        probs = self.propgate(input)
        sample = np.zeros(probs.shape)
        sample[probs > random.rand(probs.size).reshape(probs.shape)] = 1
        return probs, sample
            
    def sample(self, input):
        probs = self.propgate(input)
        sample = random.binomial(1, probs)
        return probs, sample
    
class BinaryBinaryRBM(RBM):
    def __init__(self, n_visible = 0, n_hidden = 0):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.ndarray((n_visible, n_hidden), dtype=np.float32)
        self.h_bias = np.ndarray((1, n_hidden), dtype=np.float32) #a
        self.v_bias = np.ndarray((1, n_visible), dtype=np.float32) #b
        self.reset_parameters()
        
        self.v2h_mapping = ProbabilityMappingUnit(self.W, self.h_bias)
        self.h2v_mapping = ProbabilityMappingUnit(self.W.T, self.v_bias)
        
        # for Loss backward and Optimizer
        self.grad_W = None
        self.grad_v_bias = None
        self.grad_h_bias = None
        
    def reset_parameters(self):
        a = 4 * np.sqrt(6. / (self.n_hidden + self.n_visible))
        self.W[:] = random.uniform(-a, a, self.n_visible*self.n_hidden).reshape(self.W.shape)
        self.v_bias[:] = 0
        self.h_bias[:] = 0
        
    def sample_h_given_v(self, v):
        return self.v2h_mapping.sample(v)
    
    def gibbs_hvh(self, hb1_sample):
        v_probs, v_sample = self.h2v_mapping.sample(hb1_sample)
        h_probs, h_sample = self.v2h_mapping.sample(v_sample)
        return v_probs, v_sample, h_probs, h_sample
                                
    def reconstruct(self, v):
        h_probs = self.v2h_mapping.propgate(v)
        v_recon = self.h2v_mapping.propgate(h_probs)
        return ReconstructItem(v, v_recon)
    
    def calc_cross_entropy(self, v):
        v_recon = self.reconstruct(v)
        return v_recon.cross_entropy
    
    def calc_mse(self, v):
        v_recon = self.reconstruct(v)
        return v_recon.mse
    
    def fit(self, data, batch_size, shuffle = False, 
            num_epoch = 50, lr = 0.1, momentum  = 0.9, k = 1):
        criterion = CDkLoss(self, k)
        opt = MomentumOptimizer(self, lr, momentum)
        
        train_data = DataLoader(data, batch_size)
        
        costs = []
        for epoch in range(num_epoch):
            temp = []
            for batch_data in train_data:
                loss = criterion(batch_data)
                opt.step()
                temp.append(loss)
            mean_loss = np.mean(temp)
            costs.append(mean_loss)
            print(epoch, mean_loss)
        return costs

class Loss(object):
    def __init__(self, model):
        self.model = model
        
    def __call__(self, v_input):
        return self.forward(v_input)
        
    def forward(self, v_input):
        pass

class PCDkLoss(Loss):
    def __init__(self, model, k = 1):
        super(PCDkLoss, self).__init__(model)
        self.k = k
    
    def backward(self):
        # https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/rbm.py    
        v_input = self.v_input
        raise Exception('Not implement')
        
class CDkLoss(Loss):
    def __init__(self, model, k = 1):
        super(CDkLoss, self).__init__(model)
        self.k = k
        
    def forward(self, v_input): 
        h0_probs, hn_sample = self.model.sample_h_given_v(v_input)
        for n in range(1, self.k+1):
            # backward + forward
            vn_probs, vn_sample, hn_probs, hn_sample = self.model.gibbs_hvh(hn_sample)

        ##############
        
        # DL frameworks need the loss be minimumn
        # For consistency, the grads are opposite to the fomulas in paper
        grad_W = (vn_sample.T @ hn_probs - v_input.T @ h0_probs) / len(v_input)
        grad_v = np.average(vn_sample - v_input, axis = 0)
        grad_h = np.average(hn_probs - h0_probs, axis = 0)

        ################
            
#        loss = calc_cross_entropy(v_input, vn_probs)
        loss = calc_mse(v_input, vn_probs)
        
        self.model.grad_W = grad_W
        self.model.grad_v_bias = grad_v
        self.model.grad_h_bias = grad_h
        return loss
            
class Optimizer(object):
    def step(self):
        pass
    
class MomentumOptimizer(Optimizer):
    def __init__(self, model, lr = 0.1, momentum = 0.5, penalty = 0):
        self.model = model
        self.momentum  = momentum
        self.lr  = lr
        self.penalty = penalty
        self.dW = np.zeros(model.W.shape)
        self.dv = np.zeros(model.v_bias.shape)
        self.dh = np.zeros(model.h_bias.shape)
        
    def step(self):
        momentum = self.momentum
        lr = self.lr
        
        if momentum != 0:
            self.dW = momentum * self.dW - lr * self.model.grad_W
            self.dv = momentum * self.dv - lr * self.model.grad_v_bias
            self.dh = momentum * self.dh - lr * self.model.grad_h_bias
        else:
            self.dW =  - lr * self.model.grad_W
            self.dv =  - lr * self.model.grad_v_bias
            self.dh =  - lr * self.model.grad_h_bias
        
        # in_place add
        self.model.W += self.dW
        self.model.v_bias += self.dv
        self.model.h_bias += self.dh  
