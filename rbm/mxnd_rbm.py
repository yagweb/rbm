"""
It should be noticed that, the loss definition of mxnet is different from pytorch and tensorflow. 
mxnet use Sum while the others use Mean. So the step method of mxnet need a parameter batch_size.
"""
import math
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
    
def calc_mse(v_input : nd.NDArray, v_probs : nd.NDArray):
    temp = v_input - v_probs
    return nd.mean(temp * temp).asscalar()

def calc_cross_entropy(v_input, v_probs):
    cross_entropy = nd.mean(
                v_input *  nd.log(v_probs) +
                (1 - v_input) * nd.log(1 - v_probs)
            )
    return cross_entropy.asscalar()
    
class ReconstructItem(object):
    def __init__(self, v, v_recon):
        self._v = v
        self._v_recon = v_recon
      
    @property  
    def v(self):
        return self._v.asnumpy()
    
    @property
    def v_recon(self):
        return self._v_recon.asnumpy()
    
    @property
    def cross_entropy(self):
        return calc_cross_entropy(self._v, self._v_recon)    
    
    @property
    def mse(self):
        return calc_mse(self._v, self._v_recon)

class ProbabilityMappingUnit(object):
    def __init__(self, weight : nd.NDArray, bias : nd.NDArray):
        self.weight = weight
        self.bias = bias
        
    def propgate(self, input):
        probs = nd.sigmoid(nd.dot(input, self.weight) + self.bias) # Wv + a
        return probs
            
    def sample(self, input):
        probs = self.propgate(input) 
        # look out! 
        # it was different from numpy, no mask usage
        # multinomial is hard to use here
        sample = probs > mx.random.uniform(0, 1, shape = probs.shape)
        return probs, sample
    
class BinaryBinaryRBM(object):
    def __init__(self, n_visible = 0, n_hidden = 0, ctx = mx.cpu()):
        self.ctx = ctx
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = mx.nd.empty((n_visible, n_hidden), ctx = ctx)
        self.h_bias = mx.nd.empty((n_hidden, ), ctx = ctx) # a
        self.v_bias = mx.nd.empty((n_visible, ), ctx = ctx) # b
        
        self.W.attach_grad()
        self.h_bias.attach_grad()
        self.v_bias.attach_grad()
        
        self.reset_parameters()
        
        self.v2h_mapping = ProbabilityMappingUnit(self.W, self.h_bias)
        self.h2v_mapping = ProbabilityMappingUnit(self.W.T, self.v_bias)
        
    def syn_weight(self):
        self.v2h_mapping.weight.transpose(out = self.h2v_mapping.weight)
        
    def reset_parameters(self):
        stdv = 4 * math.sqrt(6. / (self.n_hidden + self.n_visible))
        mx.nd.random_uniform(-stdv, stdv, out = self.W)
        self.v_bias[:] = 0
        mx.nd.zeros(self.h_bias.shape, out = self.h_bias)
        
    def sample_h_given_v(self, v : nd.NDArray):
        return self.v2h_mapping.sample(v)
    
    def gibbs_hvh(self, hb1_sample : nd.NDArray):
        v_probs, v_sample = self.h2v_mapping.sample(hb1_sample)
        h_probs, h_sample = self.v2h_mapping.sample(v_sample)
        return v_probs, v_sample, h_probs, h_sample
                                
    def reconstruct(self, v):
        if not isinstance(v, nd.NDArray):
            v = nd.array(v)
        h_probs = self.v2h_mapping.propgate(v)
        v_recon = self.h2v_mapping.propgate(h_probs)
        return ReconstructItem(v, v_recon)
    
    def fit(self, data, batch_size, shuffle = False, 
            num_epoch = 50, lr = 0.1, momentum  = 0.9, k = 1):
        if not isinstance(data, nd.NDArray):
            data = nd.array(data)
            
        optimizer = MomentumOptimizer(self, lr = lr, momentum = momentum)
        dataset = gluon.data.ArrayDataset(data)
        train_data = gluon.data.DataLoader(dataset,
                            batch_size = batch_size, 
                            shuffle = shuffle)
        
        criterion = CDkLoss(self, k = k)
        
        costs = []
        for epoch in range(num_epoch):
            temp = []
            for batch_data in train_data:
                loss = criterion(batch_data)
                optimizer.step(len(batch_data))
                temp.append(loss)
            mean_loss = np.mean(temp)
            costs.append(mean_loss)
            print(epoch, mean_loss)
        return costs
 
class RBMLoss(object):
    def __init__(self, model):
        self.model = model  
        
    def __call__(self, v_input):
        return self.forward(v_input)  

class CDkLoss(RBMLoss):
    def __init__(self, model, k = 1):
        super(CDkLoss, self).__init__(model)
        self.k = k
        
    def forward(self, v_input : nd.NDArray):  
        h0_probs, hn_sample = self.model.sample_h_given_v(v_input)
        for n in range(1, self.k+1):
            vn_probs, vn_sample, hn_probs, hn_sample = self.model.gibbs_hvh(hn_sample)
                        
        # calc gradients    
        nd.elemwise_sub(nd.dot(vn_sample.T, hn_probs), nd.dot(v_input.T, h0_probs), out = self.model.W.grad)
        nd.sum(vn_sample - v_input, axis = 0, out = self.model.v_bias.grad)
        nd.sum(hn_probs - h0_probs, axis = 0, out = self.model.h_bias.grad)
        
        loss = calc_mse(v_input, vn_probs)
        
        return loss
    
class Optimizer(object):
    pass
    
class MomentumOptimizer(Optimizer):
    def __init__(self, model, lr = 0.1, momentum = 0.5):
        self.model = model
        self.momentum  = momentum
        self.lr  = lr
        self.dW = nd.zeros_like(model.W)
        self.dv = nd.zeros_like(model.v_bias)
        self.dh = nd.zeros_like(model.h_bias)
        
    def step(self, batch_size):
        momentum = self.momentum
        lr = self.lr
                
        if momentum != 0:
            self.dW *= momentum
            nd.elemwise_add(- lr / batch_size * self.model.W.grad, self.dW, out = self.dW)
            
            self.dv *= momentum
            nd.elemwise_add(- lr / batch_size * self.model.v_bias.grad, self.dv, out = self.dv)
            
            self.dh *= momentum
            nd.elemwise_add(- lr / batch_size * self.model.h_bias.grad, self.dh, out = self.dh)
        else:
            self.dW = - lr / batch_size * self.model.W.grad
            self.dv = - lr / batch_size * self.model.v_bias.grad
            self.dh = - lr / batch_size * self.model.h_bias.grad
        
        self.model.W += self.dW
        self.model.v_bias += self.dv
        self.model.h_bias += self.dh
        
        self.model.syn_weight()