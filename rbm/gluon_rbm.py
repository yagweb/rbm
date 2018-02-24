import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
   
def calc_mse(v_input : nd.NDArray, v_probs : nd.NDArray):
    temp = v_input - v_probs
    return mx.nd.mean(temp * temp).asscalar()

def calc_cross_entropy(v_input, v_probs):
    cross_entropy = mx.nd.mean(
                v_input *  mx.nd.log(v_probs) +
                (1 - v_input) * mx.nd.log(1 - v_probs)
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
    def __init__(self, weight : gluon.Parameter, bias : gluon.Parameter):
        self.weight = weight
        self.bias = bias
        
    def propgate(self, input):
        probs = nd.sigmoid(nd.dot(input, self.weight.data()) + self.bias.data()) # Wv + a
        return probs
            
    def sample(self, input):
        probs = self.propgate(input) 
        # look out! 
        # it was different from numpy, no mask usage
        # multinomial is hard to use here
        sample = probs > mx.random.uniform(0, 1, shape = probs.shape)
        return probs, sample

class ProbabilityMappingUnitTranspose(ProbabilityMappingUnit):
    def __init__(self, weight : gluon.Parameter, bias : gluon.Parameter):
        super(ProbabilityMappingUnitTranspose, self).__init__(weight, bias)
        
    def propgate(self, input):
        probs = nd.sigmoid(nd.dot(input, self.weight.data().T) + self.bias.data()) # Wv + a
        return probs

@mx.init.register
class Xavier(mx.init.Initializer):
    def __init__(self, const = 4, magnitude = 3):
        super(Xavier, self).__init__(const = const,
                                     magnitude=magnitude)
        self.const = float(const)
        self.magnitude = float(magnitude)

    def _init_weight(self, name, arr):
        shape = arr.shape
        fan_in, fan_out = shape[1], shape[0]
        scale = self.const * np.sqrt(self.magnitude / fan_in + fan_out)
        mx.random.uniform(-scale, scale, out = arr)
            
class BinaryBinaryRBM(nn.Block):
    def __init__(self, n_visible = 0, n_hidden = 0, ctx = mx.cpu(), **kwargs):     
        super(BinaryBinaryRBM, self).__init__(**kwargs)
        self.ctx = ctx
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        with self.name_scope():
            self.W = self.params.get('W', grad_req='write', 
                                     # 192 = 4**2 * 6 * 2
                                     # call self.W.initialize method
                                     #init = mx.init.Xavier(factor_type = 'avg', magnitude = 192),
                                     init = Xavier(),
                                     shape = (n_visible, n_hidden))
            self.h_bias = self.params.get('h_bias', grad_req='write', 
                                     init = mx.init.Zero(), shape = (n_hidden, )) # a
            self.v_bias = self.params.get('v_bias', grad_req='write', 
                                     init = mx.init.Zero(), shape = (n_visible, )) # b        
        
        self.reset_parameters()
        
        self.v2h_mapping = ProbabilityMappingUnit(self.W, self.h_bias)
        self.h2v_mapping = ProbabilityMappingUnitTranspose(self.W, self.v_bias)
        
    def reset_parameters(self, ctx = mx.cpu()):
        self.collect_params().initialize(ctx=ctx)
        
    def sample_h_given_v(self, v : nd.NDArray):
        with v.context:
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
                        
        # use fomula to calc the grads
        nd.elemwise_sub(nd.dot(vn_sample.T, hn_probs), nd.dot(v_input.T, h0_probs), out = self.model.W.grad())
        nd.sum(vn_sample - v_input, axis = 0, out = self.model.v_bias.grad())
        nd.sum(hn_probs - h0_probs, axis = 0, out = self.model.h_bias.grad())  
        for param in self.model.params:
            # important, tell the optimizer the grad is updated
            self.model.params[param].data()._fresh_grad = True
        
        loss = calc_mse(v_input, vn_probs)
        
        return loss

@mx.optimizer.register
class Momentum(mx.optimizer.Optimizer):
    def __init__(self, learning_rate = 0.01, momentum = 0.9, **kwargs):
        super(Momentum, self).__init__(learning_rate = learning_rate, **kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = nd.zeros(weight.shape, weight.context, dtype=weight.dtype, stype=weight.stype)
        return momentum

    def _update_impl(self, index, weight, grad, state):
        assert(isinstance(weight, nd.NDArray))
        assert(isinstance(grad, nd.NDArray))
        self._update_count(index)
        lr = self._get_lr(index) * self.rescale_grad

        if state is not None:
            state *= self.momentum
            nd.elemwise_add(- lr * grad, state, out = state)
            nd.elemwise_add(weight, state, out = weight)
        else:
            nd.elemwise_add(weight, - lr * grad, out = weight)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state)
    
def MomentumOptimizer(model, lr = 0.01, momentum = 0.9, name = 'momentum'):
#    name = 'sdg'  # It also works, try it.
    return gluon.Trainer(model.collect_params(), 
        name, 
         {
                'learning_rate': lr, 
                'momentum': momentum
         })