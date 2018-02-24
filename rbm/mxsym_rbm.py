import math 
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import sym
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
    
class RBM(object):
    def __init__(self): 
        pass

class BinaryBinaryRBM(object):
    def __init__(self, n_visible = 0, n_hidden = 0, ctx = mx.cpu()):     
        super(BinaryBinaryRBM, self).__init__()
        self.ctx = ctx
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = mx.nd.empty((n_visible, n_hidden), dtype = 'float32', ctx = ctx)
        self.h_bias = mx.nd.empty((n_hidden, ), dtype = 'float32', ctx = ctx) # a
        self.v_bias = mx.nd.empty((n_visible, ), dtype = 'float32', ctx = ctx) # b
        
        self.W.attach_grad()
        self.h_bias.attach_grad()
        self.v_bias.attach_grad()
        
        self.graph = BinaryBinaryRBMGraph(self)
        
        self.init_op_reconstruct()
        
        self.reset_parameters()
        
        self.paras = {'W' : self.W,
                      'v_bias' : self.v_bias,
                      'h_bias' : self.h_bias}
    
    def reset_parameters(self):
        stdv = 4 * math.sqrt(6. / (self.n_hidden + self.n_visible))
        mx.nd.random_uniform(-stdv, stdv, out = self.W)
        self.v_bias[:] = 0
        mx.nd.zeros(self.h_bias.shape, out = self.h_bias)
        
    def init_op_reconstruct(self):
        self.op_reconstruct = self.graph.get_op_reconstruct()
                        
    def reconstruct(self, v):
        if not isinstance(v, nd.NDArray):
            v = nd.array(v)
        paras = {'X' : v}
        paras.update(self.paras)
        e = self.op_reconstruct.bind(mx.cpu(),
                                     paras,                                     
                                     grad_req = 'null')
        v_recon  = e.forward()
#        v_recon = self.op_reconstruct.eval(ctx = mx.cpu(), **paras)
        return ReconstructItem(v, v_recon[0])
    
    def calc_cross_entropy(self, v):
        v_recon = self.reconstruct(v)
        return v_recon.cross_entropy
    
    def calc_mse(self, v):
        v_recon = self.reconstruct(v)
        return v_recon.mse
    
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
    
def sample_bernoulli(probs, shape): 
    return sym.relu(sym.sign(probs - sym.random_uniform(shape = shape)))

def sample_gaussian(x, scale, shape):
    return x + sym.random_normal(loc = 0.0, shape = shape, scale = scale)

def mx_xavier_init(fan_in, fan_out, *, const = 4.0, dtype = 'float32'):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return sym.random_uniform((fan_in, fan_out), low = -k, high = k, dtype = dtype)
    
class BinaryBinaryRBMGraph(object):
    def __init__(self, model : BinaryBinaryRBM):  
        self.model = model
        
        self.X = sym.Variable('X')

        self.n_visible = model.n_visible
        self.n_hidden = model.n_hidden
        self.W = sym.Variable('W', dtype = 'float32')
        self.v_bias = sym.Variable('v_bias', dtype = 'float32')
        self.h_bias = sym.Variable('h_bias', dtype = 'float32')

    def get_op_reconstruct(self):
        h_probs = self.get_op_propgate_v2h(self.X)
        return self.get_op_propgate_h2v(h_probs)
                            
    def get_op_propgate_v2h(self, v):
        h_probs = sym.sigmoid(sym.broadcast_add(sym.dot(v, self.W), self.h_bias))
        return h_probs
    
    def get_op_sample_h_given_v(self, v, row_cnt):
        h_probs = self.get_op_propgate_v2h(v)
        h_sample = sample_bernoulli(h_probs, [row_cnt, self.n_hidden])
        return h_probs, h_sample   
      
    def get_op_propgate_h2v(self, h):
        v_probs = sym.sigmoid(sym.broadcast_add(sym.dot(h, sym.transpose(self.W)), 
                                                self.v_bias))
        return v_probs 
    
    def get_op_sample_v_given_h(self, h, row_cnt):
        v_probs = self.get_op_propgate_h2v(h)
        v_sample = sample_bernoulli(v_probs, [row_cnt, self.n_visible])
        return v_probs, v_sample               
    
    def get_op_gibbs_hvh(self, hb1_sample, row_cnt):
        v_probs, v_sample = self.get_op_sample_v_given_h(hb1_sample, row_cnt)
        h_probs, h_sample = self.get_op_sample_h_given_v(v_sample, row_cnt)
        return v_probs, v_sample, h_probs, h_sample

class RBMLoss(object):
    def __init__(self, model):
        self.model = model  
        
    def __call__(self, v_input):
        return self.forward(v_input)  

class CDkLoss(RBMLoss):
    def __init__(self, model, k = 1):
        self.model = model
        self.ctx = model.ctx
        self.k = k
        # MxNet not support symbolic shape currently
        # https://github.com/apache/incubator-mxnet/issues/4009
        # The cache is used here as a workaround
        self.ops = {}
        
    def get_op(self, row_cnt): 
        if row_cnt in self.ops:
            return self.ops[row_cnt]
        graph = self.model.graph
        X = graph.X       
        # build the graph
        h0_probs, hn_sample = graph.get_op_sample_h_given_v(X, row_cnt)
        for n in range(1, self.k+1):
            vn_probs, vn_sample, hn_probs, hn_sample = graph.get_op_gibbs_hvh(hn_sample, row_cnt)
        
        # gradients
        grad_W = (sym.dot(sym.transpose(vn_sample), hn_probs) - 
                  sym.dot(sym.transpose(X), h0_probs))
        grad_v_bias = sym.sum(vn_sample - X, axis = 0)
        grad_h_bias = sym.sum(hn_probs - h0_probs, axis = 0)
        
        # loss
        loss = sym.mean(sym.square(X - vn_probs))
        op = sym.Group([loss, grad_W, grad_v_bias, grad_h_bias])
        self.ops[row_cnt] = op
        return op
        
    def forward(self, v_input):
        op= self.get_op(v_input.shape[0])
        loss, grad_W, grad_v_bias, grad_h_bias = op.eval(ctx = self.ctx, 
                                                           X = v_input,
                                                           grad_req = 'null',
                                                           **self.model.paras)
        grad_W.copyto(self.model.W.grad)
        grad_v_bias.copyto(self.model.v_bias.grad)
        grad_h_bias.copyto(self.model.h_bias.grad)
        return loss.asscalar()
        
class Optimizer(object):
    pass

class MomentumOptimizer(Optimizer):
    def __init__(self, model, lr = 0.1, momentum = 0.5):
        self.model = model
        self.momentum  = momentum
        self.lr  = lr
        self.dW = nd.zeros_like(model.W, ctx = model.ctx)
        self.dv = nd.zeros_like(model.v_bias, ctx = model.ctx)
        self.dh = nd.zeros_like(model.h_bias, ctx = model.ctx)
        
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