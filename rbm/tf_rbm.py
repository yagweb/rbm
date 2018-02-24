import math 
import numpy as np
import tensorflow as tf
from .data import DataLoader
from .np_rbm import ReconstructItem

def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)

def tf_xavier_init(fan_in, fan_out, *, const = 4.0, dtype=np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)
    
class RBM(object):
    def __init__(self): 
        pass
    
class BinaryBinaryRBM(RBM):
    def __init__(self, n_visible = 0, n_hidden = 0):     
        super(BinaryBinaryRBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Input data
        self.X = tf.placeholder(tf.float32, [None, self.n_visible])

        # variable with initial value
        self.W = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden), dtype=tf.float32)
        self.v_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.h_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
        
        # cache for gradient
        self.grad_W = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.grad_v_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.grad_h_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
        
        self.init_op_reconstruct()
        
        self.sess = tf.Session()
        self.reset_parameters()
        
    def reset_parameters(self):
        # tf Variable has init value bind
        init = tf.global_variables_initializer()
        self.sess.run(init)
            
    def get_op_propgate_v2h(self, v):
        h_probs = tf.nn.sigmoid(tf.matmul(v, self.W) + self.h_bias)
        return h_probs
    
    def get_op_sample_h_given_v(self, v):
        h_probs = self.get_op_propgate_v2h(v)
        h_sample = sample_bernoulli(h_probs)
        return h_probs, h_sample   
      
    def get_op_propgate_h2v(self, h):
        v_probs = tf.nn.sigmoid(tf.matmul(h, 
                                tf.transpose(self.W)) + self.v_bias)
        return v_probs 
    
    def get_op_sample_v_given_h(self, h):
        v_probs = self.get_op_propgate_h2v(h)
        v_sample = sample_bernoulli(v_probs)
        return v_probs, v_sample               
    
    def get_op_gibbs_hvh(self, hb1_sample):
        v_probs, v_sample = self.get_op_sample_v_given_h(hb1_sample)
        h_probs, h_sample = self.get_op_sample_h_given_v(v_sample)
        return v_probs, v_sample, h_probs, h_sample
    
    def init_op_reconstruct(self):
        h_probs = self.get_op_propgate_v2h(self.X)
        #return v_recon
        self.op_reconstruct = self.get_op_propgate_h2v(h_probs)
                                
    def reconstruct(self, v):
        if len(v.shape) == 1:
            v = v.reshape((1, v.size))
        v_recon,  = self.sess.run([self.op_reconstruct], feed_dict = {self.X: v})
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
        opt = MomentumOptimizer(criterion, lr, momentum)
        
        train_data = DataLoader(data, batch_size)
        
        costs = []
        for epoch in range(num_epoch):
            temp = []
            for batch_data in train_data:
                loss = opt.step(batch_data)
                temp.append(loss)
            mean_loss = np.mean(temp)
            costs.append(mean_loss)
            print(epoch, mean_loss)
        return costs

class CDkLoss(object):
    def __init__(self, model, k = 1):
        self.model = model
        self.X = model.X
        self.k = k  # it not changable during training
        
        # build the graph
        h0_probs, hn_sample = self.model.get_op_sample_h_given_v(self.X)
        for n in range(1, self.k+1):
            # backward + forward
            vn_probs, vn_sample, hn_probs, hn_sample = self.model.get_op_gibbs_hvh(hn_sample)
        
        self.grad_W = (tf.matmul(tf.transpose(vn_sample), hn_probs) - 
                  tf.matmul(tf.transpose(self.X), h0_probs)) / tf.to_float(tf.shape(self.X)[0])
        self.grad_v_bias = tf.reduce_mean(vn_sample - self.X, 0)
        self.grad_h_bias = tf.reduce_mean(hn_probs - h0_probs, 0)
        
        # for loss calculation
        self.vn_probs = vn_probs
        
class Optimizer(object):
    pass

class MomentumOptimizer(Optimizer):
    def __init__(self, loss, lr = 0.1, momentum = 0.5):
        self.loss = loss
        self.model = loss.model
        self.X = self.model.X
        self.sess = self.model.sess
        self.lr = lr
        self.momentum = momentum
        if momentum < 0 :
            raise ValueError("momentum requires a positive value")
        
        self.dW = tf.Variable(tf.zeros([self.model.n_visible, self.model.n_hidden]), dtype=tf.float32)
        self.dv_bias = tf.Variable(tf.zeros([self.model.n_visible]), dtype=tf.float32)
        self.dh_bias = tf.Variable(tf.zeros([self.model.n_hidden]), dtype=tf.float32) 
        
        def f(dp_old, p_grad):
            return self.momentum * dp_old - self.lr * p_grad
        
        update_dW = tf.assign(self.dW, f(self.dW, self.loss.grad_W), name="1")
        update_dv_bias = tf.assign(self.dv_bias, f(self.dv_bias, self.loss.grad_v_bias), name="2")
        update_dh_bias = tf.assign(self.dh_bias, f(self.dh_bias, self.loss.grad_h_bias), name="3")
        
        update_W = self.model.W.assign(self.model.W + self.dW)
        update_v_bias = self.model.v_bias.assign(self.model.v_bias + self.dv_bias)
        update_h_bias = self.model.h_bias.assign(self.model.h_bias + self.dh_bias)
        
        update_ops = [update_dW, update_dv_bias, update_dh_bias,
                           update_W, update_v_bias, update_h_bias]
        # tf.Print(self.dW, [self.dW])
        with tf.control_dependencies(update_ops):
            self.loss_op = tf.reduce_mean(tf.square(self.X - self.loss.vn_probs))

        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def step(self, input):
        loss = self.sess.run(self.loss_op, feed_dict={self.X: input})
        return loss
        