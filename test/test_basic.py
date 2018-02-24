"""
# Result

## 1 Parameters
    The choose of the Optimizer parameters is depend on the dataset.
    For this test, momentum = 0.9 is better than 0; momentum = 0 with larger lr also has the same effect.
    For test in test_debug.py, momentum = 0 is better than 0.5, 0.9 etc with the same lr.

## 2 Speed
    according to this test, time consumption when use cpu: 
    TF == PyTorch > numpy (MKL) > mxnet (Openblas)
    
    why ATen is faster than MKL, mxnet is so slow (except for the influence of Openblas)?

## 3 Others
    when use gluon, the convergence speed and accuracy is unreasonable good, why? 
"""
import sys
sys.path.append('..')

import time
import numpy as np
import matplotlib.pylab as plt
import rbm
 
def generate_data(N):
    T = N * 38
    u = np.mat(np.zeros((T, 20)))
    for i in range(1, T, 38):
        if i % 76 == 1:
            u[i - 1:i + 19, :] = np.eye(20)
            u[i + 18:i + 38, :] = np.eye(20)[np.arange(19, -1, -1)]
            u[i - 1:i + 19, :] += np.eye(20)[np.arange(19, -1, -1)] 
        else:
            u[i - 1:i + 19, 1] = 1
            u[i + 18:i + 38, 8] = 1
    return u

def test(framework = 'np'):
    '''
    https://github.com/benanne/morb/blob/master/examples/example_basic.py
    '''
    np.random.seed(0)
    # generate data
    data = generate_data(200)

    # paramters
    n_visible = data.shape[1]
    n_hidden = 100
    batch_size = 32 * 2
    num_epoch = 50
    
    model = rbm.create_rbm(n_visible, n_hidden, framework = framework)
    
    # Mean loss of morb decrease from 0.037 to 0.0024 after 50 steps.
    start = time.time()    
    
    costs = model.fit(data, batch_size, num_epoch = num_epoch, 
                lr = 0.1, momentum  = 0.9, k = 1)

    print("time elapsed per epoch: %.5f" % ((time.time() - start) / num_epoch))
    plt.plot(costs)
    plt.show()
    
    #############test##########
    #The mse of the train data of morb is about 1e-6 ~ 1e-9
     
    print("----random test-----")
    for i in range(5):
        v = np.array([np.random.randint(2) for bb in range(20)])
        item = model.reconstruct(v)
        print(i, item.mse, item.cross_entropy)
    print("----fixed test-----")
    for i in range(5):
        v = np.zeros([n_visible, ])
        v[i+1 : i+1+7] = 1
        item = model.reconstruct(v)
        print(i, item.mse, item.cross_entropy)
    print("----raw data-----")
    for i in range(10):
        v = data[i]
        item = model.reconstruct(v)
        print(i, item.mse, item.cross_entropy)

if __name__ == "__main__":
    test('np')
    test('tf')
    test('torch')
    test('gluon')
    test('mxnd')
    test('mxsym')
