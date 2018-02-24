import sys
sys.path.append('..')

import time
import numpy as np
import matplotlib.pylab as plt
import rbm
    
def test(framework):
    np.random.seed(0)
    
    train_X = np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0]
     ])
    
    test_X = np.array([
        [1, 1, 1, 0, 0, 0],  # exist
        [0, 0, 1, 1, 1, 0],  # exist
        [1, 1, 0, 0, 0, 0],  # not exist
        [0, 0, 0, 1, 1, 0]   # not exist
     ])
    
    # model
    n_visible = 6
    n_hidden = 3
    model = rbm.create_rbm(n_visible, n_hidden, framework = framework)
    batch_size = 7
    costs = model.fit(train_X, batch_size, num_epoch = 100, 
                lr = 0.1, momentum  = 0.9, k = 1)

    plt.plot(costs)
    plt.show()
            
    for i in range(len(test_X)):
        v = test_X[i, :]
        item = model.reconstruct(v)
        print(item.mse, item.cross_entropy, v, item.v_recon)

if __name__ == "__main__": 
    test('np')
    test('tf')
    test('torch')
    test('gluon')
    test('mxnd')
    test('mxsym')
