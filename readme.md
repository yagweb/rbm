# rbm
Binary Binary Restricted Boltzmann Machines with 6 different implementations:
    1) pure numpy
    2) pytorch
    3) tensorflow
    4) mxnet ndarray
    5) mxnet symbolic API
    6) mxnet gluon

Some code is a little overdesigned, because I want this project to help me learn and evaluate these deep learning frameworks.

CDk is used in the training, PDk can be added easily.
Other RBMs such as Binary-Gaussian RBM can also be extended easily.

## Theory:
http://www.deeplearning.net/tutorial/rbm.html#rbm

## Code Ref:
http://www.cs.toronto.edu/~hinton/code/rbm.m
It should be noticed that, the CD-k steps we used and given in most papers is a little different from the hinton's version.
Forward step and gradient fomula both has some changes. The results are the same.
  
## Comparison
basic example in https://github.com/benanne/morb/blob/master/morb/stats.py

## Usage

### create model
```python
n_visible = 6
n_hidden = 3
model = rbm.create_rbm(n_visible, n_hidden, framework = 'np')

model = rbm.create_rbm(n_visible, n_hidden, framework = 'tf')

model = rbm.create_rbm(n_visible, n_hidden, framework = 'torch')

model = rbm.create_rbm(n_visible, n_hidden, framework = 'gluon')

model = rbm.create_rbm(n_visible, n_hidden, framework = 'mxnd')

model = rbm.create_rbm(n_visible, n_hidden, framework = 'mxsym')
```

### train
```python
batch_size = 7
costs = model.fit(train_X, batch_size, num_epoch = 100, 
            lr = 0.1, momentum  = 0.9, k = 1)

plt.plot(costs)
plt.show()
```

### forward 
```python   
item = model.reconstruct(v)
print(item.mse, item.cross_entropy, v, item.v_recon)
```

## *License*
[MIT License]()     