#

def create_rbm(n_visible, n_hidden, framework = 'np'):
    if framework == 'np':
        from .np_rbm import BinaryBinaryRBM
        return BinaryBinaryRBM(n_visible, n_hidden)
    elif framework == 'tf':
        from .tf_rbm import BinaryBinaryRBM
        return BinaryBinaryRBM(n_visible, n_hidden)
    elif framework == 'torch':
        from .torch_rbm import BinaryBinaryRBM
        return BinaryBinaryRBM(n_visible, n_hidden)
    elif framework == 'mxnd':
        from .mxnd_rbm import BinaryBinaryRBM
        return BinaryBinaryRBM(n_visible, n_hidden)
    elif framework == 'gluon':
        from .gluon_rbm import BinaryBinaryRBM
        return BinaryBinaryRBM(n_visible, n_hidden)
    elif framework == 'mxsym':
        from .mxsym_rbm import BinaryBinaryRBM
        return BinaryBinaryRBM(n_visible, n_hidden)