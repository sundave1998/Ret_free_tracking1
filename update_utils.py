import numpy as np


def proj_tangent(x, differential):
    d = differential.shape[1]
    r = differential.shape[0]
    W_hat = (np.eye(r) - 0.5*x@x.T)@differential@x.T
    
    gradf = (W_hat - W_hat.T)@x
    return gradf
    
    
    xd = x.T@differential
    return differential - 0.5* x@(xd + xd.T)
    # return differential - 0.5* x@(x.T@differential + differential.T@x)
    
def data_gen_pca(N, n, m):
    A = np.zeros((N, n, m))
    for i in range(N):
        A[i] = np.random.randn(n,m)
    return A