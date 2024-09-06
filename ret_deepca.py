import numpy as np
from update_utils import *
from tqdm import tqdm
import copy
import time

def SignAdjust(W, W_0):
    N = W_0.shape[0]
    n = W_0.shape[1]
    r = W_0.shape[2]
    
    for i in range(N):
        for j in range(r):
            if np.dot(W[i,:,j], W_0[i,:,j]) < 0:
                W[i,:,j] = - W[i,:,j]
    return W

def FastMix(X, W):
    new_X = np.zeros_like(X)
    N = X.shape[0]
    for i in range(N):
        for j in range(N):
            new_X[i] += W[i,j] * X[j]
    return new_X

def Cal_consensus_error(X):
    N = X.shape[0]
    n = X.shape[1]
    r = X.shape[2]
    X_bar = np.average(X, axis=0)
    consensus_error = 0
    for i in range(N):
        consensus_error += np.linalg.norm(X[i]- X_bar)
    return consensus_error
    
def DeePCA(A, graph_W, step_size, X_0, x_opt, max_iter=10000, lin_term=None, verbose=False):
    distance = []
    con_errors = []
    grad_norms = []
    N = X_0.shape[0]
    n = X_0.shape[1]
    r = X_0.shape[2]
    
    W = np.copy(X_0)
    S = np.copy(X_0)
    new_W = np.zeros_like(X_0)
    grad = np.zeros_like(X_0)
    # diff = np.zeros_like(X_0)
    # penalty = np.zeros_like(X_0)
    
    # graph_W = np.ones_like(graph_W)/N
    A_m = np.zeros((n,n))
    for i in range(N):
        A_m += A[i]@A[i].T
    
    all_start = time.time()
    retraction_time = 0
    communication_time = 0
    gradient_time = 0
    logging_time = 0


    for i in range(N):
        AAT = A[i]@A[i].T
        AW = AAT@W[i]
        S[i] = S[i] + AW - X_0[i]
        # S[i] = S[i] + AW 

    S = FastMix(S, graph_W)
    for i in range(N):
        new_W[i], _, V = np.linalg.svd(S[i], full_matrices=False)
        # new_W[i], _ = np.linalg.qr(S[i])
    # new_W = SignAdjust(new_W, X_0)

    # old_W = copy.deepcopy(x)
    # x = copy.deepcopy(new_W)
        # diff[i] = A[i]@A[i].T@x[i]
    print(Cal_consensus_error(new_W))

    # new_x = np.zeros_like(X_0)
    for k in tqdm(range(max_iter)):
        old_W = copy.deepcopy(W)
        W = copy.deepcopy(new_W)

        
        start = time.time()
        for i in range(N):
            AAT = A[i]@A[i].T
            AW = AAT@W[i]
            # S[i] = S[i] + .01*step_size *(AW - AAT@old_W[i])
            S[i] = S[i] + (AW - AAT@old_W[i])
            # S[i] = S[i] + (AW )
            grad[i] = AW - AAT@old_W[i]
            # grad[i] = AW 
                
        end = time.time()
        gradient_time += end-start
                    
            
        start = time.time()
        for _ in range(30):
            S = FastMix(S, graph_W)
        end = time.time()
        communication_time += end-start        
        
        
        
        start = time.time()
        for i in range(N):
            new_W[i], _, V = np.linalg.svd(S[i], full_matrices=False)
            # new_W[i], _ = np.linalg.qr(S[i])
        new_W = SignAdjust(new_W, X_0)
        end = time.time()
        gradient_time += end-start
        
        
        
        start = time.time()
        W_bar = np.average(W, axis=0)
        # u, s, v = np.linalg.svd(x_opt.T@x_bar)
        # dist = np.sqrt(2*np.abs(r - np.sum(s)))
        dist = 0
        for i in range(N):
            dist += np.linalg.norm(W[i].T@W[i]- np.eye(r), ord='fro')  
            
                                  
        grad_avg = np.average(grad, axis=0)
        grad_norm = np.linalg.norm(grad_avg)
        
        S_bar = np.average(S, axis=0)
        consensus_error = 0
        for i in range(N):
            consensus_error += np.linalg.norm(S[i]- S_bar)
            
        if (consensus_error+grad_norm+dist) < (1e-10*grad_avg.shape[0]*grad_avg.shape[1]) or np.isnan(dist):
            break

        if verbose and k%100 == 0:
            W_opt, s, v = np.linalg.svd(A_m)
            W_opt = W_opt[:, :r]
            print("iter:", k, "dist:", dist)
            print("average x:", np.linalg.norm(np.average(W, axis=0)-W_opt))
    
            # return np.zeros(max_iter)
            
            print("consensus error:", consensus_error)
            print("function value:")
            print(np.sum(np.sum(W_bar.T@A_m@W_bar)))
            print("dist", dist)
        distance.append(dist)
        grad_norms.append(grad_norm)
        con_errors.append(consensus_error)
        end = time.time()
        logging_time += end-start
        
    total_time = time.time()-all_start
    print("total time:", total_time)
    print("retraction_time time:", retraction_time)
    print("communication_time time:", communication_time)
    print("gradient time:", gradient_time)
    print("logging_time time:", logging_time)
    # print(x_bar.T@x_bar)
    # print("average x:")
    # print(x_bar)
    return np.array(distance), np.array(con_errors), np.array(grad_norms), W_bar

