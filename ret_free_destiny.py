import numpy as np
from update_utils import *
from tqdm import tqdm
import copy
import time


def DESTINY(A, W, step_size, X_0, x_opt, max_iter=10000, lin_term=None, verbose=False):
    distance = []
    con_errors = []
    grad_norms = []
    N = X_0.shape[0]
    n = X_0.shape[1]
    r = X_0.shape[2]
    
    lambd = 1
    print(lambd)
    # lambd = 1
    beta = 1
    x = np.copy(X_0)
    D = np.zeros_like(X_0)
    H = np.zeros_like(X_0)
    diff = np.zeros_like(X_0)
    penalty = np.zeros_like(X_0)
    
    A_m = np.zeros((n,n))
    for i in range(N):
        A_m += A[i]@A[i].T
    
    all_start = time.time()
    retraction_time = 0
    communication_time = 0
    gradient_time = 0
    logging_time = 0

    step_BB = np.ones(N)*step_size

    for i in range(N):
        
        xtx = x[i].T@x[i]
        xxtx = x[i]@xtx
        diff[i] = A[i]@A[i].T@xxtx
        if lin_term is not None:
            diff[i] += lin_term[i]      
        
        xtdiff = x[i].T@diff[i]
        
        G = diff[i]@(3*np.eye(r) - xtx)/2 - x[i]@(xtdiff + xtdiff.T)/2
        penalty[i] = lambd * x[i] @ (x[i].T@x[i] - np.eye(r))
        H[i] = G + penalty[i]
    D = copy.deepcopy(H)
        # diff[i] = A[i]@A[i].T@x[i]
        # if lin_term is not None:
        #     diff[i] += lin_term[i]
        # grad[i] = proj_tangent(x[i], diff[i])
        # penalty[i] = lambd * x[i] @ (x[i].T@x[i] - np.eye(r))
        # # y[i] = proj_tangent(x[i], diff[i])
        # y[i] = step_size * grad[i] - penalty[i]

    # y = np.copy(grad)
    # y = np.copy(diff)
    # y = step_size * grad - penalty
    # manual first round
    # for i in range(N):
    #     x[i] = x[i] + proj_tangent(x[i], grad[i]) - penalty[i]
        
    # new_x = np.zeros_like(X_0)
    for k in tqdm(range(max_iter)):
        old_D = copy.deepcopy(D)
        old_H = copy.deepcopy(H)
        # old_diff = copy.deepcopy(diff)
        old_x = copy.deepcopy(x)
        update_x = copy.deepcopy(x)
        # old_y = copy.deepcopy(y)
        # old_penalty = copy.deepcopy(penalty)
        
        for i in range(N):
            # update_x[i] -= step_BB[i] * D[i]
            # update_x[i] -= step_BB[i] * H[i]
            update_x[i] -= step_size * D[i]
        
        
        start = time.time()
        x = np.zeros_like(x)
        for i in range(N):
            for j in range(N):
                x[i] += W[i,j] * update_x[j] 
                # Wx[i] += W[i,j] * (old_x[j])

        end = time.time()
        communication_time += end-start
                    
            
        start = time.time()
        # grad = np.zeros_like(X_0)
        # diff = np.zeros_like(X_0)
        for i in range(N):
            xtx = x[i].T@x[i]
            xxtx = x[i]@xtx
            diff[i] = A[i]@A[i].T@xxtx
            if lin_term is not None:
                diff[i] += lin_term[i]      
            xtdiff = x[i].T@diff[i]
            
            G = diff[i]@(3*np.eye(r) - xtx)/2 - x[i]@((xtdiff + xtdiff.T)/2)
            penalty[i] = lambd * x[i] @ (x[i].T@x[i] - np.eye(r))
            H[i] = G + penalty[i]
            
            # diff[i] = A[i]@A[i].T@x[i]
            # if lin_term is not None:
            #     diff[i] += lin_term[i]          
            # grad[i] = proj_tangent(x[i], diff[i])
            # penalty[i] = lambd * x[i] @ (x[i].T@x[i] - np.eye(r))
            # print(np.linalg.norm(x[i].T@x[i] - np.eye(r)))
        end = time.time()
        gradient_time += end-start        
        
        start = time.time()
        D = np.zeros_like(x)
        for i in range(N):
            for j in range(N):
                D[i] += W[i,j] * old_D[j] 
            D[i] += H[i] -  old_H[i]
                # Wx[i] += W[i,j] * (old_x[j])
            S = x[i]-old_x[i]
            S = np.reshape(S, n*r)
            J = D[i] - old_D[i]
            J= np.reshape(J, n*r)
            step_BB[i] = np.abs(np.dot(S, J)/np.dot(J, J))

        end = time.time()
        communication_time += end-start
        

        start = time.time()
        x_bar = np.average(x, axis=0)
        # u, s, v = np.linalg.svd(x_opt.T@x_bar)
        # dist = np.sqrt(2*np.abs(r - np.sum(s)))
        dist = 0
        # for i in range(N):
        #     dist += np.linalg.norm(x[i].T@x[i]- np.eye(r), ord='fro')  
        dist = np.linalg.norm(x_bar.T@x_bar- np.eye(r), ord='fro')  
                                  
        grad_avg = np.average(D, axis=0)
        grad_norm = np.linalg.norm(grad_avg)
        
        # x_bar = np.average(x, axis=0)
        consensus_error = 0
        for i in range(N):
            consensus_error += np.linalg.norm(x[i]- x_bar)
            
        if (consensus_error+grad_norm+dist) < (1e-10*grad_avg.shape[0]*grad_avg.shape[1]) or np.isnan(dist):
            break

        if verbose and k%100 == 0:
            x_opt, s, v = np.linalg.svd(A_m)
            x_opt = x_opt[:, :r]
            print("iter:", k, "dist:", dist)
            print("average x:", np.linalg.norm(np.average(x, axis=0)-x_opt))
    
            # return np.zeros(max_iter)
            
            print("consensus error:", consensus_error)
            print("function value:")
            print(np.sum(np.sum(x_bar.T@A_m@x_bar)))
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
    return np.array(distance), np.array(con_errors), np.array(grad_norms), x_bar


# def ret_free_integral(A, W, step_size, X_0, x_opt, max_iter=10000, lin_term=None, verbose=False):
#     distance = []
#     con_errors = []
#     grad_norms = []
#     N = X_0.shape[0]
#     n = X_0.shape[1]
#     r = X_0.shape[2]
    
#     lambd = 0.2
#     beta = 0.1
#     x = np.copy(X_0)
#     y = np.zeros_like(X_0)
#     grad = np.zeros_like(X_0)
#     penalty = np.zeros_like(X_0)
    
#     A_m = np.zeros((n,n))
#     for i in range(N):
#         A_m += A[i]@A[i].T
    
    


#     for i in range(N):
#         grad[i] = A[i]@A[i].T@x[i]
#         if lin_term is not None:
#             grad[i] += lin_term[i]
#         grad[i] = proj_tangent(x[i], grad[i])
#         penalty[i] = lambd * x[i] @ (x[i].T@x[i] - np.eye(r))
#         y[i] = proj_tangent(x[i], A[i]@A[i].T@x[i])
    
#     y = np.zeros_like(grad)
    
#     # manual first round
#     # for i in range(N):
#     #     x[i] = x[i] + proj_tangent(x[i], grad[i]) - penalty[i]
        
#     # new_x = np.zeros_like(X_0)
#     for k in tqdm(range(max_iter)):
#         # old_grad = grad
#         old_x = copy.deepcopy(x)
#         # old_y = y
#         # old_penalty = penalty
                
#         diff = np.zeros_like(X_0)
#         grad = np.zeros_like(X_0)
#         for i in range(N):
        
#             diff[i] = A[i]@A[i].T@x[i]
#             if lin_term is not None:
#                  diff[i] += lin_term[i]
#             grad[i] = proj_tangent(x[i], diff[i])
#             penalty[i] = lambd * x[i] @ (x[i].T@x[i] - np.eye(r))
            
            
#         Wx = np.zeros_like(x)
#         for i in range(N):
#             for j in range(N):
#                 Wx[i] += W[i,j] * (old_x[j] - old_x[i])
#                 # Wx[i] += W[i,j] * (old_x[j])
#         y += Wx   
#         # y -=   penalty   
#         for i in range(N):
#             # x[i] = x[i] + proj_tangent(x[i], beta*Wx[i] + step_size*grad[i] + beta* 0.1* y[i]) - penalty[i]
#             # x[i] = x[i] +  step_size*grad[i]  + proj_tangent(x[i],  beta*Wx[i] + beta* 0.1* y[i]) - penalty[i]
#             # x[i] = x[i] +  step_size*grad[i]  + proj_tangent(x[i],  beta*Wx[i] + beta* y[i]) - penalty[i]
#             x[i] = x[i] +  step_size*grad[i]  + beta*Wx[i] + beta* y[i] - penalty[i]
#             # x[i] = x[i] +  step_size*grad[i]  +  beta* y[i] - penalty[i]
#             # x[i] = x[i] + proj_tangent(x[i], beta*Wx[i] + step_size*grad[i]) - penalty[i]
            
            
        
#             # print(np.linalg.norm(x[i].T@x[i] - np.eye(r)))
#         # for i in range(N):
#         #     for j in range(N):
#         #         y[i] += W[i,j] * (old_x[j] - old_x[i])
#         # y += Wx
#         # for i in range(N):
#         #     x[i] = x[i] + step_size*grad[i] + proj_tangent(x[i],   beta* 0.1* y[i]) - penalty[i]
#             # y[i] += grad[i] - old_grad[i]
#         # dist = 0
#         # for i in range(N):
#         #     dist += np.linalg.norm(x_opt-x[i])
        
#         x_bar = np.average(x, axis=0)
#         # u, s, v = np.linalg.svd(x_opt.T@x_bar)
        
#         # dist = np.sqrt(2*np.abs(r - np.sum(s)))
        
#         grad_avg = np.average(grad, axis=0)
#         grad_norm = np.linalg.norm(grad_avg)
        
#         dist = 0
#         for i in range(N):
#             dist += np.linalg.norm(x[i].T@x[i]- np.eye(r), ord='fro')        
#         # dist = np.linalg.norm(penalty)

#         x_bar = np.average(x, axis=0)
#         consensus_error = 0
#         for i in range(N):
#             consensus_error += np.linalg.norm(x[i]- x_bar)
            
            
#         if (consensus_error+grad_norm+dist) < 1e-6 or np.isnan(dist):
#             break
        
        
#         if verbose and k%100 == 0:
#             x_opt, s, v = np.linalg.svd(A_m)
#             x_opt = x_opt[:, :r]
#             print("iter:", k, "dist:", dist)
#             print("average x:", np.linalg.norm(np.average(x, axis=0)-x_opt))
    
#             # return np.zeros(max_iter)

#             print("consensus error:", consensus_error)
#             print("function value:")
#             print(np.sum(np.sum(x_bar.T@A_m@x_bar)))
#             print("dist", dist)
#         distance.append(dist)
#         grad_norms.append(grad_norm)
#         con_errors.append(consensus_error)

#     # print(x_bar.T@x_bar)
#     # print("average x:")
#     # print(x_bar)
#     return np.array(distance), np.array(con_errors), np.array(grad_norms), x_bar