import torch

import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import qpth

from qpth.qp import QPFunction
CUDA = False

if CUDA and torch.cuda.is_available(): 
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev) 

torch.cuda.set_device(0)

def InequalityInfeasLoss(A, b, single_point=False, eta=0.9, gamma=0.9, phi=10, solver="CVXPY", printz=False):
    A_t = torch.transpose(A, 1, 2)

    nBatch = A.shape[0]
    nIneq = A.shape[1]
    nVar = A.shape[2]

    b = b.reshape((nBatch, nIneq, 1))
    b_t = torch.transpose(b, 1, 2)

    q = torch.zeros((nVar + 1, 1)).to(device)

    q[-1] = -1
    q = q.repeat(A.shape[0], 1, 1)
    K = torch.cat([A_t, b_t], 1)
    K_t = torch.transpose(K, 1, 2)
    K_inv = torch.pinverse(K)

    # K'-1 * q
    constant = torch.matmul(K_inv, q)
    constant = constant.reshape((constant.shape[0], constant.shape[1]))
    constraint = torch.zeros((constant.shape[0], constant.shape[1] * 2)).to(device)
    constraint[:,:nIneq] = constant - phi
    
    L_prereq = torch.norm(torch.matmul(torch.matmul(K, K_inv), q) - q)

    # inf_{t, w} || t - (I-K^(-1)K) w ||_2^2
    # B is an identity matrix (to capture y) augmented with B_R = (K^(-1)K - I) to capture w
    B = torch.zeros(nIneq, 2*nIneq).to(device)
    B = B.reshape((1, nIneq, 2*nIneq))
    B = B.repeat(A.shape[0], 1, 1)

    B_I = torch.eye(nIneq, nIneq).to(device)

    B_I = B_I.reshape((1, nIneq, nIneq))
    B_I = B_I.repeat(A.shape[0], 1, 1)
    B[:, :nIneq, :nIneq] = B_I

    B_R = torch.bmm(K_inv, K)
    B_R = B_R
    B_R = B_R - B_I
    with torch.no_grad():
        B[:,:, nIneq:] = B_R

    B_s = torch.zeros((A.shape[0], 2*nIneq)).to(device)

    # construct constraint z[:|t|] >= -K^-1 * q ==> -z[:|t|] <= K^-1 * q
    # (2nIneq) X  (2nIneq)
    S = torch.zeros(2 * nIneq, 2 * nIneq).to(device)
    S = S.reshape((1, S.shape[0], S.shape[1]))
    S = S.repeat(A.shape[0], 1, 1)
    Z = torch.zeros((A.shape[0], nIneq, nIneq)).to(device)
    S[:,:nIneq,:nIneq] = -B_I
    S[:,nIneq:, nIneq:] = Z


    # inf_z (z^T * B^T * B * z)
    e = Variable(torch.Tensor()).to(device)

    Q = torch.bmm(torch.transpose(B, 1, 2), B)
    Q_I = torch.eye(Q.shape[1], Q.shape[2]).to(device)
    Q_I = Q_I.reshape((1, Q.shape[1], Q.shape[2]))
    Q_I = Q_I.repeat(Q.shape[0], 1, 1)
    Q += eta * Q_I

    if solver == "CVXPY":
        try:
            z = QPFunction(verbose=-1, solver=qpth.qp.QPSolvers.CVXPY)(Q, B_s, S, constraint, e, e)
        except:
            print(str(Q))
    else:
        z = QPFunction(verbose=-1)(Q, B_s, S, constraint, e, e)
    z = z.reshape((z.shape[0], z.shape[1], 1))
    loss = torch.norm(torch.matmul(B, z), dim=(1,2))
    print(loss[0])
    print(L_prereq)
    return [loss, L_prereq]