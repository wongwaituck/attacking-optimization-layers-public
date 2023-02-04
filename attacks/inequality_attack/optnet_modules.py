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

class IneqOptNet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, nineq=20, neq=0, eps=1e-4):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nineq * nCls + nineq)

    def forward(self, x):
        nBatch = x.size(0)
        # Normal FC network.
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        Q = Variable(torch.stack([torch.eye(self.nCls) for _ in range(nBatch)])).to(device)

        q = Variable(torch.ones(self.nCls)).to(device)
        x_t = x[:, :self.nineq * self.nCls]
        G = x_t.reshape(nBatch, self.nineq, self.nCls)
        x_b = x[:, self.nineq * self.nCls:]
        h = x_b
        e = Variable(torch.Tensor()).to(device)
        x = QPFunction(verbose=-1, solver=qpth.qp.QPSolvers.CVXPY)(Q, q, G, h, e, e)

        return F.log_softmax(x)