#!/usr/bin/env python3

from numpy.core.numeric import zeros_like
import setGPU
import torch
import os
import timeit
import numpy as np
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--epochs-train", help="Number of Epochs to train", type=int, default=1000)
parser.add_argument("-a", "--epochs-attack", help="Number of Epochs to attack", type=int, default=5000)
parser.add_argument("-t", "--train-sz", help="Number of training samples to use", type=int, default=30)
parser.add_argument("-k", "--attack-sz", help="Number of training samples to attack", type=int, default=5)
parser.add_argument("-e", "--neq", help="Number of equalities", type=int, default=50)
parser.add_argument("-c", "--ncl", help="Number of classes", type=int, default=50)
parser.add_argument("-l", "--learning-rate", help="Learning rate for the attack", type=float, default=50)
parser.add_argument("-r", "--seed", help="Random seed to use", type=int, default=3)
parser.add_argument("-cuda", "--use-cuda", help="Use GPU/cuda", type=int, default=1)
parser.add_argument("-eps", "--epsilon", help="Epsilon to use in defense (0 = undefended)", type=int, default=1)
args = parser.parse_args()


epsilon = args.epsilon

assert(args.attack_sz <= args.train_sz)

# Declare a simple network.

# FC-ReLU-FC-ReLU-QP-Softmax
# Where the QP OptNet layer learns the coefficients A(z_i), b(z_i) for a QP with inequality constraints:

# z_{i+1} = argmin_d 0.5 d^T I d + {1}^n d
#           s.t. A(z_i)d = b(z_i)

class OptNet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, neq=args.neq, eps=1e-4):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls
        self.neq = neq
        self.eps = eps

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, neq * nCls + neq)

    def forward(self, x):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        Q = Variable(torch.eye(self.nCls)).to(device)
        q = Variable(torch.ones(self.nCls)).to(device)
        G = torch.zeros((self.nCls, self.nCls)).to(device)
        h = torch.zeros(self.nCls).to(device)
        x_t = x[:, :self.neq * self.nCls]
        A = x_t.reshape(nBatch, self.neq, self.nCls)
        x_b = x[:, self.neq * self.nCls:]
        b = x_b
        
        x = QPFunction(verbose=-1)(Q, q, G, h, A, b)

        return F.log_softmax(x)

class OptNetDef(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, neq=args.neq, eps=1e-4):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls
        self.neq = neq
        self.eps = eps

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, neq * nCls + neq)

    def forward(self, x):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        Q = Variable(torch.eye(self.nCls)).to(device)
        q = Variable(torch.ones(self.nCls)).to(device)
        G = torch.zeros((self.nCls, self.nCls)).to(device)
        h = torch.zeros(self.nCls).to(device)
        x_t = x[:, :self.neq * self.nCls]
        A = x_t.reshape(nBatch, self.neq, self.nCls)
        x_b = x[:, self.neq * self.nCls:]
        b = x_b

        def eye_like(tensor):
            return torch.eye(*tensor.size(), out=torch.empty_like(tensor))
        
        A_p = A + 1e-8 * torch.stack([eye_like(A[0]) for _ in range(nBatch)])
        
        x = QPFunction(verbose=-1)(Q, q, G, h, A_p, b)

        return F.log_softmax(x)

# Loss via condition number of the matrix A (not used since torch.linalg.cond is implemented in 1.8.1)
# Use -torch.linalg.cond(A) instead.
def ConditionNumberLoss(A, single_point=False):
    A_inv = torch.pinverse(A)
    if single_point:
        A_norm = torch.norm(A, 'fro')
        A_inv_norm = torch.norm(A_inv, 'fro')
    else:
        A_norm = torch.norm(A, 'fro', dim=(1,2))
        A_inv_norm = torch.norm(A_inv, 'fro', dim=(1,2))
    
    cond_num = A_inv_norm * A_norm
    return -cond_num

def ZeroRowLoss(A):
    '''
    zeroing out a row
    '''
    A_prime = A.clone()
    A_prime[:,0,:] = 0
    return torch.nn.L1Loss()(A, A_prime)

def ZeroColumnLoss(A):
    '''
    zeroing out a column
    '''
    A_prime = A.clone()
    A_prime[:,:,0] = 0
    return torch.nn.L1Loss()(A, A_prime)

def ZeroSingularValue(A):
    with torch.no_grad():
        u, s, v = torch.linalg.svd(A)
        # take the smallest singular value and zero i t
        s[:, len(s[0])-1] = 0
        s_p_sm = torch.diag_embed(s)
        z = torch.zeros_like(A)
        z[:,:s_p_sm.shape[1], :s_p_sm.shape[2]] = s_p_sm
        A_prime = u @ z @ v
    return torch.nn.L1Loss()(A, A_prime)

import math
import torch.optim as optim

loss_functions = ["ZeroSingular", "ConditionNumber", "ZeroColumn", "ZeroRow"]

# check if use GPU
if (args.use_cuda and torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.manual_seed(args.seed)
nBatch, nFeatures, nHidden, nCls = args.train_sz, 500, 500, args.ncl
# Create random data
x = Variable(torch.randn(nBatch, nFeatures), requires_grad=False).to(device)
y = Variable((torch.randint(nCls, (nBatch,))).long(), requires_grad=False).to(device)

x_test = Variable(torch.randn(nBatch, nFeatures), requires_grad=False).to(device)
y_test = Variable((torch.randint(nCls, (nBatch,))).long(), requires_grad=False).to(device)

# check if running defense or not
if (epsilon == 0):
    # directory to store results
    raise Exception()
else :
    outputDir = "model_def_aI/"+ "/seed_" + str(args.seed) + "/"
    model = OptNetDef(nFeatures, nHidden, nCls, bn=False).to(device)
os.makedirs(outputDir, exist_ok=True)

loss_fn = torch.nn.CrossEntropyLoss()

# Initialize the optimizer.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
# load pre-trained network if it exists
modelPath = f"{outputDir}network{nFeatures}f{nHidden}h{nCls}c{args.neq}eqs.pt"
modelLossPath = f"{outputDir}network{nFeatures}f{nHidden}h{nCls}c{args.neq}eqs_loss.csv"
modelTestLossPath = f"{outputDir}network{nFeatures}f{nHidden}h{nCls}c{args.neq}eqs_testloss.csv"
if(os.path.exists(modelPath)):
    model.load_state_dict(torch.load(modelPath))
else:
    # train and save network
    for t in range(args.epochs_train):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 10 == 0:
            print(t, loss.item())
        losses.append(loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the zloss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    torch.save(model.state_dict(), modelPath)
    np.savetxt(modelLossPath, np.array(losses))

y_pred = model(x_test)
loss = loss_fn(y_pred, y_test)
np.savetxt(modelTestLossPath, np.array([loss.item()]))
