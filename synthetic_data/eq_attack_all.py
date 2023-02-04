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
parser.add_argument("-eps", "--epsilon", help="Epsilon to use in defense (0 = undefended)", type=int, default=0)
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

        # defence
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        S_p = torch.zeros_like(S)
        for i in range(nBatch):
            S_p[i] = torch.clamp(S[i], torch.max(S[i]).item()/(epsilon), torch.max(S[i]).item())
        
        A_p = U @ torch.diag_embed(S_p) @ Vh
        
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

# check if running defense or not
if (epsilon == 0):
    # directory to store results
    outputDir = "model_nodef/seed_" + str(args.seed) + "/"
    model = OptNet(nFeatures, nHidden, nCls, bn=False).to(device)
else :
    outputDir = "model_def_eps_" + str(epsilon) + "/seed_" + str(args.seed) + "/"
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

# loop through all loss functions
for lf in loss_functions:
    if (lf == "ZeroRow"):
        # set a row to zero
        attackerLoss = ZeroRowLoss
    elif (lf == "ZeroColumn"):
        # set a column to zero
        attackerLoss = ZeroColumnLoss
    elif (lf == "ConditionNumber"):
        # maximize condition number
        attackerLoss = lambda A: -torch.linalg.cond(A)
    elif (lf == "ZeroSingular"):
        # set the lowest singular value to zero
        attackerLoss = ZeroSingularValue
    else:
        # invalid argument
        print("Unknown attacker loss function")
        exit()
    # store iterations to nan for each point
    iterations = []
    # store runtimes
    runtimes = []
    # store condition numbers
    allConditions = []
    # store attacker/defender losses
    allAttackerLs = []
    allDefenderLs = []
    # try attacking each point in the training dataset
    for point_idx in range(args.attack_sz):
        # store attacker+defender losses along with condition number for each iteration
        conditions = []
        attackerLosses = []
        defenderLosses = []
        # current point
        current_x = x[point_idx]
        current_x = current_x.view(1, current_x.size(0))
        # from https://adversarial-ml-tutorial.org/introduction/ (modified to respect the dataset)
        delta = torch.zeros_like(current_x, requires_grad=True).to(device)
        opt = optim.SGD([delta], lr=args.learning_rate, momentum=0.9)
        for t in range(args.epochs_attack):
            attacked = False
            starttime = timeit.default_timer()
            # Normal FC network.
            temp = torch.relu(model.fc1(current_x + delta))
            temp = torch.relu(model.fc2(temp))
            x_t = temp[:, :model.neq * model.nCls]
            A = x_t.reshape(1, model.neq, model.nCls)

            if (epsilon != 0):
                U, S, V = torch.svd(A)
                S_p = torch.zeros_like(S)
                S_p[0] = torch.clamp(S[0], torch.max(S[0]).item()/(epsilon), torch.max(S[0]).item())
                
                A = U @ torch.diag_embed(S_p) @ torch.transpose(V, 1, 2)

            # calculate loss (based on one data point so we only backpropagate a single data point)
            loss = attackerLoss(A)
            attackerLosses.append(loss.item())
            conditions.append(torch.linalg.cond(A).item())
            if t%10 == 0:
                print(t, loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if torch.isnan(delta.sum()):
                # failed to find a satisyfing delta
                break
            y_pred = model(current_x + delta)

            loss = loss_fn(y_pred, torch.unsqueeze(y[point_idx], 0))
            defenderLosses.append(loss.item())
            if math.isnan(loss):
                attacked = True
                iterations.append(t)
                runtimes.append(timeit.default_timer() - starttime)
                torch.save(delta, f"{outputDir}/adversarial{point_idx}_{lf}.pt")
                print(f"Found adversarial attack for point {point_idx} against model seed_{args.seed}/network{nFeatures}f{nHidden}h{nCls}c.pt")
                break
        if not attacked:
            iterations.append(-1)
            runtimes.append(-1)
            print (f"Unable to find attack for point for point {point_idx} against model seed_{args.seed}/network{nFeatures}f{nHidden}h{nCls}c.pt")
        # save stats specific to this point
        allConditions.append(conditions)
        allAttackerLs.append(attackerLosses)
        allDefenderLs.append(defenderLosses)

    # function to save nested lists of different length to csv
    def saveNestedList(fileName, nestedList):
        # open file
        with open(fileName, 'wb') as fl:
            for item in nestedList:
                np.savetxt(fl, [item], delimiter=", ")

    saveNestedList(f"{outputDir}/{args.neq}x{args.ncl}_cond_{lf}.csv", allConditions)
    saveNestedList(f"{outputDir}/{args.neq}x{args.ncl}_targetlosses_{lf}.csv", allAttackerLs)
    saveNestedList(f"{outputDir}/{args.neq}x{args.ncl}_classlosses_{lf}.csv", allDefenderLs)

    # save epochs for ALL points
    np.savetxt(f"{outputDir}/{args.neq}x{args.ncl}epochs_{lf}.csv", np.array(iterations))
    # save runtimes 
    np.savetxt(f"{outputDir}/{args.neq}x{args.ncl}runtimes_{lf}.csv", np.array(runtimes))
