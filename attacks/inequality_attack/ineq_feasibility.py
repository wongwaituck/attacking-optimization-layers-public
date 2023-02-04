#!/usr/bin/env python3

import warnings

import random
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import os
import math
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from loss_funcs import InequalityInfeasLoss
from optnet_modules import IneqOptNet

CUDA = False

if CUDA and torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev) 

import sys

# do training
for i in range(42,43):
    try:
        seed = 42
        print("Seed: " + str(seed))
        os.system(f'touch inequality/{seed}_STARTTRAINING.txt')
        model_path = f"inequality/model_{seed}.pt"
        torch.manual_seed(seed)
        TRAIN_ITER = 1000
        ATTACK_ITER = 50000
        PRINT_FREQ = 10
        gamma = 0.9

        nBatch, nFeatures, nHidden, nCls, nIneq = 16, 20, 20, 2, 2
        # Create random data
        x = Variable(torch.randn(nBatch, nFeatures), requires_grad=False).to(device)
        y = Variable((torch.randint(nCls, (nBatch,))).long(), requires_grad=False).to(device)

        model = IneqOptNet(nFeatures, nHidden, nCls, nineq=nIneq, bn=False)
        model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        try:
            model.load_state_dict(torch.load(model_path), strict=False)
        except:
            # Initialize the optimizer.
            learning_rate = 1e-3
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            
            training_losses = np.zeros((TRAIN_ITER))

            for t in range(TRAIN_ITER):
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = model(x).to(device)

                # Compute and print loss.
                loss = loss_fn(y_pred, y)

                if t % PRINT_FREQ == 0:
                    print(t, loss.item())

                training_losses[t] = loss.data

                if math.isnan(training_losses[t]):
                    raise Exception()

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
            
            np.savetxt('inequality/seed_'+str(seed)+"_trloss.csv", training_losses)
            
            torch.save(model.state_dict(), model_path)
        torch.manual_seed(int(sys.argv[1]))
        delta = torch.randn_like(x, requires_grad=True).to(device)
        # decay eta at a constant rate
        eta = 0.9
        eta_steps = 0.9 / ATTACK_ITER
        LR = 0.01
        opt = optim.SGD([delta], lr=LR, momentum=0.9)
        current_x = x

        infeas_losses = np.zeros((ATTACK_ITER))
        prereq_losses = np.zeros((ATTACK_ITER))
        total_losses = np.zeros((ATTACK_ITER))
        advclass_losses = np.zeros((ATTACK_ITER))
        As = []
        bs = []
        

        for t in range(ATTACK_ITER):
            x = current_x
            nBatch = x.size(0)

            # Normal FC network.
            x = x.view(nBatch, -1)
            x = F.relu(model.fc1(x + delta))
            x = model.fc2(x)
            x_t = x[:, :model.nineq * model.nCls]
            A = x_t.reshape(nBatch, model.nineq, model.nCls)
            As.append(A.clone().detach())
            x_b = x[:, model.nineq * model.nCls:]
            b = x_b
            bs.append(b.clone().detach())

            # calculate loss (based on one data point so we only backpropagate a single data point)
            loss_infeas, loss_prereq = InequalityInfeasLoss(A, b, eta=eta)
            loss = gamma * loss_infeas + (1 - gamma) * loss_prereq
            
            if t % PRINT_FREQ == 0:
                print(t, loss[0].item())
                
            infeas_losses[t] = loss_infeas[0].data
            prereq_losses[t] = loss_prereq.data
            total_losses[t] = loss[0].data
            
            opt.zero_grad()
            loss[0].backward(create_graph=True)
            opt.step()
            eta = eta - eta_steps
            
            with torch.no_grad():
                if math.isnan(delta.sum().item()):
                    delta[0] = torch.randn((delta[0].shape), requires_grad=True).to(device)
                    print('randomed')

                delta[1:] = 0
                opt = optim.SGD([delta], lr=LR, momentum=0.9)
            # See if it still works
            try:
                with torch.no_grad():
                    y_pred = model(current_x + delta)
                    loss = loss_fn(y_pred, y)
                    advclass_losses[t] = loss.data
                    if math.isnan(loss):
                        print("Found adversarial attack")
                        x = current_x
                        nBatch = x.size(0)

                        # Normal FC network.
                        x = x.view(nBatch, -1)
                        x = F.relu(model.fc1(x + delta))
                        x = model.fc2(x)
                        x_t = x[:, :model.nineq * model.nCls]
                        A = x_t.reshape(nBatch, model.nineq, model.nCls)
                        As.append(A.clone().detach())
                        x_b = x[:, model.nineq * model.nCls:]
                        b = x_b
                        bs.append(b.clone.detach())
                        print(delta)
                        os.system(f'touch inequality/{seed}_WINADV.txt')
                        break
            except Exception as e:
                print(e)

                print("Found adversarial attack")
                x = current_x
                nBatch = x.size(0)

                # Normal FC network.
                x = x.view(nBatch, -1)
                x = F.relu(model.fc1(x + delta))
                x = model.fc2(x)
                x_t = x[:, :model.nineq * model.nCls]
                A = x_t.reshape(nBatch, model.nineq, model.nCls)
                As.append(A.clone().detach())
                x_b = x[:, model.nineq * model.nCls:]
                b = x_b
                bs.append(b.clone().detach())
                print(delta)
                print(A[0])
                print(b[0])

                print(As[-1][0])
                print(bs[-1][0])

                print(As[-2][0])
                print(bs[-2][0])

                print(As[-3][0])
                print(bs[-3][0])


                os.system(f'touch inequality/{seed}_WINADV.txt')
                torch.save(delta, f'inequality/{seed}_adversarial.pt')
                break
            if t%PRINT_FREQ == 0:
                print('Actual Output with Modified x: Iteration {}, loss = {:.2f}'.format(t, loss.data))
        os.system(f'touch inequality/{seed}_ENDTRAINING.txt')
        np.savetxt('inequality/seed_'+str(seed)+"_infeasloss.csv", infeas_losses)
        np.savetxt('inequality/seed_'+str(seed)+"_prereqloss.csv", prereq_losses)
        np.savetxt('inequality/seed_'+str(seed)+"_totalloss.csv", total_losses)
        np.savetxt('inequality/seed_'+str(seed)+"_advclassloss.csv", advclass_losses)
        with open('inequality/seed_'+str(seed)+"As.txt", "wt") as f:
            f.write(str(As))
            f.close()
        with open('inequality/seed_'+str(seed)+"bs.txt", "wt") as f:
            f.write(str(bs))
            f.close()
    except Exception as e:
        raise e
        os.system(f'touch inequality/{seed}_FAILTRAINING.txt')
        np.savetxt('inequality/seed_'+str(seed)+"_trloss.csv", training_losses)
        pass