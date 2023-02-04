#!/usr/bin/env python3

import setGPU
from math import inf, isnan
import torch
import torch.nn as nn
from torch.autograd import Variable
from consts import *
import torch.nn.functional as F
import argparse
import torch.optim as optim
from qpth.qp import QPFunction
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os

# define optnet model
class OptNet(nn.Module):
    def __init__(self, Qpenalty, nEq, nDecs):
        super().__init__()
        # weights on acceleration and speed
        self.Q = Variable(torch.diag_embed(torch.Tensor([1, 0.5]))).cuda()
        self.p = Variable(torch.Tensor([0, -1000])).cuda()

        # We encode  the following feasibility constraints
        # 1. Limiting max velocity to 100 km/h or 100000m/h.
        # 2. Limiting increase to acceleration (the tesla roadster can travel at 14 m/s^2)
        self.G = Variable(torch.Tensor([
            [1, 0],
            [0, -1]
        ])).cuda()
        self.h = Variable(torch.Tensor([
            14,
            0
        ])).cuda()
        #self.G = Variable(torch.zeros((1, nDecs))).cuda()
        #self.h = Variable(torch.zeros(1)).cuda()


    def forward(self, A, b, G=None, h=None):
        nBatch = A.shape[0]
        
        if G and h:
            G = torch.cat([G, torch.stack([self.G for _ in range(nBatch)])], dim=1)
            h = torch.cat([h, torch.stack([self.h for _ in range(nBatch)])], dim=1)
        else:
            G = self.G
            h = self.h

        return QPFunction(verbose=-1)(
            self.Q, self.p, G, h, A, b
        ).float()


class QPSpeedPlanning(nn.Module):
    def __init__(self, epsilon, nEqs=2, nDecs=2, nIneqs=0):
        super(QPSpeedPlanning, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fcn = nn.Linear(9219, nEqs * nDecs + nIneqs * nDecs + nEqs + nIneqs)
        self.optnet = OptNet(0.1, nEqs, nDecs)
        self.epsilon = epsilon
        self.nEqs = nEqs
        self.nDecs = nDecs
        self.nIneqs = nIneqs

    def forward(self, x):
        nBatch = x.size(0)

        # split into image and speed
        image = x.clone()[:, :IMG_DIMENSIONS[0] * IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2]]
        image = image.reshape(
            (nBatch, IMG_DIMENSIONS[2], IMG_DIMENSIONS[0], IMG_DIMENSIONS[1]))
        speed = x.clone()[:, -3].reshape((nBatch, 1))
        dest_distance = x.clone()[:, -2].reshape((nBatch, 1))
        car_distance = x.clone()[:, -1].reshape((nBatch, 1))

        x = self.conv1(image)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        # join back 
        x = torch.cat([x, speed, dest_distance, car_distance], dim=1)

        x = self.fcn(x)
        x = F.relu(x)

        if self.nIneqs > 0:
            x_A = x[:, :(self.nEqs * self.nDecs)]
            x_b = x[:, -(self.nEqs + self.nIneqs):-self.nIneqs]
            A = x_A.reshape(nBatch, self.nEqs, self.nDecs)
            b = x_b.reshape(nBatch, self.nEqs)

            x_G = x[:, (self.nEqs * self.nDecs): (self.nEqs * self.nDecs + self.nIneqs * self.nDecs)]
            x_h = x[:, -self.nIneqs:]
            G = x_G.reshape(nBatch, self.nIneqs, self.nDecs)
            h = x_h.reshape(nBatch, self.nIneqs)
        else:
            x_A = x[:, :(self.nEqs * self.nDecs)]
            x_b = x[:, -self.nEqs:]
            A = x_A.reshape(nBatch, self.nEqs, self.nDecs)
            b = x_b.reshape(nBatch, self.nEqs)
            G = None
            h = None
        
        G_p = G
        # defence
        if self.epsilon != 0:
            # perturb the matrix a little bit to handle the all zeroes case        
            try:
                U, S, V = torch.svd(A)
                S_p = torch.zeros_like(S)
                for i in range(nBatch):
                    S_p[i] = torch.clamp(S[i], torch.max(S[i]).item()/(self.epsilon), torch.max(S[i]).item())
                    if torch.eq(S_p[i], torch.zeros_like(S_p[i])).all():
                        S_p[i] += 0.01
                A_p = U @ torch.diag_embed(S_p) @ torch.transpose(V, 1, 2)
            except Exception as e:
                # this is a svd cuda bug
                print(e)
                A_p = A

        else:
            A_p = A
            
        x = self.optnet(A_p, b, G_p, h)
        return x

def get_acceleration(z):
    '''
    gets the acceleration component of the output
    '''
    return z[:, 0].clone()

def get_targetspeed(z):
    '''
    gets the target speed component of the output
    '''
    return z[:, 1].clone()

# loss functions
def ComfortPenaltyLoss(z):
    '''
    Comfort is defined as the absolute acceleration different from some comfortable threshold.
    We follow the value in Liu et al.[2017] of a threshold of 2.5m/s^2.
    '''
    THRESHOLD = 2.5
    accelerations = get_acceleration(z)
    accelerations = torch.abs(accelerations)
    discomfort = accelerations - THRESHOLD
    discomfort = discomfort.clamp(0, inf)
    return discomfort


def SafetyPenaltyLoss(z, car_dist):
    '''
    Safety is defined as the likelihood of hitting the car ahead of us given the two-second rule (https://en.wikipedia.org/wiki/Two-second_rule), given our acceleration initial speed.

    We calculate the distance travelled by both cars and penalize the distance 

    A simplifying assumption is that the car ahead is travelling at the same speed as us currently.
    '''
    SAFETY_SECONDS = 2
    accelerations = get_acceleration(z)
    add_distance_travelled = 0.5 * accelerations * SAFETY_SECONDS * SAFETY_SECONDS
    distance_exceeded = add_distance_travelled - car_dist
    distance_exceeded = distance_exceeded.clamp(0, inf)
    return distance_exceeded

def TrafficViolationPenaltyLoss(z, targets):
    # map each true class to a "speed limit"
    SPEED_MAP = {
        YIELD: 20000,
        STOP: 0,
        NO_ENTRY: 0,
        SPEED_LIMIT: 50000,
        PEDESTRIAN_XING: 10000
    }
    targets = targets.cpu().numpy()
    mapper = lambda t: SPEED_MAP[t]
    vfunc = np.vectorize(mapper)
    target_speeds = vfunc(targets)
    target_speeds = torch.Tensor(target_speeds).cuda()

    # the difference between the targetspeed and the speed limit is the loss 
    actual_target_speed = get_targetspeed(z)
    violation_speeds = actual_target_speed - target_speeds
    violation_speeds = violation_speeds.clamp(0, inf)

    return violation_speeds

def DistancePenaltyLoss(z, init_speed, dist_target):
    '''
    given the new target_speed, how long do we need to reach the destination compared to the 
    old target speed, given the current acceleration?
    '''
    acc = get_acceleration(z) 
    target_speed = get_targetspeed(z)
    lost_speed = init_speed - target_speed
    lost_speed = lost_speed.clamp(-inf, -1)
    lost_speed = -lost_speed


    speed_increase = target_speed - init_speed
    speed_increase = speed_increase.clamp(0, inf)
    acc = acc.clamp(1, inf)
    time_to_acc = speed_increase / acc


    return dist_target / lost_speed + time_to_acc

def make_logs(logger, logdir, seed):
    # dump the model and the logs
    os.makedirs(logdir, exist_ok=True)
    torch.save(model.state_dict(), f"{logdir}/model{seed}.pt")
    for k, v in logger.items():
        v_np = np.array(v)
        np.savetxt(f"{logdir}/seed_{seed}_{k}.csv", v_np)


def train(model, device, trainset, Y_train, optimizer, epoch, batch_sz, logger, logdir, seed):
    total_loss = 0
    total_safety_loss = 0
    total_violation_loss = 0
    total_comfort_loss = 0
    total_distance_loss = 0

    for idx in range(0, len(trainset), batch_sz):
        idx_start = idx
        idx_end = idx_start + batch_sz
        data = trainset[idx_start: idx_end]
        targets = Y_train[idx_start: idx_end]
        optimizer.zero_grad()
        
        output = model(data)

        speed = data.clone()[:, -3].reshape((batch_sz, 1))
        dest_distance = data.clone()[:, -2].reshape((batch_sz, 1))
        car_distance = data.clone()[:, -1].reshape((batch_sz, 1))

        safety_loss = SafetyPenaltyLoss(output, car_distance)
        comfort_loss = ComfortPenaltyLoss(output)
        traffic_violation_loss = TrafficViolationPenaltyLoss(output, targets)
        distance_loss = DistancePenaltyLoss(output, speed, dest_distance)
        loss = 100 * safety_loss + 10 * traffic_violation_loss + 0.001  * distance_loss + 0.1 * comfort_loss
        loss.sum().backward()
        optimizer.step()

        total_safety_loss += safety_loss.mean().item()
        total_violation_loss += traffic_violation_loss.mean().item()
        total_comfort_loss += comfort_loss.mean().item()
        total_distance_loss += distance_loss.mean().item()
        total_loss += loss.mean().item() 
        if isnan(total_loss):
            make_logs(logger, logdir, seed)
            exit(-1)
        if idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\nSafetyLoss: {:.6f}\tViolationLoss: {:.6f}\tDistanceLoss: {:.6f}\tComfortLoss: {:.6f}'.format(
                epoch, idx + batch_sz, len(trainset),
                100. * (idx + batch_sz) / len(trainset), loss.mean().item(),
                safety_loss.mean().item(), traffic_violation_loss.mean().item(),
                distance_loss.mean().item(), comfort_loss.mean().item()))
    total_safety_loss /= (len(trainset) / batch_sz)
    total_violation_loss /=  (len(trainset) / batch_sz)
    total_distance_loss /=  (len(trainset) / batch_sz)
    total_comfort_loss /=  (len(trainset) / batch_sz)
    total_loss /=  (len(trainset) / batch_sz)

    logger["train_loss_overall"].append(total_loss)
    logger["train_loss_safety"].append(total_safety_loss)
    logger["train_loss_violation"].append(total_violation_loss)
    logger["train_loss_distance"].append(total_distance_loss)
    logger["train_loss_comfort"].append(total_comfort_loss)

def test(model, device, testset, batch_sz, Y_test, logger):
    total_loss = 0
    total_safety_loss = 0
    total_violation_loss = 0
    total_comfort_loss = 0
    total_distance_loss = 0

    test_sz = len(testset)
    with torch.no_grad():
        data = testset
        output = model(data)
        speed = data.clone()[:, -3].reshape((test_sz, 1))
        dest_distance = data.clone()[:, -2].reshape((test_sz, 1))
        car_distance = data.clone()[:, -1].reshape((test_sz, 1))

        safety_loss = SafetyPenaltyLoss(output, car_distance)
        comfort_loss = ComfortPenaltyLoss(output)
        traffic_violation_loss = TrafficViolationPenaltyLoss(output, Y_test)
        distance_loss = DistancePenaltyLoss(output, speed, dest_distance)
        loss = 100 * safety_loss + 10 * traffic_violation_loss + 0.001 * distance_loss + 0.1 * comfort_loss
        
        total_safety_loss += safety_loss.mean().item()
        total_violation_loss += traffic_violation_loss.mean().item()
        total_comfort_loss += comfort_loss.mean().item()
        total_distance_loss += distance_loss.mean().item()
        total_loss += loss.mean().item() 

    
    print('\nTest set: Average loss: {:.4f}\nSafetyLoss: {:.6f}\tViolationLoss: {:.6f}\tDistanceLoss: {:.6f}\tComfortLoss: {:.6f}\n'.format(
        total_loss, safety_loss.mean().item(), traffic_violation_loss.mean().item(),
                distance_loss.mean().item(), comfort_loss.mean().item()))

    logger["test_loss_overall"].append(total_loss)
    logger["test_loss_safety"].append(total_safety_loss)
    logger["test_loss_violation"].append(total_violation_loss)
    logger["test_loss_distance"].append(total_distance_loss)
    logger["test_loss_comfort"].append(total_comfort_loss)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="Epochs to run the experiment", type=int,  default=30)
    parser.add_argument("-d", "--datadir", help="Directory to load the data from",  default="out")
    parser.add_argument("-l", "--logdir", help="Directory to save the log data from",  default="better_log")
    parser.add_argument("-r", "--seed", help="Random seed to use", type=int,  default=3)
    parser.add_argument("-b", "--batch-sz", help="Batch size to use", type=int,  default=100)
    parser.add_argument("-g", "--cuda", help="Use GPU", action="store_true", default=False)
    parser.add_argument("-s", "--epsilon", help="Epsilon to use", type=float,  default=0)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and args.cuda else "cpu")

    torch.manual_seed(args.seed)
    # import the dataset
    X_train = torch.load(f"{args.datadir}/X_train.pt").to(device)
    Y_train = torch.load(f"{args.datadir}/Y_train.pt").to(device)
    X_test = torch.load(f"{args.datadir}/X_test.pt").to(device)
    Y_test = torch.load(f"{args.datadir}/Y_test.pt").to(device)

    LOGS = {
        "train_loss_safety": [],
        "train_loss_violation": [],
        "train_loss_comfort": [],
        "train_loss_distance": [],
        "train_loss_overall": [],
        "test_loss_safety": [],
        "test_loss_violation": [],
        "test_loss_distance": [],
        "test_loss_comfort": [],
        "test_loss_overall": [],
    }

    model = QPSpeedPlanning(args.epsilon).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    LOGDIR = args.logdir + f'/epsilon_{args.epsilon}' 

    for epoch in range(1, args.epochs + 1):
        train(model, device, X_train, Y_train, optimizer, epoch, args.batch_sz, LOGS, LOGDIR, args.seed)
        test(model, device, X_test, args.batch_sz, Y_test, LOGS)
        scheduler.step()

    # dump the model and the logs
    os.makedirs(LOGDIR, exist_ok=True)
    torch.save(model.state_dict(), f"{LOGDIR}/model{args.seed}.pt")
    for k, v in LOGS.items():
        v_np = np.array(v)
        np.savetxt(f"{LOGDIR}/seed_{args.seed}_{k}.csv", v_np)