#!/usr/bin/env python3

from math import inf, isnan
import torch
import torch.nn as nn
from torch.autograd import Variable
from consts import *
import torch.nn.functional as F
import argparse
import torch.optim as optim
from qpth.qp import QPFunction
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import numpy as np
import os
from PIL import Image
from torchvision import transforms

# define optnet model
class OptNet(nn.Module):
    def __init__(self, Qpenalty, nEq, nDecs):
        super().__init__()
        # weights on acceleration and speed
        self.Q = Variable(torch.diag_embed(torch.Tensor([1, 0.5])))
        self.p = Variable(torch.Tensor([0, -1000]))

        # We encode  the following feasibility constraints
        # 1. Limiting max velocity to 100 km/h or 100000m/h.
        # 2. Limiting increase to acceleration (the tesla roadster can travel at 14 m/s^2)
        self.G = Variable(torch.Tensor([
            [1, 0],
            [0, -1]
        ]))
        self.h = Variable(torch.Tensor([
            14,
            0
        ]))


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
        image = torch.clamp(image, 0, 1)
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
    target_speeds = torch.Tensor(target_speeds)

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


# Loss via condition number of the matrix A
def ConditionNumberLoss(A, single_point=False):
    A_inv = torch.pinverse(A)
    if single_point:
        A_norm = torch.norm(A, 'fro')
        A_inv_norm = torch.norm(A_inv, 'fro')
    else:
        A_norm = torch.norm(A, 'fro', dim=(1,2))
        A_inv_norm = torch.norm(A_inv, 'fro', dim=(1,2))
    
    cond_num = A_inv_norm * A_norm
    return -torch.log(cond_num)


def ZeroColumnLoss(A):
    idx = torch.zeros(2).long()
    j = torch.arange(2).long()
    update_values = torch.zeros(2)
    A_prime = A.clone()
    A_prime[:,j,idx] = update_values
    return nn.MSELoss()(A, A_prime)

def ZeroRowLoss(A):
    '''
    zeroing out a row
    '''
    A_prime = A.clone()
    A_prime[:,0,:] = 0
    return nn.MSELoss()(A, A_prime)

def ZeroSingularValue(A):
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(A, full_matrices=False)
        s[:,-1] = 0
        s_p = torch.diag_embed(s)
        A_prime = u @ s_p @ vh
    return nn.MSELoss()(A, A_prime)


def train(model, device, trainset, Y, optimizer, epoch, batch_sz, logger, lossfn):

    total_loss = 0
    total_safety_loss = 0
    total_violation_loss = 0
    total_comfort_loss = 0
    total_distance_loss = 0

    data = trainset
    targets = Y
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

    x = data
    nBatch = x.size(0)

    # split into image and speed
    image = x.clone()[:, :IMG_DIMENSIONS[0] * IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2]]
    image = image.reshape(
        (nBatch, IMG_DIMENSIONS[2], IMG_DIMENSIONS[0], IMG_DIMENSIONS[1]))
    image = torch.clamp(image, 0, 1)
    x = model.conv1(image)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = model.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)

    # join back 
    x = torch.cat([x, speed, dest_distance, car_distance], dim=1)

    x = model.fcn(x)
    x = F.relu(x)

    if model.nIneqs > 0:
        x_A = x[:, :(model.nEqs * model.nDecs)]
        x_b = x[:, -(model.nEqs + model.nIneqs):-model.nIneqs]
        A = x_A.reshape(nBatch, model.nEqs, model.nDecs)
        b = x_b.reshape(nBatch, model.nEqs)

        x_G = x[:, (model.nEqs * model.nDecs): (model.nEqs * model.nDecs + model.nIneqs * model.nDecs)]
        x_h = x[:, -model.nIneqs:]
        G = x_G.reshape(nBatch, model.nIneqs, model.nDecs)
        h = x_h.reshape(nBatch, model.nIneqs)
    else:
        x_A = x[:, :(model.nEqs * model.nDecs)]
        x_b = x[:, -model.nEqs:]
        A = x_A.reshape(nBatch, model.nEqs, model.nDecs)
        b = x_b.reshape(nBatch, model.nEqs)
        G = None
        h = None
    
    G_p = G
    # defence
    if model.epsilon != 0:
        # perturb the matrix a little bit to handle the all zeroes case        
        try:
            U, S, V = torch.svd(A)
            S_p = torch.zeros_like(S)
            for i in range(nBatch):
                S_p[i] = torch.clamp(S[i], torch.max(S[i]).item()/(model.epsilon), torch.max(S[i]).item())
                if torch.eq(S_p[i], torch.zeros_like(S_p[i])).all():
                    S_p[i] += 0.01
            A_p = U @ torch.diag_embed(S_p) @ torch.transpose(V, 1, 2)
        except Exception as e:
            # this is a svd cuda bug
            print(e)
            A_p = A

    else:
        A_p = A

    cond = torch.linalg.cond(A_p)
    delta_loss = lossfn(A_p)
    delta_loss.sum().backward()
    optimizer.step()

    total_safety_loss += safety_loss.mean().item()
    total_violation_loss += traffic_violation_loss.mean().item()
    total_comfort_loss += comfort_loss.mean().item()
    total_distance_loss += distance_loss.mean().item()
    total_loss += loss.mean().item() 

    if epoch % 10 == 0 and delta_loss.sum().item() < 50:
        print(f"Delta Loss: {delta_loss.sum().item()}")
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\nSafetyLoss: {:.6f}\tViolationLoss: {:.6f}\tDistanceLoss: {:.6f}\tComfortLoss: {:.6f}'.format(
            epoch, 0 + batch_sz, len(trainset),
            100. * (0 + batch_sz) / len(trainset), loss.mean().item(),
            safety_loss.mean().item(), traffic_violation_loss.mean().item(),
            distance_loss.mean().item(), comfort_loss.mean().item()))

    logger["train_loss_overall"].append(total_loss)
    logger["train_loss_safety"].append(total_safety_loss)
    logger["train_loss_violation"].append(total_violation_loss)
    logger["train_loss_distance"].append(total_distance_loss)
    logger["train_loss_comfort"].append(total_comfort_loss)
    logger["train_loss_delta"].append(delta_loss.sum().item())
    logger["train_loss_cond"].append(cond.sum().item())
    return output

def map_lossfnarg_to_lossfn(s):
    if s == "ZeroSingularValue":
        return ZeroSingularValue
    elif s == "ZeroRowLoss":
        return ZeroRowLoss
    elif s == "ZeroColLoss":
        return ZeroColumnLoss
    elif s == "ConditionNumberLoss":
        return lambda x : -torch.log(torch.linalg.cond(x))
    else:
        print(f"Loss function not found: {s}")
        exit(-1)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="Epochs to run the experiment", type=int,  default=10000)
    parser.add_argument("-d", "--datadir", help="Directory to load the data from",  default="out")
    parser.add_argument("-m", "--modeldir", help="Directory to load models from",  default="better_log")
    parser.add_argument("-r", "--seed", help="Random seed to use", type=int,  default=87)
    parser.add_argument("-l", "--logdir",help="Directory to save logs to, together with data", default="./cond_atk_rebuttal")
    parser.add_argument("-f", "--loss-function",help="Loss function to use", default="ZeroRowLoss")
    parser.add_argument("-t", "--datatype", help="Dataset to Attack", default="train")
    parser.add_argument("-i", "--id", help="Data point to attack", type=int, default=4955)
    parser.add_argument("-b", "--attack-sz", help="Attack size to use", type=int,  default=1)
    parser.add_argument("-g", "--cuda", help="Use GPU", action="store_true", default=False)
    parser.add_argument("-s", "--epsilon", help="Epsilon to use", type=float,  default=0.0)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and args.cuda else "cpu")

    torch.manual_seed(args.seed)
    # import the dataset
    X_train = torch.load(f"{args.datadir}/X_train.pt").to(device)
    Y_train = torch.load(f"{args.datadir}/Y_train.pt").to(device)
    X_test = torch.load(f"{args.datadir}/X_test.pt").to(device)
    Y_test = torch.load(f"{args.datadir}/Y_test.pt").to(device)

    model = QPSpeedPlanning(args.epsilon).to(device)
    model.load_state_dict(torch.load(f"{args.modeldir}/epsilon_{args.epsilon}/model{args.seed}.pt"))

    lossfn = map_lossfnarg_to_lossfn(args.loss_function)
    if args.datatype == "train":
        X_attack = X_train[args.id:args.id+args.attack_sz]
        Y_attack = Y_train[args.id:args.id+args.attack_sz]
    else:
        X_attack = X_train[args.id:args.id+args.attack_sz]
        Y_attack = Y_train[args.id:args.id+args.attack_sz]
    LOGDIR = args.logdir + f'/epsilon_{args.epsilon}/seed{args.seed}'  
    EPOCHS = []
    for i in range(args.attack_sz):
        print(i)
        LOGS = {
            "train_loss_safety": [],
            "train_loss_violation": [],
            "train_loss_comfort": [],
            "train_loss_distance": [],
            "train_loss_overall": [],
            "train_loss_delta": [],
            "train_loss_cond": []
        }
        X = X_attack[i].reshape((1, X_attack.shape[1]))
        Y = Y_attack[i].reshape((1,))
        delta = torch.zeros((1,3,56,56), requires_grad=True).to(device)
        optimizer = optim.Adam([delta], lr=0.005)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        found = False
    
        for epoch in range(1, args.epochs + 1):
            deltap = delta.clone()[0]
            #deltap = nn.ZeroPad2d(12)(deltap)
            #deltap = nn.ZeroPad2d((18,20, 10,16))(deltap)
            deltap_t = torch.zeros_like(X)
            deltap_t[:,:3*56*56] = deltap.reshape(3*56*56)
            deltap = deltap_t
           
            # allow arbitrary perturbations in sticker
            deltap = torch.clamp(deltap, -0.32, 0.32)
            new_image = X[0] + deltap
            #with torch.no_grad():
            #    new_image[:,:56*56*3] = torch.clamp(new_image[:,:56*56*3], 0, 1)
            if epoch % 300 == 0:
                scheduler.step()
            out = train(model, device, new_image, Y, optimizer, epoch, 1, LOGS, lossfn)
            
            if isnan(out.sum().item()):
                print(f"[+] Found adversarial attack for model{args.seed} image {i}")
                found = True
                # found adversarial example
                EPOCHS.append(epoch)

                os.makedirs(LOGDIR, exist_ok=True)
                for k, v in LOGS.items():
                    v_np = np.array(v)
                    np.savetxt(f"{LOGDIR}/img_{i}_{k}_{args.loss_function}_{args.epsilon}.csv", v_np)

                # save the delta
                torch.save(deltap, f"{LOGDIR}/adversarial{i}_{args.loss_function}_{args.epsilon}.pt")

                # save the image
                adv_x = new_image
                image = adv_x.clone()[:, :IMG_DIMENSIONS[0] * IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2]].clamp(0, 1)
                image = image.reshape(
                    (IMG_DIMENSIONS[2], IMG_DIMENSIONS[0], IMG_DIMENSIONS[1]))
                original_img = X[0][:IMG_DIMENSIONS[0] * IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2]].reshape(
                    (IMG_DIMENSIONS[2], IMG_DIMENSIONS[0], IMG_DIMENSIONS[1]))
                im = transforms.ToPILImage()(image)
                org_im = transforms.ToPILImage()(original_img)
                im.save(f"{LOGDIR}/adv_ex{i}_{args.loss_function}_{args.epsilon}.jpg", "JPEG")
                org_im.save(f"{LOGDIR}/adv_ex{i}_{args.loss_function}_{args.epsilon}_original.jpg", "JPEG")
                break  
        if not found:
            EPOCHS.append(-1)
            os.makedirs(LOGDIR, exist_ok=True)
            for k, v in LOGS.items():
                v_np = np.array(v)
                np.savetxt(f"{LOGDIR}/img_{i}_{k}_{args.loss_function}_{args.epsilon}.csv", v_np) 
    os.makedirs(LOGDIR, exist_ok=True)
    np.savetxt(f"{LOGDIR}/epochs_{args.loss_function}_{args.epsilon}.csv", EPOCHS)