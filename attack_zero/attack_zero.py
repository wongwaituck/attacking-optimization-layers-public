#!/usr/bin/env python3

import setGPU
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import natsort
from PIL import Image
from qpth.qp import QPFunction
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import numpy as np
import argparse


def decode_onehot(encoded_board):
    """Take the unique argmax of the one-hot encoded board."""
    v,I = torch.max(encoded_board, 0)
    return ((v>0).long()*(I+1)).squeeze()


class CustomDataSet(Dataset):
    def __init__(self, main_dir, train_sz, test_sz=0, train=True, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        if train:
            self.total_imgs = natsort.natsorted(all_imgs)[OFFSET:OFFSET+train_sz]
        else:
            self.total_imgs = natsort.natsorted(all_imgs)[train_sz:train_sz + test_sz]
    
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


class OptNet(nn.Module):
    def __init__(self, n, Qpenalty):
        super().__init__()
        nx = (n**2)**3
        self.Q = Variable(Qpenalty*torch.eye(nx)).cuda()
        self.G = Variable(torch.zeros((nx,nx))).cuda()
        self.h = Variable(torch.zeros(nx)).cuda()
        self.Qpenalty = Qpenalty

    def forward(self, puzzles, A, b):
        nBatch = puzzles.size(0)

        p = -puzzles.view(nBatch, -1)

        res = QPFunction(verbose=-1)(
            self.Q, p, self.G, self.h, A, b
        ).float().view_as(puzzles)
        return res

class Net(nn.Module):
    def __init__(self, outputShape, epsilon=0):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(43264, 7368)
        self.fcn = nn.Linear(7368, 40 * 64 + 64 + 40)
        self.optnet = OptNet(2, 0.1)
        self.outputShape = outputShape
        self.epsilon = epsilon

    def forward(self, x, use_optnet=True):
        nBatch = x.size(0)
        Nsq = self.outputShape[1]
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fcn(x)
        x = F.relu(x)
        x_t = x[:, :64]
        x_b = x[:, 64:-40]
        x_bb = x[:, -40:]
        x_t = x_t.reshape(self.outputShape)
        x_t = F.softmax(x_t, dim=3)

        A = x_b.reshape(nBatch, 40, 64)
        b = x_bb.reshape(nBatch, 40)
        
        if self.epsilon != 0:
            U, S, V = torch.svd(A)
            S_p = torch.zeros_like(S)
            for i in range(nBatch):
                S_p[i] = torch.clamp(S[i], torch.max(S[i]).item()/(self.epsilon), torch.max(S[i]).item())
        
            A_p = U @ torch.diag_embed(S_p) @ torch.transpose(V, 1, 2)
        else:
            A_p = A

        if use_optnet:
            x = self.optnet(x_t, A_p, b)
        
        
        # # also do a forward pass on other solvers
        # Q = (torch.eye(64) * self.optnet.Qpenalty).cpu().detach().numpy()
        # p  = x_t[0].cpu().detach().view(nBatch, -1).numpy()
        # A = A[0].cpu().detach().numpy()
        # b = b[0].cpu().detach().numpy()
       

        # # CVXPY Solvers
        # z = cp.Variable(64)
        # prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(z, Q) - p @ z),
        #          [A @ z == b])
        # for solver in cp.installed_solvers():
        #     prob.solve(solver=solver, verbose=False)
        #     print(f"{solver}'s optimal value is", prob.value)

        return x

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
    idx = torch.zeros(40).long().cuda()
    j = torch.arange(40).long().cuda()
    update_values = torch.zeros(40).cuda()
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

def ZeroMatrixLoss(A):
    '''
    try to attack the zero matrix
    '''
    A_prime = torch.zeros_like(A)
    return nn.MSELoss()(A, A_prime)


def map_lossfnarg_to_lossfn(s):
    if s == "ZeroSingularValue":
        return ZeroSingularValue
    elif s == "ZeroRowLoss":
        return ZeroRowLoss
    elif s == "ZeroColLoss":
        return ZeroColumnLoss
    elif s == "ConditionNumberLoss":
        return lambda x : -torch.log(torch.linalg.cond(x))
    elif s == "ZeroMatrixLoss":
        return ZeroMatrixLoss
    else:
        print(f"Loss function not found: {s}")
        exit(-1)


if __name__=="__main__":
    use_cuda = torch.cuda.is_available()
    assert(use_cuda)
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--epochs", help="Number of Epochs to train", type=int, default=1000)
    parser.add_argument("-t", "--train-sz", help="Number of training samples to use", type=int, default=100)
    parser.add_argument("-s", "--test-offset", help="Offset to the test samples to use", type=int, default=50000)
    parser.add_argument("-d", "--datadir",help="Directory to read data from, shoudl be produced from create.py", default="./data_one/2")
    parser.add_argument("-l", "--logdir",help="Directory to save logs to, together with data", default="./cond_atk_loss_fn/")
    parser.add_argument("-f", "--loss-function",help="Loss function to use", default="ZeroRowLoss")
    parser.add_argument("-m", "--modeldir",help="Directory to the model", default="model_nodef")
    parser.add_argument("-lr", "--learning-rate",help="learning rate to use in the attack finding", type=float, default=1)
    parser.add_argument("-eps", "--epsilon",help="epsilon to use in the defense", type=float, default=0)
    parser.add_argument("-r", "--seed", help="Random seed to use", type=int, default=3)
    args = parser.parse_args()


    torch.manual_seed(args.seed)

    TRAIN_SZ = args.train_sz
    BATCH_SZ = 1
    EPOCHS = args.epochs
    OFFSET = args.test_offset
    PRINT_FREQ = 10
    LOG_DIR = args.logdir

    train_kwargs = {'batch_size': BATCH_SZ}
    # load x's and the model
    transform = transforms.Compose([
            transforms.ToTensor()])
    train_dataset = CustomDataSet(f'{args.datadir}/images', train_sz=TRAIN_SZ, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)

    with open(f'{args.datadir}/labels.pt', 'rb') as f:
        Y = torch.load(f)
        train_Y = Y[OFFSET:OFFSET+TRAIN_SZ].to(device)
    instance_shape = Y[0].shape

    attackerLossFn = map_lossfnarg_to_lossfn(args.loss_function)

    for i in range(1, 2):
        model = Net(((BATCH_SZ, instance_shape[0], instance_shape[1], instance_shape[2])), args.epsilon).to(device)
        model.load_state_dict(torch.load(f"{args.modeldir}/model{i}.pt"))
        model.eval()
        seed_log_dir = f"{LOG_DIR}seed{i}"
        os.makedirs(seed_log_dir, exist_ok=True)
        epochs = []
        for idx, img in enumerate(train_loader):
            x = img.to(device)
            delta = torch.zeros_like(x, requires_grad=True)
            # opt = optim.SGD([delta], lr=4000000, momentum=0.9)
            opt = optim.SGD([delta], lr=args.learning_rate, momentum=0.9)
            condition_numbers = []
            target_losses = []
            class_losses = []
            found = False
            for t in range(1, EPOCHS + 1):
                x = img.to(device)
                current_x = x
                nBatch = x.size(0)
                Nsq = model.outputShape[1]

                x = torch.clamp(x + delta, 0, 1)
                x = model.conv1(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = model.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = torch.flatten(x, 1)
                x = model.fc1(x)
                x = F.relu(x)
                x = model.fcn(x)
                x = F.relu(x)
                x_t = x[:, :64]
                x_b = x[:, 64:-40]
                x_bb = x[:, -40:]
                x_t = x_t.reshape(model.outputShape)
                ex_t = x_t.exp()
                x_t = F.softmax(x_t, dim=3)

                A = x_b.reshape(nBatch, 40, 64)
                cond_num = torch.linalg.cond(A)
                condition_numbers.append(cond_num)
                
                # calculate loss
                loss = attackerLossFn(A)
                if t%PRINT_FREQ == 0:
                    print(f"[*] Model {i} image {idx} loss from attack steps ", t, loss.sum().item())
                    
                opt.zero_grad()
                loss.sum().backward()
                opt.step()
                target_losses.append(loss.sum().item())

                # See if it still works
                y_pred = model(torch.clamp(current_x + delta, 0, 1))
                loss = F.mse_loss(y_pred[0], train_Y[idx], reduction='sum').item()
                class_losses.append(loss)
                if math.isnan(loss):
                    print(f"[+] Found adversarial attack for model{i} image {idx}")
                    torch.save(delta, f"{seed_log_dir}/adversarial{idx}_{args.loss_function}_{args.epsilon}.pt")
                    im = transforms.ToPILImage()((current_x+delta)[0])
                    im.save(f"{seed_log_dir}/adv_ex{idx}_{args.loss_function}_{args.epsilon}.jpg", "JPEG")
                    epochs.append(t)
                    found = True
                    break
            np_condnums = np.array(condition_numbers)
            np.savetxt(f"{seed_log_dir}/img{idx}_cond_{args.loss_function}_{args.epsilon}.csv", np_condnums)
            np_targetlosses = np.array(target_losses)
            np.savetxt(f"{seed_log_dir}/img{idx}_targetlosses_{args.loss_function}_{args.epsilon}.csv", np_targetlosses)
            np_classlosses = np.array(class_losses)
            np.savetxt(f"{seed_log_dir}/img{idx}_classlosses_{args.loss_function}_{args.epsilon}.csv", np_classlosses)
            if not found:
                print(f"[-] Cannot find adversarial attack for model{i} image {idx}")
                epochs.append(-1)
        np_epochs = np.array(epochs)
        np.savetxt(f"{seed_log_dir}/epochs_{args.loss_function}_{args.epsilon}.csv", np_epochs)