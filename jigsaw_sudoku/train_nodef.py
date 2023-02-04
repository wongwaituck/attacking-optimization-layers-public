#!/usr/bin/env python3

import setGPU
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import natsort
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from qpth.qp import QPFunction
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np

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
            self.total_imgs = natsort.natsorted(all_imgs)[:train_sz]
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

    def forward(self, puzzles, A, b):
        nBatch = puzzles.size(0)

        p = -puzzles.view(nBatch, -1)

        return QPFunction(verbose=-1)(
            self.Q, p, self.G, self.h, A, b
        ).float().view_as(puzzles)

class Net(nn.Module):
    def __init__(self, outputShape):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(43264, 7368)
        self.fcn = nn.Linear(7368, 40 * 64 + 64 + 40)
        self.optnet = OptNet(2, 0.1)
        self.outputShape = outputShape

    def forward(self, x):
        nBatch = x.size(0)
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
        x = self.optnet(x_t, A, b)
        return x

def train(model, device, train_loader, optimizer, epoch, batch_sz, Y, writer, logger):
    # model.train()
    total_loss = 0
    correct = 0
    for idx, img in enumerate(train_loader):
        y_idx_start = batch_sz * idx
        y_idx_end = y_idx_start + batch_sz
        data, target = img.to(device), Y[y_idx_start:y_idx_end].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
        if idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item()))
        for i in range(batch_sz):
            output_i = decode_onehot(output[i])
            target_i = decode_onehot(target[i])
            correct += (output_i.eq(target_i).sum().item() / 16.0)
    total_loss /= len(train_loader.dataset)
    correct /= len(train_loader.dataset)
    writer.add_scalar('Loss/train', total_loss, epoch)
    writer.add_scalar('Accuracy/train', correct, epoch)
    logger["train_loss"].append(total_loss)
    logger["train_acc"].append(correct)

def test(model, device, test_loader, batch_sz, Y, writer, logger):
    model.eval()
    test_loss = 0
    correct = 0
    test_sz = len(test_loader.dataset)
    with torch.no_grad():
        for idx, img in enumerate(test_loader):
            y_idx_start = batch_sz * idx
            y_idx_end = y_idx_start + batch_sz
            data, target = img.to(device), Y[y_idx_start:y_idx_end].to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target).item()   # sum up batch loss
            for i in range(batch_sz):
                output_i = decode_onehot(output[i])
                target_i = decode_onehot(target[i])
                correct += (output_i.eq(target_i).sum().item() / 16.0)
    test_loss = test_loss/len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    correct /= len(test_loader.dataset)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', correct, epoch)
    logger["test_loss"].append(test_loss)
    logger["test_acc"].append(correct)

if __name__=="__main__":
    TOTAL_TRAIN = 70000
    torch.set_default_tensor_type(torch.FloatTensor)

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="Number of Epochs to train", type=int, default=20)
    parser.add_argument("-b", "--batch", help="Batch size", type=int, default=500)
    parser.add_argument("-t", "--train-sz", help="Number of training samples to use", type=int, default=20000)
    parser.add_argument("-s", "--test-pct", help="Percentage of test samples to use", type=float, default=0.2)
    parser.add_argument("-d", "--datadir",help="Directory to read data from, shoudl be produced from create.py", default="./data_one/2")
    parser.add_argument("-m", "--modeldir",help="Directory to save model checkpoints to, together with data", default="./model_nodef")
    parser.add_argument("-r", "--seed", help="Random seed to use", type=int, default=2)
    parser.add_argument("-g", "--cuda", help="Whether to use the GPU", action="store_true", default=True)
    args = parser.parse_args()

    LOG_DIR = f"{args.modeldir}/seed_{args.seed}" 
    EPOCHS = args.epochs
    BATCH_SZ = args.batch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and args.cuda else "cpu")

    TRAIN_SZ = args.train_sz
    TEST_SZ = int(TRAIN_SZ  / (1-args.test_pct) * args.test_pct)

    assert(TRAIN_SZ % BATCH_SZ == 0)
    assert(TEST_SZ % BATCH_SZ == 0)
    assert(TRAIN_SZ + TEST_SZ <= TOTAL_TRAIN)
    
    torch.manual_seed(args.seed)
    print("SEED:", args.seed)
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    LOGS = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    train_kwargs = {'batch_size': BATCH_SZ}
    test_kwargs = {'batch_size': BATCH_SZ}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # load x's
    transform = transforms.Compose([
        transforms.ToTensor()])
    train_dataset = CustomDataSet(f'{args.datadir}/images', train_sz=TRAIN_SZ, transform=transform)
    test_dataset = CustomDataSet(f'{args.datadir}/images', train_sz=TRAIN_SZ, test_sz=TEST_SZ, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,**test_kwargs)

    # load Y
    with open(f'{args.datadir}/labels.pt', 'rb') as f:
        Y = torch.load(f)
        train_Y = Y[:TRAIN_SZ]
        test_Y = Y[TRAIN_SZ:TRAIN_SZ + TEST_SZ]
    instance_shape = Y[0].shape
    model = Net(((BATCH_SZ, instance_shape[0], instance_shape[1], instance_shape[2]))).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch, BATCH_SZ, train_Y, writer, LOGS)
        test(model, device, test_loader, BATCH_SZ, test_Y, writer, LOGS)
        scheduler.step()

    # dump the model and the logs
    torch.save(model.state_dict(), f"{args.modeldir}/model{args.seed}.pt")
    for k, v in LOGS.items():
        v_np = np.array(v)
        np.savetxt(f"{LOG_DIR}/{k}.csv", v)