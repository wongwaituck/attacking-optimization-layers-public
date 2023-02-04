#!/usr/bin/env python3

from consts import *
from PIL import Image
import argparse
import torch
from torchvision.transforms import Compose, ToTensor, Resize
import os
import random
import glob
from tqdm import tqdm


preprocess = Compose([
    Resize((56,56)),
    ToTensor()
])

def create_sample(datadir, setting="TRAIN"):
    '''
    creates a sample from the training set or test set
    returns X[56 x 56 x 3 + 3], where the first 56 * 56 * 3 are pixels of the image, and the last 3 entries represnet current speed, distance to destination, distance from car ahead respectively.

    note that we don't explicitly need a y since we calculate loss based on the input data 
    '''
    
    # init image dir
    if setting == "TRAIN":
        IMAGE_DIR = f"{datadir}/train"
    else:
        IMAGE_DIR = f"{datadir}/test"
    
    # select a class
    clazz = random.choice(CLASSES)
    clazz_dir = CLASS_MAP_REV[clazz]
    IMAGE_DIR += f"/{clazz_dir}"

    # load the image
    image_paths = glob.glob(f"{IMAGE_DIR}/*.ppm")
    image_path = random.choice(image_paths)
    image = Image.open(image_path)
    x_1 = preprocess(image).reshape((56*56*3))

    # generate speed in meters (between 0 - 100km/h)
    speed = torch.Tensor([random.random() * 100000])

    # create distance to destination (in m) (between 0-10km) 
    dest_distance = torch.Tensor([random.random() * 10 * 1000])

    # create distance to car ahead (some random distance between 0 and the destination)
    car_distance = torch.Tensor([random.uniform(0, dest_distance)])

    # output as single 1D tensor X
    x = torch.cat([x_1, speed, dest_distance, car_distance])
    y = torch.Tensor([clazz])
    return x, y


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help="Train size", type=int, default=10000)
    parser.add_argument("-te", "--test", help="Test size", type=int, default=1000)
    parser.add_argument("-d", "--datadir", help="Directory to load the data from",  default="belgium_ts")
    parser.add_argument("-o", "--outdir", help="Directory to save the created data to",  default="out")
    args = parser.parse_args()
    random.seed(1)

    TRAIN_SZ = args.train
    TEST_SZ = args.test
    IMG_SZ = IMG_DIMENSIONS[0] * IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2]
    DATA_DIR = args.datadir

    X_train = torch.zeros((TRAIN_SZ, IMG_SZ + 3))
    Y_train = torch.zeros((TRAIN_SZ,))
    X_test = torch.zeros((TEST_SZ, IMG_SZ + 3))
    Y_test = torch.zeros((TEST_SZ,))

    for i in tqdm(range(TRAIN_SZ)):
        X, Y = create_sample(DATA_DIR, setting="TRAIN")
        X_train[i] = X
        Y_train[i] = Y

    for i in tqdm(range(TEST_SZ)):
        X, Y = create_sample(DATA_DIR, setting="TEST")
        X_test[i] = X
        Y_test[i] = Y

    # save the tensors
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(X_train, f"{args.outdir}/X_train.pt")
    torch.save(Y_train, f"{args.outdir}/Y_train.pt")
    torch.save(X_test, f"{args.outdir}/X_test.pt")
    torch.save(Y_test, f"{args.outdir}/Y_test.pt")