
import argparse
import glob
import os

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

#from PIL import Image
import transforms 
#from torchvision import transforms
from tensorboardX import SummaryWriter
from conf import settings
from utils import *
from lr_scheduler import FindLR
from criterion import LSR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-base_lr', type=float, default=1e-7, help='min learning rate')
    parser.add_argument('-max_lr', type=float, default=10, help='max learning rate')
    parser.add_argument('-num_iter', type=int, default=100, help='num of iteration')
    parser.add_argument('-gpus', nargs='+', type=int, default=0, help='gpu device')
    args = parser.parse_args()


    train_transforms = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToCVImage(),
        transforms.RandomResizedCrop(settings.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        #transforms.RandomErasing(),
        #transforms.CutOut(56),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])

    train_dataloader = get_train_dataloader(
        settings.DATA_PATH,
        train_transforms,
        args.b,
        args.w
    )

    net = get_network(args)
    net = init_weights(net)

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    
    net = nn.DataParallel(net, device_ids=args.gpus)
    net = net.cuda()

    lsr_loss = LSR()

    #apply no weight decay on bias
    params = split_weights(net)
    optimizer = optim.SGD(params, lr=args.base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    #set up warmup phase learning rate scheduler
    lr_scheduler = FindLR(optimizer, max_lr=args.max_lr, num_iter=args.num_iter)
    epoches = int(args.num_iter / len(train_dataloader)) + 1

    n = 0
    learning_rate = []
    losses = []
    for epoch in range(epoches):

        #training procedure
        net.train()
        
        for batch_index, (images, labels) in enumerate(train_dataloader):
            if n > args.num_iter:
                break

            lr_scheduler.step()

            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            predicts = net(images)
            loss = lsr_loss(predicts, labels)
            if torch.isnan(loss).any():
                n += 1e8
                break
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1
            print('Iterations: {iter_num} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.8f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                iter_num=n,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(train_dataloader.dataset),
            ))

            learning_rate.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())
            n += 1

    learning_rate = learning_rate[10:-5]
    losses = losses[10:-5]

    fig, ax = plt.subplots(1,1)
    ax.plot(learning_rate, losses)
    ax.set_xlabel('learning rate')
    ax.set_ylabel('losses')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

    fig.savefig('result.jpg')

