
"""author 
   baiyu
"""

import argparse
import glob
import os

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#from PIL import Image
import transforms 
#from torchvision import transforms
from tensorboardX import SummaryWriter
from conf import settings
from utils import *
from lr_scheduler import WarmUpLR
from criterion import LSR

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-e', type=int, default=450, help='training epoches')
    parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    parser.add_argument('-gpus', nargs='+', default=0, help='gpu device')
    args = parser.parse_args()

    #checkpoint directory
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    #tensorboard log directory
    log_path = os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)

    #get dataloader
    train_transforms = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToCVImage(),
        transforms.RandomResizedCrop(settings.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])

    test_transforms = transforms.Compose([
        transforms.ToCVImage(),
        transforms.CenterCrop(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])

    train_dataloader = get_train_dataloader(
        settings.DATA_PATH,
        train_transforms,
        args.b,
        args.w
    )

    test_dataloader = get_test_dataloader(
        settings.DATA_PATH,
        test_transforms,
        args.b,
        args.w
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    net = get_network(args)
    net = init_weights(net)
    net.to(device)

    
    if not isinstance(args.gpus, int):
        net = nn.DataParallel(net, device_ids=args.gpus)

    #visualize the network
    visualize_network(writer, net)

    cross_entropy = nn.CrossEntropyLoss() 
    #loss_function = LSR()
    lsr_loss = LSR()

    #apply no weight decay on bias
    params = split_weights(net)
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    #set up warmup phase learning rate scheduler
    iter_per_epoch = len(train_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #set up training phase learning rate scheduler
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES)
    #train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.e - args.warm)

    best_acc = 0.0
    for epoch in range(1, args.e + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        #training procedure
        net.train()
        
        for batch_index, (images, labels) in enumerate(train_dataloader):
            if epoch <= args.warm:
                warmup_scheduler.step()

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predicts = net(images)
            loss = lsr_loss(predicts, labels)
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\t'.format(
                loss.item(),
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(train_dataloader.dataset),
            ))

            #visualization
            visualize_lastlayer(writer, net, n_iter)
            visualize_train_loss(writer, loss.item(), n_iter)

        visualize_learning_rate(writer, optimizer.param_groups[0]['lr'], epoch)
        visualize_param_hist(writer, net, epoch) 

        net.eval()

        total_loss = 0
        correct = 0
        for images, labels in test_dataloader:

            images = images.to(device)
            labels = labels.to(device)

            predicts = net(images)
            _, preds = predicts.max(1)
            correct += preds.eq(labels).sum().float()

            loss = cross_entropy(predicts, labels)
            total_loss += loss.item()

        test_loss = total_loss / len(test_dataloader)
        acc = correct / len(test_dataloader.dataset)
        print('Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc))
        print()

        visualize_test_loss(writer, test_loss, epoch)
        visualize_test_acc(writer, acc, epoch)

        #save weights file
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue
        
        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    
    writer.close()










    


    

