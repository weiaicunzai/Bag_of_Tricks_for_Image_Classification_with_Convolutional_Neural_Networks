

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
from tensorboardX import SummaryWriter
from conf import settings
from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=12, help='batch size for dataloader')
    parser.add_argument('-lr', type=int, default=0.1, help='initial learning rate')
    parser.add_argument('-e', type=int, default=120, help='training epoches')
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
        transforms.RandomResizedCrop(settings.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop(),
        transforms.Normalize(settings.TEST_MEAN, settings.TEST_STD),
        transforms.ToTensor()
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
    net.to(device)

    #visualize the network
    visualize_network(writer, net)

    loss_function = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES)

    best_acc = 0.0
    for epoch in range(1, args.e):
        scheduler.step()

        #training procedure
        net.train()
        
        for batch_index, (images, labels) in enumerate(train_dataloader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predicts = net(images)
            loss = loss_function(predicts, labels)
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\t'.format(
                loss.item(),
                epoch=epoch,
                trained_samples=batch_index * len(images),
                total_samples=len(train_dataloader.dataset)
            ))

            #visualization
            visualize_lastlayer(writer, net, n_iter)
            visualize_train_loss(writer, loss.item(), n_iter)

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

            loss = loss_function(predicts, labels)
            total += loss.item()


        test_loss = total / len(test_dataloader)
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










    


    

