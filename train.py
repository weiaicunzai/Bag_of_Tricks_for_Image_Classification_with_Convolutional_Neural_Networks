

import argparse

import numpy as np
import cv2
import torch
import glob

#from PIL import Image
import transforms.transforms as transforms
from utils import get_network, get_train_dataloader, get_test_dataloader
from conf import settings

path = '/Users/didi/Downloads/train/2d281959a02178bbcdeea424c8757b1d.jpg'




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='user gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=12, help='batch size for dataloader')
    
    args = parser.parse_args()

    net = get_network(args)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
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



    #cv2.imshow('origin', image)
    #trans = transforms.CenterCrop()
    #image = trans(image)
#trans = transforms.ToFloat()
#image = trans(image)
#print(image.dtype)
    #trans = transforms.RandomResizedCrop(224)
    #image = trans(image)
#print(image.dtype)
    #trans = transforms.RandomHorizontalFlip()
    #image = trans(image)
#print(image.dtype)
    #trans = transforms.ColorJitter()
    #image = trans(image)
#print(image.dtype)
    #trans = transforms.Normalize(mean=[0.7, 0.8, 0.3], std=[0.1, 0.2, 0.3])
    #image = trans(image)
#print(image.dtype)
#trans = transforms.ToTensor()
#image = trans(image)

    #cv2.imshow('test', image)
    #cv2.waitKey(0)
#print(type(image))
#print(image.size())
#print(image.dtype)
#print(torch.max(image))
#cv2.imshow('test', image)
#cv2.waitKey(0)
#print(image.shape)
#
#print(image.dtype)
#print(np.max(image))
#image = image.astype('float32')
#
#print(image.dtype)
#print(np.max(image))
#
#
#
#

#transform_train = transforms.Compose([
#        transforms.ToFloat(), #
#        transforms.RandomResizedCrop(224),
#        transforms.RandomHorizontalFlip(),
#        transforms.ColorJitter(),
#        transforms.Normalize(mean_train, std_train),
#        transforms.ToTensor(),
#    ])
#
#transform_test = transforms.Compose([
#        transforms.CenterCrop(),
#        transforms.Normalize(mean_train, std_train)
#])