


import numpy as np
import cv2
import torch
import glob

from PIL import Image
from transforms import transforms


path = '/Users/didi/Downloads/train/2d281959a02178bbcdeea424c8757b1d.jpg'

#data_r = np.array([])
#data_g = np.array([])
#data_b = np.array([])
data_r = []
data_g = []
data_b = []

for i in glob.iglob('/Users/didi/Downloads/train/*.jpg'):
    image_cv = cv2.imread(i)
    b, g, r = cv2.split(image_cv)
    data_b.append(b.flatten())
    data_r.append(r.flatten())
    data_g.append(g.flatten())
    
print('fff')
data_b = np.hstack(data_b)
data_r = np.hstack(data_r)
data_g = np.hstack(data_g)
print('fff')
#print('b')
#data_g = np.hstack([cv2.imread(i)[:, :, 1].flatten() for i in glob.iglob('/Users/didi/Downloads/train/*.jpg')])
#print('g')
#data_r = np.hstack([cv2.imread(i)[:, :, 2].flatten() for i in glob.iglob('/Users/didi/Downloads/train/*.jpg')])
#print('r')
print(np.mean(data_b) / 255.0, np.mean(data_g) / 255.0, np.mean(data_r) / 255.0)
print(np.std(data_b) / 255.0, np.std(data_g) / 255.0, np.std(data_r) / 255.0)
    #image_cv = cv2.merge((r, g, b))


    #image_cv = cv2.resize(image_cv, (224, 224), interpolation=cv2.INTER_LINEAR)
    #print(image_cv.dtype)

    #image_pil = Image.open(i)
    #image_pil = image_pil.resize((224, 224), resample=Image.BILINEAR)

    ##print(image_cv.dtype)
    ##print(image_pil.dtype)

    #image_pil.show('pil')

    #image_pil = np.array(image_pil)
    #print(image_pil.dtype)

    #r, g, b = cv2.split(image_pil)
    #image_pil = cv2.merge((b, g, r))

    #print(np.mean(image_cv - image_pil))

    #diff = image_cv - image_pil
    #print('diff ', np.max(diff))
    #print('diff ', np.min(diff))
    #diff = diff + - np.min(diff)
    #diff = diff / np.max(diff) * 255

    #print('diff ', np.max(diff))
    #print('diff ', np.min(diff))

    #cv2.imshow('diff', diff)
    #cv2.imshow('cv2', image_cv)
    #cv2.waitKey(0)




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