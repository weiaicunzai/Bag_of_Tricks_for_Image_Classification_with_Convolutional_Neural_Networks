# Bag of Tricks for Image Classification with Convolutional Neural Networks 


experiments on Paper <Bag of Tricks for Image Classification with Convolutional Neural Networks>

we will use CUB_200_2011 dataset instead of ImageNet, vgg network instead of resnet, just
to verify if the tricks are introduced in the Paper would work on other dataset and network


baseline: 

vgg16 64.06% on cub200_2011 dataset, lr=0.1, batchsize=64

+xavier init 
+warmup training:

improved acc from 64.06% to 66.07%

