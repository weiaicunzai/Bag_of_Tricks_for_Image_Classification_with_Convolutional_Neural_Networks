# Bag of Tricks for Image Classification with Convolutional Neural Networks 


experiments on Paper <Bag of Tricks for Image Classification with Convolutional Neural Networks>

I will use CUB_200_2011 dataset instead of ImageNet, vgg network instead of resnet, just
to verify if the tricks were introduced in the Paper would work on other dataset and network


baseline(training from sctrach, no ImageNet pretrain weights are used): 

vgg16 64.60% on cub200_2011 dataset, lr=0.01, batchsize=64

effective of stacking tricks 

|trick|acc|
|:---:|:---:|
|baseline|64.60%|
|xavier init,warmup training|66.07%|
|no weight decay|70.14%|
|label smoothing|71.20%|