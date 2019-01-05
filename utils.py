

import os

import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from conf import settings
from dataset.dataset import CUB_200_2011_Train, CUB_200_2011_Test

def get_network(args):

    if args.net == 'vgg16':
        from models.vgg import vgg16
        net = vgg16()

    elif args.net == 'vgg11':
        from models.vgg import vgg11
        net = vgg11()
    
    elif args.net == 'vgg13':
        from models.vgg import vgg13
        net = vgg13()
    
    elif args.net == 'vgg19':
        from models.vgg import vgg19
        net = vgg19()

    return net

def get_train_dataloader(path, transforms, batch_size, num_workers):
    """ return training dataloader
    Args:
        path: path to CUB_200_2011 dataset
        transforms: transforms of dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
    Returns: train_data_loader:torch dataloader object
    """
    train_dataset = CUB_200_2011_Train(path, transform=transforms)
    train_dataloader =  DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_dataloader

def get_test_dataloader(path, transforms, batch_size, num_workers):
    """ return training dataloader
    Args:
        path: path to CUB_200_2011 dataset
        transforms: transforms of dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
    Returns: train_data_loader:torch dataloader object
    """
    test_dataset = CUB_200_2011_Test(path, transform=transforms)
    test_dataloader =  DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return test_dataloader

def get_lastlayer_params(net):
    """get last trainable layer of a net
    Args:
        network architectur
    
    Returns:
        last layer weights and last layer bias
    """
    last_layer_weights = None
    last_layer_bias = None
    for name, para in net.named_parameters():
        if 'weight' in name:
            last_layer_weights = para
        if 'bias' in name:
            last_layer_bias = para
        
    return last_layer_weights, last_layer_bias

def visualize_network(writer, net):
    """visualize network architecture"""
    input_tensor = torch.Tensor(3, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE) 
    input_tensor.to(next(net.parameters()).device)
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

def visualize_lastlayer(writer, net, n_iter):
    """visualize last layer grads"""
    weights, bias = get_lastlayer_params(net)
    writer.add_scalar('LastLayerGradients/grad_norm2_weights', weights.grad.norm(), n_iter)
    writer.add_scalar('LastLayerGradients/grad_norm2_bias', bias.grad.norm(), n_iter)

def visualize_train_loss(writer, loss, n_iter):
    """visualize training loss"""
    writer.add_scalar('Train/loss', loss, n_iter)

def visualize_param_hist(writer, net, epoch):
    """visualize histogram of params"""
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def visualize_test_loss(writer, loss, epoch):
    """visualize test loss"""
    writer.add_scalar('Test/loss', loss, epoch)

def visualize_test_acc(writer, acc, epoch):
    """visualize test acc"""
    writer.add_scaler('Test/Accuracy', acc, epoch)
