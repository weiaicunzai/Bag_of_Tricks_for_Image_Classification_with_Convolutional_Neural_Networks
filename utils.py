
from torch.utils.data import DataLoader

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

    if args.gpu:
        net = net.cuda()
        
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
