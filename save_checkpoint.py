import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models



def save_checkpoint(data_dir, save_dir, model, epochs, arch):
    """
    Checkpoints a given model to a given directory.
    Parameters:
        data_dir = str, directory to the image folder (containing at least a folder "train")
        save_dir = str, where and how the model checkpoint should be stored
        model = torch.nn, a trained neural network
        epochs = int, number of passes through the data performed to train the model
        arch = str, name of the model architecture used
    Returns:
        nothing, merely checkpoints the model
    """
    #get classes
    train_data = datasets.ImageFolder(data_dir + '/train')
    model.class_to_idx = train_data.class_to_idx
    #distinguish between input layers
    if arch == 'vgg16':
        input_layers = 25088
    else:
        input_layers = 1024
    #save the model
    model.checkpoint = {'input_size': input_layers,
                        'output_size': 102,
                        'epochs': epochs,
                        'arch': arch,
                        'classifier': model.classifier,
                        'class_to_idx': model.class_to_idx,
                        'state_dict': model.state_dict()}

    torch.save(model.checkpoint, save_dir)