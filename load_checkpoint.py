import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import models

def load_checkpoint(filepath):
    """
    Loads a previously saved checkpoint
    Parameters:
        filepath: str, path to the previously saved checkpoint
    Returns:
        nn.torch, a pre-trained model
    """
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    #no back-prop needed
    for param in model.parameters():
        param.requires_grad = False
        
    return model