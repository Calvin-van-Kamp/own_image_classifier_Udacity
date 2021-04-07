import torch
from torch import nn
from torch import optim
from torchvision import models

#get the pre-trained neural network
def model_init(arch, hidden_units):
    """
    Returns an initialised model ready for training.
    Parameters:
        arch = str, the name of the chosen model's architechture out of 'vgg16' and 'densenet121'.
        hidden_units = int, number of hidden units used for the classifier
    Returns:
        A model ready for training by model_train.py
    """
    #choose architecture
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        #freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        #create the classifier
        model.classifier = nn.Sequential (nn.Linear(25088,hidden_units),
                                          nn.ReLU(),
                                          nn.Dropout(0.2),
                                          nn.Linear(hidden_units,102),
                                          nn.LogSoftmax (dim = 1))
    
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        #freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        #create the classifier
        model.classifier = nn.Sequential (nn.Linear(1024,hidden_units),
                                          nn.ReLU(),
                                          nn.Dropout(0.2),
                                          nn.Linear(hidden_units,102),
                                          nn.LogSoftmax (dim = 1))
    
    else:
        print('YOUR CHOSEN ARCHITECTURE IS NOT SUPPORTED, VGG16 WILL BE USED BY DEFAULT')
        model = models.vgg16(pretrained=True)
        #freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        #create the classifier
        model.classifier = nn.Sequential (nn.Linear(25088,hidden_units),
                                          nn.ReLU(),
                                          nn.Dropout(0.2),
                                          nn.Linear(hidden_units,102),
                                          nn.LogSoftmax (dim = 1))
        
    return model
