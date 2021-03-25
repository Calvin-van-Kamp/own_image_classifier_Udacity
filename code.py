###BLOCK 1###

# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import collections
#import fc_model

from workspace_utils import active_session
from time import time

###################################################################
import helper

###BLOCK 2###

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

###BLOCK 3###

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

##########################################################################################################
data_iter = iter(testloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)

###BLOCK 4###

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
###BLOCK 5###

#Get the pre-trained neural network
#Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = models.vgg16(pretrained=True)

#Freeze parameters so we don't backprop through them
for param in vgg.parameters():
    param.requires_grad = False

vgg.classifier = nn.Sequential (nn.Linear(25088,1024), 
                                nn.ReLU(), 
                                nn.Dropout(0.2), 
                                nn.Linear(1024,102),
                                nn.LogSoftmax (dim = 1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(vgg.classifier.parameters(), lr=0.001)

vgg.to(device);

###BLOCK 6###

start = time()
with active_session():
    train_loss_list = valid_loss_list = valid_accuracy_list = []
    #Check the pretrained neural network
    epochs = 30
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            #Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            #Prevent gradients from being stored
            optimizer.zero_grad()
            #Forward and backprop
            logps = vgg.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            valid_loss = 0
            valid_accuracy = 0
            
            with torch.no_grad():
                vgg.eval()
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = vgg.forward(inputs)
                    valid_batch_loss = criterion(logps, labels)

                    valid_loss += valid_batch_loss.item()

                    # Calculate validation accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            vgg.train()
#             running_loss = 0
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")

            train_loss_list.append(running_loss)
            valid_loss_list.append(valid_loss/len(validloader))
            valid_accuracy_list.append(valid_accuracy/len(validloader))
            running_loss = 0
            
print('Time taken to run: {}'.format(time() - start))