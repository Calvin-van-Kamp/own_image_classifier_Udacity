###BLOCK 1###

# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np

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
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

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

###BLOCK 7###

def testing(model):
    test_losses = []
    test_loss = 0
    accuracy = 0
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate test accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            test_losses.append(accuracy/len(testloader))

    print(f"Test accuracy of : {accuracy/len(testloader):.3f}")
    
    
###BLOCK 8###

# TODO: Save the checkpoint
vgg.class_to_idx = train_data.class_to_idx
vgg.checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'epochs': 5,
                  'arch': 'vgg16',
                  'optimizer': optimizer.state_dict,
                  'classifier': vgg.classifier,
                  'class_to_idx': vgg.class_to_idx,
                  'optimizer_dict': optimizer.state_dict(),
                  'state_dict': vgg.state_dict()}

torch.save(vgg.checkpoint, 'vgg_checkpoint.pth')

###BLOCK 9###

# could not get fc_model to load so I went with this more hands-on approach
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

model = load_checkpoint('vgg_checkpoint.pth')
model.to(device)

###BLOCK 10###

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an torch tensor.
    '''
    image = Image.open(image)
    #resize image
    image.thumbnail((255,255))
    image_w, image_h = image.size
    #crop image
    left = (image_w - 224)/2
    top = (image_h - 224)/2
    right = (image_w + 224)/2
    bottom = (image_h + 224)/2
    image = image.crop((left, top, right, bottom))
    #make an array
    np_image = np.array(image)
    #convert to floats
    np_image = np_image/255.0
    #normalize
    norm_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    #transpose
    output = norm_image.transpose((2, 0, 1))
    
    return torch.from_numpy(output)
  
###BLOCK 11###

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

###BLOCK 12###
  
def predict(image_path, model, topk=5):
  ''' Predict the class (or classes) of an image using a trained deep learning model.
      Dependent on process_images().
  '''
  #get image processed
  input_image = process_image(image_path)
  #make it a tensor so that it can be entered into the model
  input_image = input_image.float()
  input_image = torch.Tensor(input_image)
  #GPU if available
  input_image = input_image.to(device)
  #final changes
  input_image.unsqueeze_(0)
  #get the probabilities
  logps = model(input_image)
  ps = np.exp(logps)
  #get the top "topk" classes
  top_p, top_class = ps.topk(topk, dim=1)
  class_names = []
  for x in classes:
      class_names.append(cat_to_name['{}'.format(x)])
        
  return top_p[0].tolist(), top_class[0].add(1).tolist(), class_names
  
###BLOCK 13###

#get a list of all flower names to subscript
all_class_names = []
for x in model.class_to_idx:
    all_class_names.append(cat_to_name['{}'.format(x)])
    
#process the image and show it
image_path = "flowers/test/1/image_06743.jpg"
imshow(image)
plt.axis('off')
#get model results from image_path
probs, classes, class_names = predict(image_path, model)
#get position of class index from image_path to automate the title
class_pos = re.findall('[0-9]+', image_path)[0]
plt.title(all_class_names[int(class_pos)-1])

###BLOCK 14###

x_pos = [i for i, _ in enumerate(classes)]

plt.barh(x_pos, probs, color= "red")

plt.yticks(x_pos, class_names)
plt.show()
