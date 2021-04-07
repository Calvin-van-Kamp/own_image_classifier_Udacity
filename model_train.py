import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import models
import re
from workspace_utils import active_session



def model_train(dataloaders_dict, gpu, epochs, learning_rate, model):
    """
    Trains an initialised model, prints out feedback and returns a trained model
    Parameters:
        dataloaders = dict, contains dataloaders to iterate over:
            ['trainloaders'] = torch dataloader for the training data
            ['validloaders'] = torch dataloader for the validation data
            ['testloaders'] = torch dataloader for the test data (not used here)
        gpu = boolean, determines whether the gpu is used or not
        epochs = int, the number of times the data is iterated over while training
        learning_rate = float, learning rate used to trin the model
        model = torch.nn, initialised model to be trained
    Returns:
        A trained model ready to checkpoint
    """
    
    #put into cpu or gpu, first checking for availability
    if torch.cuda.is_available():
        device = torch.device("cuda" if gpu else "cpu")
        model.to(device)

    else:
        if gpu:
            print('No GPU was found and CPU will be used')
        device = torch.device("cpu")
        model.to(device)
    
    #get GPU or CPU as string for ease
    device_name = "cuda" if (gpu == True) else "cpu"

    #specify the type of loss
    criterion = nn.NLLLoss()

    #only train the classifier parameters, feature parameters are frozen
    model_optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


    with active_session():
        model_train_loss_list = []
        model_valid_loss_list = []
        model_valid_accuracy_list = []
        #Check the pretrained neural network
        running_loss = 0
        for epoch in range(epochs):
            for inputs, labels in dataloaders_dict['trainloader']:
                #Move input and label tensors to the selected device
                if device_name == "cuda":
                    inputs, labels = inputs.to(device), labels.to(device)
                #Prevent gradients from being stored
                model_optimizer.zero_grad()
                #Forward and backprop
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                model_optimizer.step()

                running_loss += loss.item()
            else:
                valid_loss = 0
                valid_accuracy = 0

                with torch.no_grad():
                    model.eval()
                    for inputs, labels in dataloaders_dict['validloader']:
                        #Move input and label tensors to the selected device
                        if device_name == "cuda":
                            inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        valid_batch_loss = criterion(logps, labels)

                        valid_loss += valid_batch_loss.item()

                        # Calculate validation accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                model.train()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/len(dataloaders_dict['trainloader']):.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders_dict['validloader']):.3f}.. "
                      f"Validation accuracy: {valid_accuracy/len(dataloaders_dict['validloader']):.3f}")

                model_train_loss_list.append(running_loss/len(dataloaders_dict['trainloader']))
                model_valid_loss_list.append(valid_loss/len(dataloaders_dict['validloader']))
                model_valid_accuracy_list.append(valid_accuracy/len(dataloaders_dict['validloader']))
                running_loss = 0

    return model
