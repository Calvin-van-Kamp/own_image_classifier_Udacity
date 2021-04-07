from process_image import process_image
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import collections
import re
from PIL import Image
from cat_to_name import cat_to_name_dict

def predict_func(image_path, model, topk, device, categories):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        Dependent on process_image().
        Parameters:
            image_path: str, path to the test image
            model = torch.nn, pre-trained model
            topk = top how many classes returned
            device = whether to run on gpu or cpu
            categories = str, path to a json file where categories are stored
        Returns:
            list 1: a list of the top top_k class probabilities
            list 2: a list of the top top_k class numbers
            list 3: a list of the top top_k class names
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
    #get a list of all classes
    cat_to_name = cat_to_name_dict(categories)
    all_class_names = []
    for x in model.class_to_idx:
        all_class_names.append(cat_to_name['{}'.format(x)])
    #get the top class names out
    i = top_class[0]
    class_names = []
    for m in i:
        class_names.append(all_class_names[m])
        
    return top_p[0].tolist(), top_class[0].add(1).tolist(), class_names