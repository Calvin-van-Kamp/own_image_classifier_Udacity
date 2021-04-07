from PIL import Image
import numpy as np
import torch


def process_image(image):
    """
    Processes an image for a forward pass through the model.
    Parameters:
        image = str, path to where the test image is located
    Returns:
        A torch tensor
    """
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