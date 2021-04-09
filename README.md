# own_image_classifier_Udacity

A neural network classifier written to classify PIL images. Part of Udacity's Neural Networks with Python nanodegree programme.

## Functionality

The goal of this project is to build an easily usable image classifier for images, with multiple pre-trained structures to choose from. 

## Prerequisites

The images to be classified should be in a single folder with sub-folders, namely 
* train
* test
* validation
with images inside for each purpose.

A .json file containing all possible classes should also be provided.

These should be placed in the same folder as the .py files provided here.

## Output

A list of classes and probabilities of what the given test image might be.

## Usage

For training a classifier as well as save it for later use, train.py is used; and for output using a previously saved classifier predict.py is used.

For example, training a model using a GPU would look as follows:
```
python train.py --dir '/image_folder' gpu-- True
```

And using it:
```
python predict.py --gpu True --path 'image_folder/test/1/image_06743.jpg'
```
