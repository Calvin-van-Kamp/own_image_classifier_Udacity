import argparse

def get_train_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method

    # Argument 1: data directory
    parser.add_argument('--dir', type = str, default = 'flowers/', 
                        help = 'path to the folder of flower images')
    # Argument 2: save checkpoint or not
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', 
                        help = 'path to where the checkpoint should be saved')
    # Argument 3: neural network architechture
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                        help = 'Neural Network to use out of vgg16 and densenet121')
    # Argument 4: learning rate
    parser.add_argument('--learning_rate', type = float, default = 0.001, 
                        help = 'learning rate')
    # Argument 5: hidden units
    parser.add_argument('--hidden_units', type = int, default = 860,
                        help = 'hidden units, 860 by default')
    # Argument 6: epochs
    parser.add_argument('--epochs', type = int, default = 5, 
                        help = 'epochs')
    # Argument 7: gpu or not
    parser.add_argument('--gpu', type = bool, default = False, 
                        help = 'gpu or cpu (cpu default)')

    in_args = parser.parse_args()
    return in_args