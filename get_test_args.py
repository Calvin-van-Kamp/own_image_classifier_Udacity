import argparse

def get_test_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method

    # Argument 1: single image path
    parser.add_argument('--path', type = str, default = '', 
                        help = 'path to specific image')
    # Argument 2: checkpoint load
    parser.add_argument('--load_dir', type = str, default = 'checkpoint.pth', 
                        help = 'path to where the checkpoint should be loaded from')
    # Argument 3: how many classes to show
    parser.add_argument('--topk', type = int, default =3, 
                        help = 'how many classes to return for')
    # Argument 4: which dictionary of class names is used
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                        help = 'directory of a dictionary of class names')
    # Argument 5: gpu or not
    parser.add_argument('--gpu', type = bool, default = False, 
                        help = 'gpu or cpu (cpu default)')

    in_args = parser.parse_args()
    return in_args