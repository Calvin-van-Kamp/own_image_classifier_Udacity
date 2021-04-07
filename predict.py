from get_test_args import get_test_args
from load_checkpoint import load_checkpoint
from process_image import process_image
from predict_func import predict_func
import torch

#path, load_dir, top-k, category_names, gpu
inputs = get_test_args()

#load the checkpoint
model = load_checkpoint(inputs.load_dir)

#put into cpu or gpu, first checking for availability
if torch.cuda.is_available():
    device = torch.device("cuda" if inputs.gpu else "cpu")
    model.to(device)

else:
    if inputs.gpu:
        print('No GPU was found and CPU will be used')
    device = torch.device("cpu")
    model.to(device)

#get the top classes
top_probs, top_classes, class_names = predict_func(inputs.path, model, inputs.topk, device, inputs.category_names)

#return results:
print('The top {} probabilities are: {}'.format(inputs.topk, top_probs))
print('The top {} class_names are: {}'.format(inputs.topk, class_names))