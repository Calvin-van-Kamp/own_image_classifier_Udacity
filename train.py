from get_train_args import get_train_args
from model_init import model_init
from load_data import load_data
from model_train import model_train
from save_checkpoint import save_checkpoint

#dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu
inputs = get_train_args()

#create dataloaders
dataloaders_dict = load_data(inputs.dir)

#Initialize a model
model = model_init(inputs.arch, inputs.hidden_units)

#train the model
final_model = model_train(dataloaders_dict, inputs.gpu, inputs.epochs, inputs.learning_rate, model)

#save checkpoint
save_checkpoint(inputs.dir, inputs.save_dir, final_model, inputs.epochs, inputs.arch)