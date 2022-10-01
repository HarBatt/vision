import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loading import ImageDataset
from utils import Parameters
from cyclegan import CycleGAN

params = Parameters()
params.data_root = "./data"
params.dataset_name = "thermal" #dataset 
params.print_every = 5 #Log frequency
params.device = "cuda" if torch.cuda.is_available() else "cpu" #Device

params.epoch = 0 #Epoch to start training from 
params.n_epochs = 10 #number of epochs of training
params.learning_rate = 0.0001 #Learning rate for optimizers
params.beta1 = 0.5  #beta1 for Adam optimizer
params.beta2 = 0.999 #beta2 for Adam optimizer
params.decay_epoch = 1  #epoch to start lr decay
params.batch_size = 4 #size of the batches

params.img_height = 384 #height of the input image
params.img_width = 384 #width of the input image
params.channels = 1 #number of channels of the input image
params.n_residual_blocks = 2  #number of residual blocks in the models
params.lambda_cycle = 10.0 #cycle loss weight
params.lambda_identity = 5.0 #identity loss weight


# Image transformations
transformations = [
    transforms.Resize((params.img_height, params.img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
]

root_path = os.path.join(params.data_root, params.dataset_name)
# Training data loader
train_dataloader = DataLoader(
    ImageDataset(root_path, transforms_=transformations, unaligned=True, mode = "train"),
    batch_size= params.batch_size,
    shuffle=True,
)

# Test data loader
val_dataloader = DataLoader(
    ImageDataset(root_path, transforms_=transformations, unaligned=True, mode="test"),
    batch_size= params.batch_size,
    shuffle=True,
)

cyclegan = CycleGAN(params)
cyclegan.train(train_dataloader, val_dataloader)