import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from itertools import cycle
from torchvision import datasets
from torchvision import transforms
# from utils import reparameterize
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse
import getpass

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")
parser.add_argument('--dataset', type=str, default='mnist', help="dataset to be used for training/testing (gray_shapes/moving_mnist)")
parser.add_argument('--num_specified_chunks', type=int, default=4, help="number of specified chunks")
parser.add_argument('--num_unspecified_chunks', type=int, default=1, help="number of unspecified chunks")
parser.add_argument('--z_chunk_size', type=int, default=32, help="size of each feature")
parser.add_argument('--z_num_chunks', type=int, default=4, help="number of features")
parser.add_argument('--c_chunk_size', type=int, default=20, help="size of each context vector chunk")
parser.add_argument('--c_num_chunks', type=int, default=5, help="number of context vector chunks")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--test_batch_size', type=int, default=1, help="test batch size")
parser.add_argument('--image_size', type=int, default=64, help="height and width of the image")
parser.add_argument('--num_channels', type=int, default=3, help="number of channels in the images")
parser.add_argument('--pred_mask_ratio', type=int, default=0.6, help="probability of having a zero in the mask vector")
parser.add_argument('--kl_divergence_coef', type=int, default=1, help="coefficient for kl-divergence loss")
parser.add_argument('--center_loss_coef', type=int, default=0.2, help="coefficient for center loss")
parser.add_argument('--context_weight', type=int, default=0.1, help="weight for previous context vector")

# optimization parameters
parser.add_argument('--lrate', type=float, default=0.0001, help="initial learning rate")
parser.add_argument('--center_loss_lrate', type=float, default=0.001, help="learning rate for center loss update")
parser.add_argument('--beta_1', type=float, default=0.5, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")

# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder', help="model save for decoder")
parser.add_argument('--prediction_save', type=str, default='prediction', help="model save for prediction")
parser.add_argument('--context_vector_save', type=str, default='context_vector', help="model save for context vector")
parser.add_argument('--context_vector_chunks_save', type=str, default='context_vector_chunks', help="model save for context vector chunks")
parser.add_argument('--folder_save', type=str, default='augmentation_colour_and_rotation')

parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")

parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=120, help="flag to indicate the final epoch of training")
parser.add_argument('--is_training', type=bool, default=True, help="flag to indicate if it is training or inference.")

# tsne plot
parser.add_argument('--tsne_num_points', type=int, default=5000, help="number of points in TSNE plot")

# style transfer
parser.add_argument('--style_transfer_num_images', type=int, default=8, help="number of images in style transfer in each row")

FLAGS = parser.parse_args()

