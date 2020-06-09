import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd.gradcheck import zero_gradients
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import math
from networks import *
import argparse
from utils import weights_init
from flags import FLAGS
import torch._utils

if not os.path.exists('style_transfer_results_'+FLAGS.folder_save):
		os.makedirs('style_transfer_results_'+FLAGS.folder_save)

encoder = Encoder()
encoder.apply(weights_init)

decoder = Decoder()
decoder.apply(weights_init)

encoder.load_state_dict(torch.load(os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.encoder_save)))
decoder.load_state_dict(torch.load(os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.decoder_save)))

encoder.cuda()
decoder.cuda()

encoder.eval()
decoder.eval()

# load data set and create data loader instance
print('Loading Shapes3D Dataset...')
data_dir = './shapes_data/test/'
transform = transforms.Compose([transforms.Resize((FLAGS.image_size, FLAGS.image_size)), transforms.ToTensor()])
dset = datasets.ImageFolder(data_dir, transform = transform)
celeba = torch.utils.data.DataLoader(dset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)

img1 = next(iter(celeba))[0].cuda()
img2 = next(iter(celeba))[0].cuda()

latent1 = encoder(img1)
latent2 = encoder(img2)

spec_latents1 = latent1[0]
spec_latents2 = latent2[0]

unspec_latents1 = latent1[1]
unspec_latents2 = latent2[1]

# for Z_f
for f in range(FLAGS.z_num_chunks):
	z = []
	for fea in range(FLAGS.z_num_chunks):
		if(fea==f):
			z.append(spec_latents2[fea])
		else:
			z.append(spec_latents1[fea])
	img_new_unspec_1 = decoder(z, unspec_latents1)
	
	save_image(torch.cat([img1[:FLAGS.style_transfer_num_images], img2[:FLAGS.style_transfer_num_images], img_new_unspec_1[:FLAGS.style_transfer_num_images]], dim=0), './style_transfer_results_'+FLAGS.folder_save+'/feature_'+(str)(f)+'_transferred_unspec_1.png', nrow=FLAGS.style_transfer_num_images)
	
# for Z_u
img_new_unspec_2 = decoder(spec_latents1, unspec_latents2)
save_image(torch.cat([img1[:FLAGS.style_transfer_num_images], img2[:FLAGS.style_transfer_num_images], img_new_unspec_2[:FLAGS.style_transfer_num_images]], dim=0), './style_transfer_results_'+FLAGS.folder_save+'/feature_unspecified_transferred.png', nrow=FLAGS.style_transfer_num_images)
