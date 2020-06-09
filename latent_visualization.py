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
from sklearn.manifold import TSNE
import argparse
from utils import weights_init
import torch._utils
from flags import FLAGS
try:
	torch._utils._rebuild_tensor_v2
except AttributeError:
	def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
		tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
		tensor.requires_grad = requires_grad
		tensor._backward_hooks = backward_hooks
		return tensor
	torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
from flags import FLAGS

COLOUR_NAMES = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan']

if __name__ == '__main__':
	"""
	models
	"""
	encoder = Encoder()
	encoder.apply(weights_init)

	decoder = Decoder()
	decoder.apply(weights_init)

	encoder.load_state_dict(torch.load(os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.encoder_save)))
	decoder.load_state_dict(torch.load(os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.decoder_save)))
	
	encoder = encoder.cuda()
	decoder = decoder.cuda()

	encoder.eval()
	decoder.eval()

# load data set and create data loader instance
print('Loading Shapes3D Dataset...')
data_dir = './shapes_data/test/'
transform = transforms.Compose([transforms.Resize((FLAGS.image_size, FLAGS.image_size)), transforms.ToTensor()])
dset = datasets.ImageFolder(data_dir, transform = transform)
sprites = torch.utils.data.DataLoader(dset, batch_size=FLAGS.test_batch_size, shuffle=True, drop_last=True)

spec_chunks = [[] for i in range(FLAGS.z_num_chunks)]
all_chunks = [[] for i in range(FLAGS.z_num_chunks + 1)]
label = []

for i,(img,labels) in enumerate(sprites):
	enc = encoder(Variable(img.view(FLAGS.test_batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size).cuda()))
	spec_latent = enc[0]
	unspec_latent = enc[1]
	for chunk in range(FLAGS.z_num_chunks):
		spec_chunks[chunk].append(spec_latent[chunk].data.view(-1).cpu().numpy())
		all_chunks[chunk].append(spec_latent[chunk].data.view(-1).cpu().numpy())
	all_chunks[FLAGS.z_num_chunks].append(unspec_latent.data.view(-1).cpu().numpy())
	if (i == FLAGS.tsne_num_points):
		break

l = len(spec_chunks[0])
vis_chunks = spec_chunks[0] 
for chunk in range(1, FLAGS.z_num_chunks+1):
	vis_chunks+=all_chunks[chunk]
vis_chunks = np.array(vis_chunks)
print(vis_chunks.shape)

X = TSNE(n_components=2, perplexity=100).fit_transform(vis_chunks)

# PLOT SPECIFIED CHUNKS ONLY

vis_x=[]
vis_y=[]

for i in range(FLAGS.z_num_chunks):
	vis_x.append(X[ i*l : (i+1)*l, 0])
	vis_y.append(X[ i*l : (i+1)*l, 1])

fig, ax = plt.subplots(1)
ax.set_yticklabels([])
ax.set_xticklabels([])

for i in range(FLAGS.z_num_chunks):
	plt.scatter(vis_x[i], vis_y[i], marker='.', c=COLOUR_NAMES[i], cmap=plt.cm.get_cmap("jet", 10), s=1)

plt.axis('off')
plt.clim(-0.5, 10.5)

plt.savefig('tsne_factors_of_variation_'+FLAGS.folder_save+'_VAE_specified')
	

# PLOT ALL CHUNKS (SPECIFIED + UNSPECIFIED)

vis_x=[]
vis_y=[]

for i in range(FLAGS.z_num_chunks+1):
	vis_x.append(X[ i*l : (i+1)*l, 0])
	vis_y.append(X[ i*l : (i+1)*l, 1])

fig, ax = plt.subplots(1)
ax.set_yticklabels([])
ax.set_xticklabels([])

for i in range(FLAGS.z_num_chunks+1):
	plt.scatter(vis_x[i], vis_y[i], marker='.', c=COLOUR_NAMES[i], cmap=plt.cm.get_cmap("jet", 10), s=1)

plt.axis('off')
plt.clim(-0.5, 10.5)

plt.savefig('tsne_factors_of_variation_'+FLAGS.folder_save+'_all')
