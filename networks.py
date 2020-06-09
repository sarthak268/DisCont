import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np
from utils import imshow_grid
from utils import transform_config

from itertools import cycle
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from flags import FLAGS

from torch.distributions.normal import Normal
import random

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=FLAGS.num_channels, out_channels=64, kernel_size=4, stride=2, bias=True)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=True)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, bias=True)
		self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, bias=True)
		
		self.features1 = nn.Linear(in_features=4608, out_features=1024, bias=True)
		self.features2 = nn.Linear(in_features=1024, out_features=(FLAGS.num_specified_chunks + FLAGS.num_unspecified_chunks)*FLAGS.z_chunk_size, bias=True)

		self.linears_z = nn.ModuleList([nn.Linear(in_features=(FLAGS.num_specified_chunks + FLAGS.num_unspecified_chunks)*FLAGS.z_chunk_size, out_features=FLAGS.z_chunk_size, bias=True) for i in range(FLAGS.num_specified_chunks)])
		
		self.linear_mean = nn.Linear((FLAGS.num_specified_chunks + FLAGS.num_unspecified_chunks)*FLAGS.z_chunk_size, FLAGS.z_chunk_size)
		self.linear_std = nn.Linear((FLAGS.num_specified_chunks + FLAGS.num_unspecified_chunks)*FLAGS.z_chunk_size, FLAGS.z_chunk_size)
		
		self.bn_1 = nn.BatchNorm2d(64)
		self.bn_2 = nn.BatchNorm2d(128)
		self.bn_3 = nn.BatchNorm2d(256)
		self.bn_4 = nn.BatchNorm2d(512)
		self.bn_5 = nn.BatchNorm1d(1024)
		self.bn_6 = nn.BatchNorm1d((FLAGS.num_specified_chunks + FLAGS.num_unspecified_chunks)*FLAGS.z_chunk_size)
		
		self.relu = nn.ReLU()
		self.elu = nn.ELU(0.1)
				
	def reparameterize(self, mu, logvar):
		if FLAGS.is_training:
			std = logvar.mul(0.5).exp_()
			eps = torch.randn_like(std)
			if(FLAGS.cuda):
				eps = eps.cuda()
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, x):
		x = self.elu(self.bn_1(self.conv1(x)))
		x = self.elu(self.bn_2(self.conv2(x)))
		x = self.elu(self.bn_3(self.conv3(x)))
		x = self.elu(self.bn_4(self.conv4(x)))
		x = x.view(x.size(0), x.size(1)*x.size(2)*(x.size(3)))
		latent_space = self.elu(self.bn_5(self.features1(x)))
		latent_space = self.elu(self.bn_6(self.features2(latent_space)))  

		l_mean = self.linear_mean(latent_space)
		l_std = self.linear_std(latent_space)  

		specified_latents = [self.linears_z[i](latent_space) for i in range(FLAGS.z_num_chunks)]
		unspecified_variational_latent = self.reparameterize(l_mean, l_std)
		return [specified_latents, unspecified_variational_latent, l_mean, l_std] 

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.features1 = nn.Linear(in_features=(FLAGS.num_specified_chunks + FLAGS.num_unspecified_chunks)*FLAGS.z_chunk_size, out_features=1024, bias=True)
		self.features2 = nn.Linear(in_features=1024, out_features=4608, bias=True)
		
		self.deconv_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, bias=True)
		self.deconv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, bias=True)
		self.deconv_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, bias=True)
		self.deconv_4 = nn.ConvTranspose2d(in_channels=64, out_channels=FLAGS.num_channels, kernel_size=4, stride=2, bias=True)
		
		self.bn_1 = nn.BatchNorm1d(num_features=1024)
		self.bn_2 = nn.BatchNorm1d(num_features=4608)
		self.bn_3 = nn.BatchNorm2d(num_features=256)
		self.bn_4 = nn.BatchNorm2d(num_features=128)
		self.bn_5 = nn.BatchNorm2d(num_features=64)
		self.bn_6 = nn.BatchNorm2d(num_features=FLAGS.num_channels)

		self.tanh = nn.Tanh()
		self.relu =nn.ReLU()
		self.leaky_relu = nn.LeakyReLU(0.2)

	def forward(self, specified_latents, unspecified_variational_latent):
		latent = specified_latents[0]
		for i in range(1, len(specified_latents)):
			latent = torch.cat([latent, specified_latents[i]], dim=1)
		latent = torch.cat([latent, unspecified_variational_latent], dim=1)
		latent = self.bn_1(self.features1(latent))
		latent = self.bn_2(self.features2(latent))
		x = latent.view(latent.size(0), 512, 3, 3)
		x = self.relu(self.bn_3(self.deconv_1(x)))
		x = self.relu(self.bn_4(self.deconv_2(x)))
		x = self.relu(self.bn_5(self.deconv_3(x)))
		x = self.relu(self.bn_6(self.deconv_4(x)))
		return x

class ContextVector(nn.Module):
	def __init__(self):
		super(ContextVector, self).__init__()
		
		self.linear1 = nn.Linear(in_features=FLAGS.batch_size*FLAGS.num_specified_chunks*FLAGS.z_chunk_size, out_features=4096)
		self.linear2 = nn.Linear(in_features=4096, out_features=FLAGS.num_specified_chunks*FLAGS.c_num_chunks*FLAGS.c_chunk_size)

		self.linears_c = nn.ModuleList([nn.Linear(in_features=FLAGS.c_num_chunks*FLAGS.c_chunk_size, out_features=FLAGS.c_chunk_size, bias=True) for i in range(FLAGS.c_num_chunks)])

		self.relu = nn.ReLU()
		
	def forward(self, latent_chunks):
		z_chunks = latent_chunks[0].view(1, FLAGS.z_chunk_size, FLAGS.batch_size)
		for i in range(1, len(latent_chunks)):
			z_chunks = torch.cat([z_chunks, latent_chunks[i].view(1, FLAGS.z_chunk_size, FLAGS.batch_size)])

		z_chunks = z_chunks.view(-1)

		if(FLAGS.cuda):
			context_vectors_full_view = self.linear1(z_chunks).view(-1).cuda()
		else:
			context_vectors_full_view = self.linear1(z_chunks).view(-1)
		
		context_vectors_full_view = self.relu(context_vectors_full_view).view(-1)
		context_vectors_full_view = self.linear2(context_vectors_full_view)
		context_vectors_full_view = self.relu(context_vectors_full_view)

		context_vectors = torch.zeros((FLAGS.num_specified_chunks, FLAGS.c_num_chunks, FLAGS.c_chunk_size))

		for i in range(FLAGS.num_specified_chunks):
			curr_context_vector = context_vectors_full_view.view(FLAGS.num_specified_chunks, -1)[i]
			for j in range(FLAGS.c_num_chunks):
				context_vectors[i, j] = self.linears_c[j](curr_context_vector)

		context_vectors_full_view = context_vectors_full_view.view(FLAGS.z_num_chunks, -1)

		return context_vectors, context_vectors_full_view
		
if __name__ == '__main__':

	x = torch.randn((FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size))
	y = [torch.randn(FLAGS.c_chunk_size) for i in range(FLAGS.c_num_chunks*FLAGS.z_num_chunks)]
	l = [random.randint(0, FLAGS.c_num_chunks-1) for i in range(FLAGS.c_num_chunks*FLAGS.z_num_chunks)]

	if(FLAGS.cuda):
		x = x.cuda()

	encoder = Encoder()
	
	if(FLAGS.cuda):
		encoder.cuda()

	s, u, _, _ = encoder.forward(x)
	print(s[0].shape, u.shape)

	decoder = Decoder()

	if(FLAGS.cuda):
		decoder.cuda()

	x_dash = decoder.forward(s, u)
	print(x_dash.shape)

	context_vector_network = ContextVector()
	
	if(FLAGS.cuda):
		context_vector_network.cuda()

	cv, cv_full_view = context_vector_network.forward(s)
	print(cv.shape, cv_full_view.shape)