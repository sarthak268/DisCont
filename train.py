import os
import argparse
import numpy as np
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F

from utils import weights_init, matrix_log_density_gaussian
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from networks import *
from utils import imshow_grid, mse_loss, reparameterize, transform_config, augment_batch, mix_latents, get_augmentations_and_mask
from flags import FLAGS
from sklearn import manifold

from torchvision.utils import save_image

import random

def compute_center_loss(centers, embeddings, labels, num_classes):

	latent_labels = torch.LongTensor(labels)

	if(FLAGS.cuda):
		latent_labels = latent_labels.cuda()
		embeddings = embeddings.cuda()

	distmat = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(FLAGS.z_num_chunks*FLAGS.batch_size, num_classes) + torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, FLAGS.z_num_chunks*FLAGS.batch_size).t()
	distmat.addmm_(1, -2, embeddings, centers.t())

	classes = torch.arange(num_classes).long()
	
	if(FLAGS.cuda):
		classes = classes.cuda()
		distmat = distmat.cuda()

	labels = latent_labels.unsqueeze(1).expand(FLAGS.batch_size*FLAGS.z_num_chunks, num_classes)
	mask = labels.eq(classes.expand(FLAGS.batch_size*FLAGS.z_num_chunks, num_classes))

	dist = distmat * mask.float()
	loss = dist.clamp(min=1e-12, max=1e+12).sum() / FLAGS.batch_size*FLAGS.z_num_chunks

	return loss

if __name__ == '__main__':

	torch.autograd.set_detect_anomaly(True)
	encoder = Encoder()
	encoder.apply(weights_init)

	decoder = Decoder()
	decoder.apply(weights_init)

	cv_network = ContextVector()
	cv_network.apply(weights_init)

	if FLAGS.load_saved:
		encoder.load_state_dict(torch.load(os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.encoder_save)))
		decoder.load_state_dict(torch.load(os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.decoder_save)))
		cv_network.load_state_dict(torch.load(os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.context_vector_save)))

	combine_optimizer = optim.Adam(
		list(encoder.parameters()) + list(decoder.parameters()) + list(cv_network.parameters()),
		lr = FLAGS.lrate,
		betas = (FLAGS.beta_1, FLAGS.beta_2)
	)

	if torch.cuda.is_available() and not FLAGS.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	if not os.path.exists('checkpoints_'+FLAGS.folder_save):
		os.makedirs('checkpoints_'+FLAGS.folder_save)

	if not os.path.exists('reconstructed_images_'+FLAGS.folder_save):
		os.makedirs('reconstructed_images_'+FLAGS.folder_save)

	if not os.path.exists('predicted_images_'+FLAGS.folder_save):
		os.makedirs('predicted_images_'+FLAGS.folder_save)

	if not os.path.exists('visualizations_'+FLAGS.folder_save):
		os.makedirs('visualizations_'+FLAGS.folder_save)

	if not os.path.exists('context_vectors_'+FLAGS.folder_save):
		os.makedirs('context_vectors_'+FLAGS.folder_save)

	if not FLAGS.load_saved:
		with open(FLAGS.log_file, 'w') as log:
			log.write('Epoch\tIteration\tReconstruction_loss\tKL_divergence_loss\t')
			log.write('Generator_loss\tDiscriminator_loss\tDiscriminator_accuracy\n')
	
	print('Loading Shapes3D Dataset...')
	data_dir = './shapes_data/train/'
	transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
	dset = datasets.ImageFolder(data_dir, transform = transform)
	sprites = torch.utils.data.DataLoader(dset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)

	X1 = torch.zeros(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size)
	X2 = torch.zeros(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size)
	
	cv_full_view = Variable(torch.zeros((FLAGS.z_num_chunks, FLAGS.c_num_chunks*FLAGS.c_chunk_size)))

	if FLAGS.cuda:
		encoder.cuda()
		decoder.cuda()
		cv_network.cuda()
		X1 = X1.cuda()
		X2 = X2.cuda()
		cv_full_view = cv_full_view.cuda()

	for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
		print('Epoch #' + str(epoch) + '..........................................................................')

		for iteration in range(len(sprites)):
			combine_optimizer.zero_grad()
			
			image_batch_1 = next(iter(sprites))[0]
			X1.copy_(image_batch_1)
			
			# augmented_batch = augment_batch(X1)
			augmented_batch, mask = get_augmentations_and_mask(X1)

			encoder_outputs = encoder(X1)
			specified_latents, unspecified_variational_latent, mu, logvar = encoder_outputs[0], encoder_outputs[1], encoder_outputs[2], encoder_outputs[3]

			augmented_encoder_outputs = encoder(augmented_batch)
			aug_specified_latents, aug_unspecified_variational_latent, aug_mu, aug_logvar = augmented_encoder_outputs[0], augmented_encoder_outputs[1], augmented_encoder_outputs[2], augmented_encoder_outputs[3]
			
			# kl loss
			kl_loss = FLAGS.kl_divergence_coef * (-0.5 * (torch.sum(1 + logvar - mu.pow(2) - logvar.exp())))
			kl_loss /= FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size
			
			# reconstruction loss batch
			image_batch_recon = decoder(specified_latents, unspecified_variational_latent)
			recon_loss = mse_loss(image_batch_recon, X1)

			gen_loss = recon_loss + kl_loss

			# center loss
			cv, cv_full_view = cv_network(specified_latents)
			transformed_chunks = torch.zeros(FLAGS.batch_size*FLAGS.z_num_chunks, FLAGS.c_num_chunks*FLAGS.c_chunk_size)

			with torch.no_grad():

				for i in range(FLAGS.batch_size):
					
					transformed_temp_chunks = []

					for j in range(FLAGS.z_num_chunks):
						curr_tensor = specified_latents[j][i]
						curr_tensor = curr_tensor.repeat(FLAGS.batch_size).view(FLAGS.batch_size, FLAGS.z_chunk_size)
						transformed_temp_chunks.append(curr_tensor)
				
					curr_cv, curr_cv_full_view = cv_network(transformed_temp_chunks)
					transformed_chunks[i*FLAGS.z_num_chunks : (i+1)*FLAGS.z_num_chunks] = curr_cv_full_view

			lab_list = [i for i in range(FLAGS.z_num_chunks)]
			transformed_chunks_labels = lab_list*FLAGS.batch_size

			center_loss = compute_center_loss(cv_full_view, transformed_chunks, transformed_chunks_labels, num_classes=FLAGS.z_num_chunks)
			
			# masked augmentation loss
			aug_loss = 0
			for i, val in enumerate(mask[:-1]):
				val = 1-val
				aug_loss+=val*mse_loss(aug_specified_latents[i], specified_latents[i])
			aug_loss+=mse_loss(aug_unspecified_variational_latent, unspecified_variational_latent)

			# total losses, backprop and optimization
			loss = gen_loss + center_loss + aug_loss

			cv_full_view.retain_grad() # retain gradients of the CV for CV update
			center_loss.backward(retain_graph=True)
			cv_gradient = cv_full_view.grad.clone() # ensure gradient is not None, save it before we detach from computation graph
			cv_full_view = cv_full_view.detach() # detach CV var from computation graph so as to not interfere with other gradients
			cv_full_view = cv_full_view - FLAGS.center_loss_lrate*cv_gradient # CV update

			loss.backward()
			combine_optimizer.step()
			
			if (iteration + 1) % 50 == 0:
				print('')
				print('----------------------------------------------------------------------')
				print('Epoch #' + str(epoch))
				print('Iteration #' + str(iteration))

				print('')
				print('Reconstruction loss: ' + str(recon_loss.data.storage().tolist()[0]))
				print('KL-Divergence loss: ' + str(kl_loss.data.storage().tolist()[0]))
				print('Center loss: '+str(center_loss.item()))
				print('Mask Augmentation loss: '+str(aug_loss.item()))
				print()
				print('Generator loss: ' + str(gen_loss.data.storage().tolist()[0]))
				print('----------------------------------------------------------------------')

		# save model after every 5 epochs
		if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
			torch.save(encoder.state_dict(), os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.encoder_save))
			torch.save(decoder.state_dict(), os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.decoder_save))
			torch.save(cv_network.state_dict(), os.path.join('checkpoints_'+FLAGS.folder_save, FLAGS.context_vector_save))


		image_batch_1 = next(iter(sprites))[0]
		X1.copy_(image_batch_1)
		
		latent = encoder(X1)
		dec = decoder(latent[0], latent[1])

		save_image(torch.cat([X1[:8, :, :, :], dec[:8, :, :, :]]), './reconstructed_images_'+FLAGS.folder_save+'/'+str(epoch)+'.png', nrow=4)
