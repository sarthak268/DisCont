import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Compose, ToTensor, ToPILImage, RandomVerticalFlip, RandomRotation, RandomHorizontalFlip, ToTensor, Grayscale, RandomResizedCrop
from flags import FLAGS
import math
import random
from torch.nn import functional as F
import numbers
import numpy as np

# compose a transform configuration
transform_config = Compose([ToTensor()])

class GaussianSmoothing(nn.Module):
	"""
	Apply gaussian smoothing on a
	1d, 2d or 3d tensor. Filtering is performed seperately for each channel
	in the input using a depthwise convolution.
	Arguments:
		channels (int, sequence): Number of channels of the input tensors. Output will
			have this number of channels as well.
		kernel_size (int, sequence): Size of the gaussian kernel.
		sigma (float, sequence): Standard deviation of the gaussian kernel.
		dim (int, optional): The number of dimensions of the data.
			Default value is 2 (spatial).
	"""
	def __init__(self, channels, kernel_size, sigma, dim=2):
		super(GaussianSmoothing, self).__init__()
		if isinstance(kernel_size, numbers.Number):
			kernel_size = [kernel_size] * dim
		if isinstance(sigma, numbers.Number):
			sigma = [sigma] * dim

		# The gaussian kernel is the product of the
		# gaussian function of each dimension.
		kernel = 1
		meshgrids = torch.meshgrid(
			[
				torch.arange(size, dtype=torch.float32)
				for size in kernel_size
			]
		)
		for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
			mean = (size - 1) / 2
			kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
					  torch.exp(-((mgrid - mean) / std) ** 2 / 2)

		# Make sure sum of values in gaussian kernel equals 1.
		kernel = kernel / torch.sum(kernel)

		# Reshape to depthwise convolutional weight
		kernel = kernel.view(1, 1, *kernel.size())
		kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

		self.register_buffer('weight', kernel)
		self.groups = channels

		if dim == 1:
			self.conv = F.conv1d
		elif dim == 2:
			self.conv = F.conv2d
		elif dim == 3:
			self.conv = F.conv3d
		else:
			raise RuntimeError(
				'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
			)

	def forward(self, input):
		"""
		Apply gaussian filter to input.
		Arguments:
			input (torch.Tensor): Input to apply gaussian filter on.
		Returns:
			filtered (torch.Tensor): Filtered output.
		"""
		return self.conv(input, weight=self.weight, groups=self.groups)

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(torch.Tensor(img))
        img = img * mask

        return img


def mse_loss(input1, target):
	return torch.sum((input1 - target).pow(2)) / input1.data.nelement()


def reparameterize(training, mu, logvar):
	if training:
		std = logvar.mul(0.5).exp_()
		eps = Variable(std.data.new(std.size()).normal_())
		return eps.mul(std).add_(mu)
	else:
		return mu


def weights_init(layer):
	if isinstance(layer, nn.Conv2d):
		layer.weight.data.normal_(0.0, 0.05)
		layer.bias.data.zero_()
	elif isinstance(layer, nn.BatchNorm2d):
		layer.weight.data.normal_(1.0, 0.02)
		layer.bias.data.zero_()
	elif isinstance(layer, nn.Linear):
		layer.weight.data.normal_(0.0, 0.05)
		layer.bias.data.zero_()


def imshow_grid(images, shape=[2, 8], name='default', save=False):
	"""Plot images in a grid of a given shape."""
	fig = plt.figure(1)
	grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

	size = shape[0] * shape[1]
	for i in range(size):
		grid[i].axis('off')
		grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

	if save:
		plt.savefig('reconstructed_images/' + str(name) + '.png')
		plt.clf()
	else:
		plt.show()

# reference: https://github.com/YannDubs/disentangling-vae/blob/master/disvae/utils/math.py
 
def log_density_gaussian(x, mu, logvar):
	"""Calculates log density of a Gaussian.
	Parameters
	----------
	x: torch.Tensor or np.ndarray or float
		Value at which to compute the density.
	mu: torch.Tensor or np.ndarray or float
		Mean.
	logvar: torch.Tensor or np.ndarray or float
		Log variance.
	"""
	normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
	inv_var = torch.exp(-logvar)
	log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
	return log_density

def matrix_log_density_gaussian(z, mu, logvar):

	z_cat = torch.cat([z[i].unsqueeze(1) for i in range(len(z))], dim=1).unsqueeze(1) # batch, 1, 3, 8
	mu_cat = torch.cat([mu[i].unsqueeze(1) for i in range(len(mu))], dim=1).unsqueeze(0) # 1, batch, 3, 8
	logvar_cat = torch.cat([logvar[i].unsqueeze(1) for i in range(len(logvar))], dim=1).unsqueeze(0) # 1, batch, 3, 8

	batch_size, dim = z[0].shape[0], FLAGS.num_chunks

	normalization = - 0.5 * (math.log(2 * math.pi) + logvar_cat) # 1, batch, 3 8
	inv_var = torch.exp(-logvar_cat) # 1, batch, 3, 8
	log_density = normalization - 0.5 * ((z_cat - mu_cat)**2 * inv_var)
	
	return log_density

# Check for one dimension (chunk_size=1) only

def matrix_log_density_gaussian_(x, mu, logvar):
	"""Calculates log density of a Gaussian for all combination of bacth pairs of
	`x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
	instead of (batch_size, dim) in the usual log density.
	Parameters
	----------
	x: torch.Tensor
		Value at which to compute the density. Shape: (batch_size, dim).
	mu: torch.Tensor
		Mean. Shape: (batch_size, dim).
	logvar: torch.Tensor
		Log variance. Shape: (batch_size, dim).
	batch_size: int
		number of training images in the batch
	"""
	batch_size, dim = x.shape
	x = x.view(batch_size, 1, dim)
	mu = mu.view(1, batch_size, dim)
	logvar = logvar.view(1, batch_size, dim)
	return log_density_gaussian_(x, mu, logvar)


def log_density_gaussian_(x, mu, logvar):
	"""Calculates log density of a Gaussian.
	Parameters
	----------
	x: torch.Tensor or np.ndarray or float
		Value at which to compute the density.
	mu: torch.Tensor or np.ndarray or float
		Mean.
	logvar: torch.Tensor or np.ndarray or float
		Log variance.
	"""
	normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
	inv_var = torch.exp(-logvar)
	log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
	return log_density

def sample_augmentation():
	augs = ['rot', 'flip', 'gauss_n', 'gauss_s']
	return random.choice(augs)

def rotate_aug(image_batch):
	angle = random.choice([90, 180, 270])
	aug = torch.zeros_like(image_batch)

	for i in range(image_batch.shape[0]):
		aug[i] = ToTensor()(RandomRotation(degrees=angle)(ToPILImage(mode='RGB')(image_batch[i].cpu())))

	if(FLAGS.cuda):
		aug = aug.cuda()
	return aug

def flip_aug(image_batch):
	flip_dir = random.choice([0, 1])
	aug = torch.zeros_like(image_batch)

	if(flip_dir==0):
		for i in range(image_batch.shape[0]):
			aug[i] = ToTensor()(RandomHorizontalFlip(p=1)(ToPILImage(mode='RGB')(image_batch[i].cpu())))
	else:
		for i in range(image_batch.shape[0]):
			aug[i] = ToTensor()(RandomVerticalFlip(p=1)(ToPILImage(mode='RGB')(image_batch[i].cpu())))

	if(FLAGS.cuda):
		aug = aug.cuda()
	return aug

def color_aug(image_batch):
	aug = torch.zeros_like(image_batch)

	for i in range(image_batch.shape[0]):
		aug[i] = ToTensor()(Grayscale()(ToPILImage(mode='RGB')(image_batch[i].cpu())))
	
	if(FLAGS.cuda):
		aug = aug.cuda()
	return aug

def cutout_aug(image_batch):
	aug = torch.zeros_like(image_batch)

	n_holes = [5, 10, 15, 20]

	for i in range(image_batch.shape[0]):
		# aug[i] = ToTensor()(Cutout(n_holes=1, length=random.choice(n_holes))(ToPILImage(mode='RGB')(image_batch[i].cpu())))
		aug[i] = Cutout(n_holes=1, length=random.choice(n_holes))(image_batch[i].cpu())

	if(FLAGS.cuda):
		aug = aug.cuda()
	return aug

def crop_and_resize_aug(image_batch):
	aug = torch.zeros_like(image_batch)

	for i in range(image_batch.shape[0]):
		aug[i] = ToTensor()(RandomResizedCrop(size=FLAGS.image_size)(ToPILImage(mode='RGB')(image_batch[i].cpu())))
	
	if(FLAGS.cuda):
		aug = aug.cuda()
	return aug	

def gauss_smoothing(image_batch):
	smoothing = GaussianSmoothing(3, 5, 1)
	if(FLAGS.cuda):
		smoothing.cuda()
	img = F.pad(image_batch, (2, 2, 2, 2), mode='reflect')
	return smoothing(img)

def gauss_noise(image_batch):
	std = random.choice([0.5, 1, 2, 5])
	noise = torch.randn(image_batch.size()) * std
	if(FLAGS.cuda):
		noise = noise.cuda()
	return image_batch + noise

def augment_batch(image_batch):
	aug = sample_augmentation()

	if(aug=='rot'):
		augmented_img = rotate_aug(image_batch)
		return augmented_img
	elif(aug=='flip'):
		augmented_img = flip_aug(image_batch)
		return augmented_img
	elif(aug=='gauss_s'):
		augmented_img = gauss_smoothing(image_batch)
		return augmented_img
	elif(aug=='gauss_n'):
		augmented_img = gauss_noise(image_batch)
		return augmented_img

def get_augmentations_and_mask(image_batch):
	neg_augs = ['rot_or_flip', 'color', 'cutout', 'crop_and_resize']
	neutral_augs = ['gauss_n', 'gauss_s']
	
	mask = [0]*(len(neg_augs)+1)  #always consider the non neg augs to be 0

	final_batch = image_batch

	for i, aug in enumerate(neg_augs):
		if(random.choice([0, 1])==1):
			mask[i] = 1

			if(aug=='rot_or_flip'):
				if(random.choice([0, 1])==1):
					final_batch = rotate_aug(final_batch)
				else:
					final_batch = flip_aug(final_batch)

			if(aug=='color'):
				final_batch = color_aug(final_batch)

			if(aug=='cutout'):
				final_batch = cutout_aug(final_batch)

			if(aug=='crop_and_resize'):
				final_batch = crop_and_resize_aug(final_batch)

	for i, aug in enumerate(neutral_augs):
		if(random.choice([0, 1])==1):
			if(aug=='gauss_s'):
				final_batch = gauss_smoothing(final_batch)
			if(aug=='gauss_n'):
				final_batch = gauss_noise(final_batch)

	return final_batch, mask

def mix_latents(s, aug_s, v, aug_v):
	if(random.choice([0, 1])==0):
		ret_v = v
	else:
		ret_v = aug_v

	ret_s = []

	for i in range(len(s)):
		if(random.choice([0, 1])==0):
			ret_s.append(s[i])
		else:
			ret_s.append(aug_s[i])

	return ret_s, ret_v


if __name__ == '__main__':
	print(sample_augmentation())

	rand = torch.randn((5, 3, 64, 64))
	print(gauss_smoothing(rand).shape)
	print(gauss_noise(rand).shape)
	print(rotate_aug(rand).shape)
	print(flip_aug(rand).shape)
	print(color_aug(rand).shape)
	print(cutout_aug(rand).shape)
	print(crop_and_resize_aug(rand).shape)
