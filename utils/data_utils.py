import os
import math
from torch import nn, max
import logging
import urllib.request as request
import torchvision.transforms.functional as TTF
from torchvision.transforms import Lambda
from torch import FloatTensor, zeros, load
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import scipy.io as sio
from numbers import Number
import numpy as np
from itertools import chain, product
from collections.abc import Iterable
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, ConcatDataset
from torch.utils.data import TensorDataset, ConcatDataset, Dataset
from utils import dist 
import time 
import torch
import glob
import torch
from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
from data.MovingMNIST import MovingMNIST
from torchvision.utils import save_image
from skimage.transform import resize
from pathlib import Path


def generate_env_data(p):

	vals_per_action = 3

	env_name = 'env/AnimalAI'
	env = UnityEnvironment( n_arenas=p['n_arenas'],
							file_name=env_name)

	files = glob.glob("examples/configs/env_configs/*.yaml") # list of all .yaml files in a directory 


	for file in files:
		
		img_tensor = torch.zeros(p['n_arenas'], p['n_trials'], p['n_steps'], 3, 84, 84)
		act_tensor = torch.zeros(p['n_arenas'], p['n_trials'], p['n_steps'], 2, 1)

		config = ArenaConfig(file)

		iteration = 0
		for t_o in range(p['n_trials']):

			obs = env.reset(arenas_configurations=config, train_mode=True)['Learner']
			action = {}

			# First action in new environment - don't do anything 
			rand_action = torch.randint(0,3,(p['n_actions']*p['n_arenas'],1))
			rand_action = rand_action.cuda() if p['gpu'] else rand_action
			action['Learner'] = rand_action
			
			for t_i in range(p['n_steps']):
					
				# run batch
				info 	 =  env.step(vector_action=action)['Learner']
				vis_obs  = torch.FloatTensor(info.visual_observations[0]).cuda().permute(0,-1,1,2)
				vel_obs  = torch.tensor(info.vector_observations).cuda()
				text_obs = info.text_observations
				reward 	 = info.rewards				
				
				img_tensor[:, t_o, t_i] = vis_obs
				act_tensor[:, t_o, t_i] = rand_action.view(p['n_arenas'], p['n_actions'], 1)

			
		torch.save(img_tensor, 'imgs'+file.split('.')[0].split('\\')[1]+'.pt') 
		torch.save(act_tensor, 'acts'+file.split('.')[0].split('\\')[1]+'.pt') 
	env.close()

def roll(x, shift, dim, fill_pad=None):
	
    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift).cuda())
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim))), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim)).cuda())
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([gap, x.index_select(dim, torch.arange(shift).cuda())], dim=dim)
		
def train_val_split(p, dataset, vs=None):

		shuffle_dataset = True 
		random_seed = 5
		if not vs:
			validation_split = 0.1
		else:
			validation_split = vs
			
		if hasattr(dataset, 'data'):
			dataset_size = len(dataset.data)
		else: # is class
			dataset_size = dataset.__len__()
		
		# added for celeba, may preclude up to 999 samples from other sets
		# dataset_size = int(np.floor(dataset_size / 1000.0)*1000.0)
		if not p['dataset']=='cardiff':
			dataset_size = int(np.floor(dataset_size / 100.0)*100.0)
		
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		if shuffle_dataset :
			#np.random.seed(random_seed)
			np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
		# Creating PT data samplers and loaders:
		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)

		train_loader = torch.utils.data.DataLoader(dataset, batch_size=p['b'], #pin_memory=True,
												   sampler=train_sampler, num_workers=0)
		validation_loader = torch.utils.data.DataLoader(dataset, batch_size=p['b'], 
														sampler=valid_sampler, num_workers=0)

		return train_loader, validation_loader


def get_dataset(p, split='train', transform=None, static=True, exp=None,
				target_transform=None, download=True, path='data', from_matlab=False):
	
	# exp = integer corresponding to animalai environment configuration

	class MNISTransform(object):
		def __init__(self, b,t,dim):
			self.b = b
			self.t = t
			self.dim = dim
		def __call__(self, image):
			
			image = TTF.to_tensor(image) # (batch, c, h, w)
			new_im = torch.zeros(image.shape[0],32,32)
			new_im[:,2:30,2:30] = image
			if self.t > 1:
				new_im = new_im.unsqueeze(1)	 # (batch, 1, c, h, w)
				new_im = new_im.expand(self.t,*self.dim) # (batch, t, c, h, w)
			return new_im

	
	name = p['dataset']
	batch_size = p['b']
	train = (split == 'train')
	root = os.path.join(os.path.realpath(path), name)
	if name == 'animalai':

		# p['n_arenas'] = 20
		# p['n_trials'] = 5
		# p['n_steps']  = 20
		# p['n_actions'] = 2
		# model_inner = 4 

		exps = glob.glob('data/animalai/imgs*.npz') # list of all .yaml files in a directory 
		exps = [x.split('imgs')[1] for x in exps]

		im_data  = [] ;  act_data = []

		_imshape  = (-1, 3, 84, 84) #if static else (-1, p['n_steps'],3, 84, 84)
		_actshape = (-1, 1, 2) # if static else (-1, p['n_steps'], 2, 3)
		tensordatasets = []
				
		for x in exps:
			# food preferences avoidance
			#if x.startswith(('1', '2', '4')):
			if x.startswith(('allObjectsRandom')):
				im_files  = glob.glob('data/animalai/imgs{}'.format(x)) 
				act_files = glob.glob('data/animalai/acts{}'.format(x)) 
				#im_data = torch.load(im_file[0]).view(*_imshape).cpu()
								
				for im_file in im_files:

					im_data = np.load(im_file)
					im_data = np.array(im_data['arr_0'])
					im_data = torch.from_numpy(im_data).to(torch.float)
					
					im_data = im_data.squeeze(1)#.view(*_imshape) 
					im_data = im_data.permute(0,3,1,2)
					im_data = im_data.view(-1, 3, 84, 84)
					tensordatasets.append(TensorDataset(im_data))
					break
				else:
					continue
				
		## sort validation data
		#im_tensors  = torch.cat(im_data,  dim=0)
		#act_tensors = torch.cat(act_data, dim=0)
        #
		##if static:
		#	# repeat single image n_model (meta) times
		##	im_tensors = im_tensors.unsqueeze(1).expand(-1,p['model_inner'],*_imshape[1:])
        #
		#dataset = (*[im_tensors, act_tensors])
		
		dataset = ConcatDataset(tensordatasets)
		
		return train_val_split(p, dataset)
			
	elif name == 'stl10':
		
		data =	datasets.STL10(root=root,
							  split=split,
							  transform=transforms.Compose([transforms.ToTensor()]), 
							  download=True)
		
		train =  DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
		test  =  DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
			

		return train, test
				  
	elif name in ['mnist','RFMnist', 'mnist_spatial_z']:

		root = os.path.join('data','mnist', 'processed', 'MNIST','raw')

		if static:
			transform = MNISTransform(p['b'],1,p['imdim'])
		else:
			transform = MNISTransform(p['b'],p['n_steps'],p['imdim'])

		if 'rotating' in p.keys():
			if p['rotating']:
				data = rotate_mnist(p['t'], data)

		data = datasets.MNIST(root=root, 
							 train=True, download=True,
							 transform=transform)
							 
		if from_matlab:
			return data.data[:batch_size].numpy()
		
		return train_val_split(p, data)
		
	elif name == 'moving_mnist':
		
		x = np.load("./data/movingmnistdata.npz", encoding="bytes")
		

		dset = np.load("./data/movingmnistdata.npz", encoding="bytes")["arr_0"]
		
		if from_matlab:
			#return matlab.double(dset[:p['b']])
			return dset[:batch_size]
		imgs = torch.from_numpy(dset).float() / 255.
		imgs = imgs.view(-1,10, 1, 32, 32)
		
		
		data = TensorDataset(imgs, torch.zeros(imgs.shape[0]))

		if from_matlab:
			#return matlab.double(data.data[:p['b']])
			return data.data[:batch_size].numpy()
		
		
		return train_val_split(p, data)

	elif name == 'mnist_sequences':
		root = os.path.join(root, 'processed')
		data = datasets.MNIST(root=root, train=True, download=True,
		transform= transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]))
		
		images = data.data
		labels = data.targets
		new_ims = []
		
		# lossy method
		all_seqs = [[0,1,2,3], [3,4,5,6], [6,7,8,9]]

		# hacky way to ensure image-label alignment
		lim = min([min([images[labels==i].size(0) for i in s])] for s in all_seqs)[0]
		new_ims     = [torch.stack([images[labels == i][:lim,:,:] for i in s], dim=1) for s in all_seqs]	
		new_targets = [torch.stack([labels[labels == i][:lim] for i in s], dim=1) for s in all_seqs]
		new_ims = torch.cat(new_ims, dim=0) ; new_targets = torch.cat(new_targets, dim=0)
		
		# prevent batch underflow for divsors of 100
		def roundown(x): # could replace 100 with p['b']
			return int(math.floor(x / 100.0)) * 100
		data.data = np.expand_dims(new_ims[:roundown(new_ims.size(0))], axis=2)
		#data.data = new_ims[:roundown(new_ims.size(0))]
		data.targets = new_targets[:roundown(new_ims.size(0))].numpy()

		return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
	
	elif name == 'lsun_bedroom':
		data = datasets.LSUN(root=root, 
							 train=True, 
							classes=['bedroom_train'],
							transform=transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5),
													 (0.5, 0.5, 0.5)),
							]), download=True)
							 
	elif name == 'dsprites':

		url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
		
		if not os.path.exists('data/dsprites'):
			os.makedirs('data/dsprites')

		try:
			dset = np.load("data/dsprites/dsprites.npz")
		except:
			request.urlretrieve(url, "data/dsprites/dsprites.npz")
			dset = np.load("data/dsprites/dsprites.npz", encoding="bytes")	
		
		if from_matlab:
			return dset[:batch_size]

		imgs = torch.from_numpy(dset['imgs']).unsqueeze(1).float()
		data = TensorDataset(imgs)
		if from_matlab:
			return data.data[:batch_size].numpy()
		
		return train_val_split(p, data)
	
	elif name == 'celeba':
		
		data = datasets.ImageFolder('./data/celeba/aligned', transform=transforms.ToTensor())
		return train_val_split(p, data)
	
	elif name == 'fec':
		
		fec_path = os.path.join(os.getcwd(),'data','faces')
		data = datasets.ImageFolder(fec_path, transform=transforms.ToTensor())
		return train_val_split(p, data)	
		
	elif name == 'emnist':
		root = os.path.join(os.getcwd(),'data','emnist')
		data = datasets.EMNIST(root=root, 
							 train=True, 
							 split='letters',
							 transform=transforms.Compose([Lambda(lambda img: TTF.rotate(img, -90)),
												 Lambda(lambda img: TTF.hflip(img)),
												 MNISTransform(p['b'],1,p['imdim'])]), 
							 download=True)		
		return train_val_split(p, data)
	
	elif name == '3dfaces':
		
		loc = 'data/3dfaces/basel_face_renders.pth'
		data = torch.load(loc).float().div(255).view(-1, 1, 64, 64)
		return train_val_split(p, data)

	elif name == 'cardiff':
		class CardiffDataset(Dataset):
			def __init__(self, paths, dynamic, chans):
				self.paths = paths
				self.n_samples = len(paths)
				self.dynamic = dynamic
				self.chans = chans

			def __getitem__(self, index):
				tmp = load(self.paths[index], lambda storage, loc: storage)
				if not self.dynamic:
					tmp = tmp.view(-1, 6, 256, 256)
					tmp = tmp[:, -self.chans:,:, :]
				else:
					tmp = tmp[:, :, -self.chans:, :, :]

				return tmp[:20].float()			
			def __len__(self):
				return self.n_samples


		pt_files = list(Path('E:/animalai/animal_ai/data/cardiff_data').rglob('*.pt'))
		train_data = CardiffDataset(pt_files, p['dynamic'], p['chans']) 
		train_loader = torch.utils.data.DataLoader(train_data, batch_size=p['b'], num_workers=0)
		
		test_data = CardiffDataset(pt_files, p['dynamic'], p['chans']) 
		validation_loader = torch.utils.data.DataLoader(test_data, batch_size=p['b'],num_workers=0)

		return train_loader, validation_loader

		#links = [CardiffDataset(x, p['dynamic'], p['chans']) for x in pt_files]
		#data = ConcatDataset(links)	
		#if p['dynamic']:
		#	data = zeros(p['b']*p['b'], 5, 6, 256, 256)
		#	data = data[:,:,-p['chans']:,:,:]
		#else:
		#	data = zeros(p['b']*p['b'], 6, 256, 256)
		#	data = data[:,-p['chans']:,:,:]
		
	
	elif name == 'car_racing':
		
		class NumpyDataset(Dataset):

			def __init__(self, root_path, transforms):
				self.data_numpy_list = [x for x in glob.glob(os.path.join(root_path, '*.npz'))]
				self.transforms = transforms
				self.terminal = []
				self.actions   = []
				self.observations = []

				for ind in range(len(self.data_numpy_list)):
				#for ind in range(200):
					data_slice_file_name = self.data_numpy_list[ind]
					data_i = np.load(data_slice_file_name, mmap_mode='r')
					#self.terminals.append(data_i['terminals'])
					self.actions.append(torch.from_numpy(np.asarray(np.swapaxes(data_i['actions'], -1, 1))[:-1]).float())
					self.observations.append(torch.from_numpy(np.asarray(data_i['observations'])).float())
					
				self.observations = torch.cat(self.observations, dim=0).view(-1,10,3,64,64)
				self.actions = torch.cat(self.actions, dim=0).view(-1,10,3)
				self.data_len = self.observations.shape[0]

			def __getitem__(self, index):

				observations = self.observations[index]
				actions = self.actions[index]
				#self.data = np.stack((self.data, self.data, self.data)) # gray to rgb 64x64 to 3x64x64
				#if self.transforms:
				#	print(observations.shape)
				#	observations = self.transforms(observations)

				return observations, actions

			def __len__(self):
				return self.data_len
		
		data = NumpyDataset(os.path.join(os.getcwd(),'data','car_racing'), transforms.ToTensor())
		return train_val_split(p, data)

	else:
		print('No such dataset : {}'.format(name))
		0/0


def rotate_mnist(t, data):
	""" fully rotate an mnist character in nt """
	bg_value = -0.5 
	new_imgs = []
	thresh = Variable(torch.Tensor([0.1])) # threshold
	image = np.reshape(data, (-1, 28))
	for i in range(t):
		angle = -(360 // 12) * i
		new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
		new_img = torch.tensor(new_img).view((-1, 1, 28,28))
		new_img = (new_img < thresh).float() * 1
		new_img = new_img.add(-1).abs()
		new_imgs.append(new_img)
	return torch.stack(new_imgs, dim=1)
	

def data_check(p, data):
	
	if isinstance(data, list):
		data = [ Variable( x.cuda() if p['gpu'] else x ) for x in data]
	else:
		data = Variable(data.cuda() if p['gpu'] else data)

	assert not max(data[0]) > 1.
	return data
