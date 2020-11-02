import os 
import math 
import yaml 
import numpy as np
from utils import model_utils as mutils 
from utils import train_utils as tutils 

from modules import * 
from os.path import join
from pprint import pprint, pformat
from scipy.io import savemat
from logging import getLogger, shutdown
from torch.nn import MSELoss, Module, CrossEntropyLoss, L1Loss
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
from torch import nn, optim, zeros, save, Tensor, FloatTensor, cuda, no_grad, isnan, load, min, max
from torch.autograd import Variable, set_detect_anomaly, detect_anomaly
from argparse import ArgumentParser
import torch
import glob 
import scipy.io as sio
from animalai.envs.brain import BrainParameters
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class VRNN(Module):
	def __init__(self, p):
		super(VRNN,self).__init__()
		
		self.p = p
		
		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		self.methandle = tutils.MetricsHandler() 		
		
		self.mse  = MSELoss(reduction='sum').cuda() if p['gpu'] else MSELoss()
		self.iter_loss = 0.0		
		self.has_con = self.p['nz_con'][0] > 0 
		self.has_dis = self.p['nz_dis'][0][0] > 0 	
		
		# Initialise Distributions
		self.prior_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(p)
		
		p['z_params']	= self.q_dist.nparams 

		# Initialise Modules - learnable		
		if p['dataset'] in ['CarRacing', 'car_racing']:
			self.encoder = CarRacingEncoder(p,0)
			self.decoder = CarRacingDecoder(p,0)
		
		else:
			print('No such dataset')
			0/0

		if p['gpu']:
			self.cuda()
		
	def plot(self, i, input_image, plot_vars, vis):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
		matsdir = os.path.join(self.p['plot_dir'], self.p['model_name'], 'mats')
		
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		os.makedirs(matsdir) if not os.path.exists(matsdir) else None
	
		save_image(pred[0].data.cpu(), pdir+'/p{}.png'.format(i))
		save_image(input_image[0,-1,:3].data.cpu(), pdir+'/b{}.png'.format(i))
		
			
		x_data = self.methandle.metrics['iter']
		y1 	   = self.methandle.metrics['recon']
		y2 	   = self.methandle.metrics['cont_kl']
		y3 	   = self.methandle.metrics['disc_kl']
		
		
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(x_data,y1, label='recon')
		ax.plot(x_data,y2, label='cont_kl')
		ax.plot(x_data,y3, label='disc_kl')
		fig.savefig(os.path.join(pdir,'m{}.png'.format(i)))
		plt.close(fig)
		
		vis(i)
	
	def reset(self):
		print('Reset Not Implemented')
		self.iter_loss = 0.
		pass

	def loss(self, curiter, image, pred, z_pc, z_pd, eval=False):

		loss 			   = 0.		
		train_loss 		   = [] 
		train_norm_kl_loss = []
		train_cat_kl_loss  = [] 
		layer_loss 		   = 0.

		#err_loss = self.mse(image, pred)	
		flat_dim = np.prod(self.p['imdim'])
		
		err_loss = F.binary_cross_entropy(pred.view(-1,flat_dim), image.view(-1,flat_dim))
		err_loss *= flat_dim

		if self.has_con:
			kloss_args	= (z_pc,   # mu, sig
						   self.p['z_con_capacity'][0], # anealing params
						   curiter)	# data size
						   
			norm_kl_loss = self.q_dist.calc_kloss(*kloss_args) #/ self.p['b']
		else:
			norm_kl_loss = torch.tensor(0)
		
		if self.has_dis:
			kloss_args	 = (z_pd,  # alpha
							self.p['z_dis_capacity'][0],  # anneling params 
							self.p['nz_dis'][0], # nclasses per categorical dimension
							curiter)	# data size
						  
			cat_kl_loss = self.cat_dist.calc_kloss(*kloss_args) #/ self.p['b']
		else:
			cat_kl_loss = torch.tensor(0)
		
		if self.p['elbo_loss']:	
			layer_loss = norm_kl_loss + cat_kl_loss + err_loss
		else:
			layer_loss = err_loss 
		
		#loss = layer_loss / np.prod(self.p['imdim'][1:])
		loss = layer_loss / flat_dim
		#loss += layer_loss 

		#if self.p['dataset'] == 'mnist':
		#	loss /= np.prod(self.p['imdim'][1:])
		
		metrics = {'iter':curiter, 
				   'recon':err_loss.item(), 
				   'cont_kl':norm_kl_loss.item(), 
				   'disc_kl':cat_kl_loss.item()}

		return loss, metrics
	
	def decode(self, z, action=None):
		# maybe try with action as y axis
		if action==None:
			action = torch.zeros((z.shape[0], self.p['n_actions']))
			if self.p['gpu']:
				action = action.cuda()
		predictions = torch.zeros(z.shape[0], z.shape[1], 3, 64, 64)
		for b in rangee(z.shape[0]):
			for t in range(z.shape[1]):
				if t == 0:
					hs = None
				xhat, hs = self.decoder(z[b,t], action, hs)
				predictions[b,t] = xhat.detach().cpu()
			
		return torch.stack(predictions, dim=1)
	
	def forward(self, iter, imseq, actions, eval=False, to_matlab=False):
			
		if to_matlab:
			
			self.p['gpu'] = False
			self.reset()
			if not isinstance(image, Tensor) or isinstance(image, FloatTensor):
				image = FloatTensor(np.asarray(image)).unsqueeze(0).unsqueeze(0)
				#from torchvision.utils import save_image
				#save_image(image, 'lol.png')

			if not actions is None:
				actions = FloatTensor(actions).unsqueeze(0).unsqueeze(0)

		init_decoder_states	 = []
		# -----------------------------------------
		# Initialise Q/P/D States w/ context frames
		# -----------------------------------------

		# Forward 
		for t in range(imseq.shape[1]-1):

			# Frame Encoder / ResNet 16

			# Encoding - p(z2|x) or p(z1 |x,z2)
			z_pc, z_pd	   = self.encoder(imseq[:,t])
			
			# Latent Sampling
			latent_sample = []

			# Continuous sampling 
			if self.has_con:
				norm_sample = self.q_dist.sample_normal(params=z_pc, train=self.training)
				latent_sample.append(norm_sample)

			# Discrete sampling
			if self.has_dis:
				for ind, alpha in enumerate(z_pd):
					cat_sample = self.cat_dist.sample_gumbel_softmax(alpha, train=self.training)
					latent_sample.append(cat_sample)
			
			z = torch.cat(latent_sample, dim=-1)

			# Decoding - p(x|z)
			
			pred ,hs = self.decoder(z, actions[:,t], hs)
		
			if self.training:

				iter_loss, metrics = self.loss(iter, imseq[:,t+1], pred, z_pc, z_pd) 
				if iter % 10000:
					self.methandle.extend(metrics)

				self.iter_loss += iter_loss

		if to_matlab:
			return z.detach().numpy(), foveated.detach().numpy()
			
		elif eval:
			return z, pred

class HVRNN(Module):
	def __init__(self, p):
		super(HVRNN,self).__init__()
		
		self.p = p
		
		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		self.methandle = tutils.MetricsHandler() 		
		
		self.mse  = MSELoss(reduction='sum').cuda() if p['gpu'] else MSELoss()
		self.iter_loss = 0.0

		
		# Initialise Distributions
		self.p_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(p)
		
		p['z_params']	= self.q_dist[0].nparams 

		# Initialise Modules - learnable
		self.d_init = mutils.ListModule(*[HVRNNInitStates(p,l) for l in range(p['layers'])])
		self.q_init = mutils.ListModule(*[HVRNNInitStates(p,l) for l in range(p['layers'])])
		self.p_init = mutils.ListModule(*[HVRNNInitStates(p,l) for l in range(p['layers'])])
		
		self.encoder     = mutils.ListModule(*[HVRNNEncoder(p,l)    for l in range(p['layers'])])
		self.decoder     = mutils.ListModule(*[HVRNNDecoder(p,l)    for l in range(p['layers'])])
		
		self.p_net       = mutils.ListModule(*[HVRNNPrior(p,l)      for l in range(p['layers'])])
		self.q_net       = mutils.ListModule(*[HVRNNPosterior(p,l)  for l in range(p['layers'])])
	
		if p['gpu']:
			self.cuda()
		
	def plot(self, i, input_image, plot_vars, vis):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
		matsdir = os.path.join(self.p['plot_dir'], self.p['model_name'], 'mats')
		
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		os.makedirs(matsdir) if not os.path.exists(matsdir) else None
	
		save_image(pred[0,-3:].data.cpu(), pdir+'/p{}.png'.format(i))

		save_image(input_image[0,0,-1,-3:].data.cpu(), pdir+'/b{}.png'.format(i))
			
		x_data = self.methandle.metrics['iter']
		y1 	   = self.methandle.metrics['recon']
		y2 	   = self.methandle.metrics['cont_kl']
		y3 	   = self.methandle.metrics['disc_kl']
		
		
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(x_data,y1, label='recon')
		ax.plot(x_data,y2, label='cont_kl')
		ax.plot(x_data,y3, label='disc_kl')
		fig.savefig(os.path.join(pdir,'m{}.png'.format(i)))
		plt.close(fig)
		#vis(i)
	def reset(self):
		self.iter_loss = 0
		pass
				
	def loss(self, curiter, x, x_hat, q_zparams, p_zparams, eval=False):

		loss 			   = 0.		
		cont_kl = 0
		kl_sum = []

		#err_loss = self.mse(image, pred)	
		flat_dim_idx = np.prod(self.p['imdim'])
		flat_dim = torch.tensor(np.prod(self.p['imdim'])).float()
		err_loss = F.binary_cross_entropy(x_hat.view(-1,flat_dim_idx), x.view(-1,flat_dim_idx))
		
		loss += (err_loss * flat_dim)
		
		if not p_zparams:
		
			# assuming annealing params are constant over layers
			cont_min, cont_max, cont_num_iters, cont_gamma = self.p['z_con_capacity'][0]		
			cont_cap_current = (cont_max - cont_min) * cur_iter / float(cont_num_iters) + cont_min
			cont_cap_current = min(cont_cap_current, cont_max)

			# Calculate continuous capacity loss
		
			for l in range(self.p['layers']):
				
				qmu, qsig = torch.chunk(q_zparams[l], 2, dim=-1)
				pmu, psig = torch.chunk(p_zparams[l], 2, dim=-1)

				elemwise_kl = 0.5 * (torch.log(psig) - torch.log(qsig) + qsig / psig + (qmu - pmu).pow(2) / pvsig - 1)
				cont_kl = cont_gamma * torch.abs(cont_cap_current - elemwise_kl .sum(-1))
				loss += cont_kl
				kl_sum.append(cont_kl.item())
				
		loss /= flat_dim
		
		metrics = {'iter':curiter, 
				   'recon':err_loss.item(), 
				   'cont_kl':sum(kl_sum),
				   'disc_kl':0}

		return loss, metrics
	
	def decode(self, z, l):
		return self.decoder[l](z, l).data

	def decode_seq(self, z, with_context=False):

		predictions = [[] for i in range(self.p['layers'])]
		d_h = [[] for i in range(self.p['layers'])]
		d_c = [[] for i in range(self.p['layers'])]  
		for t in range(z[0].shape[1]):
			for l_r in reversed(range(self.p['layers'])):
				qz_sample = z[l_r][:,t]
				# Decoding - p(x|z)
				#dec_xin = cat(z_qparams, pred[l_r+1], dim=1) if l_r < 2 else qz_sample
				if t == 0:
					hidden = None
				else:
					hidden = (d_h[l_r],d_c[l_r])

				pred, d_h[l_r], d_c[l_r] = self.decoder[l_r](hidden, qz_sample)

				predictions[l_r].append(pred)

		for l in range(self.p['layers']):
			#print(predictions[l_r].shape)
			predictions[l] = torch.stack(predictions[l], dim=1)
		return predictions
	
	def forward(self, iter, imseq, eval=False, to_matlab=False, plotting = False):
		
		imseq = imseq[0]
		print(imseq.shape)
		print(imseq[0].shape)
		context = imseq[:,:self.p['n_context']] 
		imseq = imseq[:,self.p['n_context']:] 

		q_h = [[] for i in range(self.p['layers'])]  ; q_c = [[] for i in range(self.p['layers'])]  
		p_h = [[] for i in range(self.p['layers'])]  ; p_c = [[] for i in range(self.p['layers'])]  
		
		q_zp = [[] for i in range(self.p['layers'])] ; p_zp = [[] for i in range(self.p['layers'])]
		d_h = [[] for i in range(self.p['layers'])]  ; d_c = [[] for i in range(self.p['layers'])] ; 
		enc  = [[] for i in range(self.p['layers'])] ; pred  = [[] for i in range(self.p['layers'])]
		z_pparams = [[] for i in range(self.p['layers'])] ; z_qparams  = [[] for i in range(self.p['layers'])]
		prediction_plots = []
	
		init_decoder_states  = []
		# -----------------------------------------
		# Initialise Q/P/D States w/ context frames
		# -----------------------------------------
		layers = [x for x in range(self.p['layers'])]
		
		with torch.no_grad():
			decinit = [] ; infinit = []
			for c in range(self.p['n_context']):
				for l in layers:
										
					if l == 0:
						enc[l] = self.encoder[l](context[:,c])
					else:
						enc[l] = self.encoder[l](enc[l-1])

					d_h[l], d_c[l] = self.d_init[l](enc[l])
					q_h[l], q_c[l] = self.q_init[l](enc[l])
					p_h[l], p_c[l] = self.p_init[l](enc[l])

					q_h[l] = q_h[l][-1] ; q_c[l] = q_c[l][-1]
					p_h[l] = p_h[l][-1] ; p_c[l] = p_c[l][-1]
					d_h[l] = [x for x in reversed(d_h[l])]
					d_c[l] = [x for x in reversed(d_c[l])]
				
		del l, c
		
		# Forward 
		#for t in range(imseq.shape[1]-1):
		for t in range(150):

			# Frame Encoder / ResNet 16
			for l in range(self.p['layers']):
				if l == 0:
					if t >= imseq.shape[1]:
						enc[l] = self.encoder[l](pred[0])
					else:
						enc[l] = self.encoder[l](imseq[:,t])
				else:
					enc[l] = self.encoder[l](enc[l-1])
			del l 

			# Backward 
			for l_r in reversed(range(self.p['layers'])):
				
				qnet_z_in = None
				pnet_f_in = enc_prev[l_r][-1] if t > 0 else None 
				qnet_f_in = enc[l_r][-1] 

				# Step LSTMs & sample z
				q_hidden = (q_h[l_r],q_c[l_r])
				z_qparams[l_r], q_h[l_r], q_c[l_r] = self.q_net[l_r](q_hidden, qnet_z_in, qnet_f_in)
				qz_sample = self.q_dist[l_r].sample_normal(z_qparams[l_r], self.training)

				if t > 0:
					#pnet_z_in = cat(pz_sample[l_r], dim=1) if l_r < 2 else None
					pnet_z_in = None
					p_hidden = (p_h[l_r],p_c[l_r])
					z_pparams[l_r], p_h[l_r], p_c[l_r] = self.p_net[l_r](p_hidden, pnet_z_in, pnet_f_in)
					pz_sample = self.p_dist[l_r].sample_normal(z_pparams[l_r], self.training)

				# Decoding - p(x|z)
				#dec_xin = cat(z_qparams, pred[l_r+1], dim=1) if l_r < 2 else qz_sample
				dec_xin = qz_sample

				d_hidden = (d_h[l_r],d_c[l_r])
				pred[l_r], d_h[l_r], d_c[l_r] = self.decoder[l_r](d_hidden, dec_xin)
				if l_r == 0 and plotting:
					prediction_plots.append(pred[l_r])

			enc_prev = enc

			if self.training:
			
				iter_loss, metrics = self.loss(iter, imseq[:,t+1,-self.p['chans']:], pred[0], z_qparams, z_pparams) 
				if iter % 10000:
					self.methandle.extend(metrics)

				self.iter_loss += iter_loss			
				
			del l_r
		if plotting:
				return prediction_plots
		if to_matlab:

			return self.z[0].detach().numpy(), foveated.detach().numpy()
			
		elif eval:
			return z_pparams, pred[0]

		elif self.enc_mode:
			return self.z[0]

class HVRNNSpatial(Module):
	def __init__(self, p):
		super(HVRNNSpatial,self).__init__()
		
		self.p = p
		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		self.methandle = tutils.MetricsHandler() 		
		
		self.mse  = MSELoss(reduction='sum').cuda() if p['gpu'] else MSELoss()
		self.iter_loss = 0.0

		
		# Initialise Distributions
		self.p_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(p)
		
		p['z_params']	= self.q_dist[0].nparams 

		self.encoder     = mutils.ListModule(*[HVRNNEncoder(p,l)    for l in range(p['layers'])])
		self.decoder     = mutils.ListModule(*[HVRNNSpatialDecoder(p,l)    for l in range(p['layers'])])
		
		self.p_net       = mutils.ListModule(*[HVRNNSpatialPrior(p,l)      for l in range(p['layers'])])
		self.q_net       = mutils.ListModule(*[HVRNNSpatialPosterior(p,l)  for l in range(p['layers'])])
	
		if p['gpu']:
			self.cuda()
		
	def plot(self, i, input_image, plot_vars, vis):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
		matsdir = os.path.join(self.p['plot_dir'], self.p['model_name'], 'mats')
		
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		os.makedirs(matsdir) if not os.path.exists(matsdir) else None
	
		save_image(pred[0,-3:].data.cpu(), pdir+'/p{}.png'.format(i))

		save_image(input_image[0,0,-1,-3:].data.cpu(), pdir+'/b{}.png'.format(i))
			
		x_data = self.methandle.metrics['iter']
		y1 	   = self.methandle.metrics['recon']
		y2 	   = self.methandle.metrics['cont_kl']
		y3 	   = self.methandle.metrics['disc_kl']
		
		
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(x_data,y1, label='recon')
		ax.plot(x_data,y2, label='cont_kl')
		ax.plot(x_data,y3, label='disc_kl')
		fig.savefig(os.path.join(pdir,'m{}.png'.format(i)))
		plt.close(fig)
		#vis(i)
	def reset(self):
		self.iter_loss = 0
		pass
				
	def loss(self, curiter, x, x_hat, q_zparams, p_zparams, eval=False):

		loss 			   = 0.		
		cont_kl = 0
		kl_sum = []

		#err_loss = self.mse(image, pred)	
		flat_dim_idx = np.prod(self.p['imdim'])
		flat_dim = torch.tensor(np.prod(self.p['imdim'])).float()
		err_loss = F.binary_cross_entropy(x_hat.view(-1,flat_dim_idx), x.view(-1,flat_dim_idx))
		
		loss += (err_loss * flat_dim)
		
		if not p_zparams:
		
			# assuming annealing params are constant over layers
			cont_min, cont_max, cont_num_iters, cont_gamma = self.p['z_con_capacity'][0]		
			cont_cap_current = (cont_max - cont_min) * cur_iter / float(cont_num_iters) + cont_min
			cont_cap_current = min(cont_cap_current, cont_max)

			# Calculate continuous capacity loss
		
			for l in range(self.p['layers']):
				
				qmu, qsig = torch.chunk(q_zparams[l], 2, dim=-1)
				pmu, psig = torch.chunk(p_zparams[l], 2, dim=-1)

				elemwise_kl = 0.5 * (torch.log(psig) - torch.log(qsig) + qsig / psig + (qmu - pmu).pow(2) / pvsig - 1)
				cont_kl = cont_gamma * torch.abs(cont_cap_current - elemwise_kl .sum(-1))
				loss += cont_kl
				kl_sum.append(cont_kl.item())
				
		loss /= flat_dim
		
		metrics = {'iter':curiter, 
				   'recon':err_loss.item(), 
				   'cont_kl':sum(kl_sum),
				   'disc_kl':0}

		return loss, metrics
	
	def decode(self, z, l):
		return self.decoder[l](z, l).data

	def decode_seq(self, z, with_context=False):

		predictions = [[] for i in range(self.p['layers'])]
		d_h = [[] for i in range(self.p['layers'])]
		d_c = [[] for i in range(self.p['layers'])]  
		for t in range(z[0].shape[1]):
			for l_r in reversed(range(self.p['layers'])):
				qz_sample = z[l_r][:,t]
				# Decoding - p(x|z)
				#dec_xin = cat(z_qparams, pred[l_r+1], dim=1) if l_r < 2 else qz_sample
				if t == 0:
					hidden = None
				else:
					hidden = (d_h[l_r],d_c[l_r])

				pred, d_h[l_r], d_c[l_r] = self.decoder[l_r](hidden, qz_sample)

				predictions[l_r].append(pred)

		for l in range(self.p['layers']):
			#print(predictions[l_r].shape)
			predictions[l] = torch.stack(predictions[l], dim=1)
		return predictions
	
	def forward(self, iter, imseq, eval=False, to_matlab=False, plotting = False):
		
		imseq = imseq[0]
		print(imseq.shape)
		print(imseq[0].shape)
		context = imseq[:,:self.p['n_context']] 
		imseq = imseq[:,self.p['n_context']:] 

		q_h = [[] for i in range(self.p['layers'])]  ; q_c = [[] for i in range(self.p['layers'])]  
		p_h = [[] for i in range(self.p['layers'])]  ; p_c = [[] for i in range(self.p['layers'])]  
		
		q_zp = [[] for i in range(self.p['layers'])] ; p_zp = [[] for i in range(self.p['layers'])]
		d_h = [[] for i in range(self.p['layers'])]  ; d_c = [[] for i in range(self.p['layers'])] ; 
		enc  = [[] for i in range(self.p['layers'])] ; pred  = [[] for i in range(self.p['layers'])]
		z_pparams = [[] for i in range(self.p['layers'])] ; z_qparams  = [[] for i in range(self.p['layers'])]
		prediction_plots = []
	
		init_decoder_states  = []
		
		# Forward 
		for t in range(imseq.shape[1]-1):
		#for t in range(150):

			# Frame Encoder / ResNet 16
			for l in range(self.p['layers']):
				if l == 0:
					if t >= imseq.shape[1]:
						enc[l] = self.encoder[l](pred[0])
					else:
						enc[l] = self.encoder[l](imseq[:,t])
				else:
					enc[l] = self.encoder[l](enc[l-1])
			del l 

			# Backward 
			for l_r in reversed(range(self.p['layers'])):
				
				qnet_z_in = None
				pnet_f_in = enc_prev[l_r][-1] if t > 0 else None 
				qnet_f_in = enc[l_r][-1] 

				# Step LSTMs & sample z
				q_hidden = (q_h[l_r],q_c[l_r]) if t>0 else None
				print(qnet_f_in.shape)
				z_qparams[l_r], q_h[l_r], q_c[l_r] = self.q_net[l_r](q_hidden, qnet_z_in, qnet_f_in)
				qz_sample = self.q_dist[l_r].sample_normal(z_qparams[l_r], self.training)

				if t > 0:
					#pnet_z_in = cat(pz_sample[l_r], dim=1) if l_r < 2 else None
					pnet_z_in = None
					p_hidden = (p_h[l_r],p_c[l_r]) if t>0 else None
					z_pparams[l_r], p_h[l_r], p_c[l_r] = self.p_net[l_r](p_hidden, pnet_z_in, pnet_f_in)
					pz_sample = self.p_dist[l_r].sample_normal(z_pparams[l_r], self.training)

				# Decoding - p(x|z)
				#dec_xin = cat(z_qparams, pred[l_r+1], dim=1) if l_r < 2 else qz_sample
				dec_xin = qz_sample

				d_hidden = (d_h[l_r],d_c[l_r])
				pred[l_r], d_h[l_r], d_c[l_r] = self.decoder[l_r](d_hidden, dec_xin)
				if l_r == 0 and plotting:
					prediction_plots.append(pred[l_r])

			enc_prev = enc

			if self.training:
			
				iter_loss, metrics = self.loss(iter, imseq[:,t+1,-self.p['chans']:], pred[0], z_qparams, z_pparams) 
				if iter % 10000:
					self.methandle.extend(metrics)

				self.iter_loss += iter_loss			
				
			del l_r
		if plotting:
				return prediction_plots
		if to_matlab:

			return self.z[0].detach().numpy(), foveated.detach().numpy()
			
		elif eval:
			return z_pparams, pred[0]

		elif self.enc_mode:
			return self.z[0]

class HVRNNPrednet(HVRNN):
	def __init__(self, p):
		super(HVRNNPrednet,self).__init__(p)
	
		self.e_err = mutils.ListModule(*[ErrorUnit(p,l) for l in range(p['layers'])])

	
	def forward(self, iter, imseq, eval=False, to_matlab=False, plotting = False):
		
		imseq = imseq[0]
		context = imseq[:,:self.p['n_context']] 
		imseq = imseq[:,self.p['n_context']:] 

		q_h = [[] for i in range(self.p['layers'])]  ; q_c = [[] for i in range(self.p['layers'])]  
		p_h = [[] for i in range(self.p['layers'])]  ; p_c = [[] for i in range(self.p['layers'])]  
		
		q_zp = [[] for i in range(self.p['layers'])] ; p_zp = [[] for i in range(self.p['layers'])]
		d_h = [[] for i in range(self.p['layers'])]  ; d_c = [[] for i in range(self.p['layers'])] ; 
		enc  = [[] for i in range(self.p['layers'])] ; pred  = [[] for i in range(self.p['layers'])]
		z_pparams = [[] for i in range(self.p['layers'])] ; z_qparams  = [[] for i in range(self.p['layers'])]
		prediction_plots = []
	
		init_decoder_states  = []
		# -----------------------------------------
		# Initialise Q/P/D States w/ context frames
		# -----------------------------------------
		layers = [x for x in range(self.p['layers'])]
		
		with torch.no_grad():
			decinit = [] ; infinit = []
			for c in range(self.p['n_context']):
				for l in layers:
										
					if l == 0:
						enc[l] = self.encoder[l](context[:,c])
					else:
						enc[l] = self.encoder[l](enc[l-1])

					d_h[l], d_c[l] = self.d_init[l](enc[l])
					q_h[l], q_c[l] = self.q_init[l](enc[l])
					p_h[l], p_c[l] = self.p_init[l](enc[l])

					q_h[l] = q_h[l][-1] ; q_c[l] = q_c[l][-1]
					p_h[l] = p_h[l][-1] ; p_c[l] = p_c[l][-1]
					d_h[l] = [x for x in reversed(d_h[l])]
					d_c[l] = [x for x in reversed(d_c[l])]

		del l, c
		err = [[torch.ones_like(x) for x in enc[l]] for l in range(self.p['layers'])]

		# Forward 
		#for t in range(imseq.shape[1]-1):
		for t in range(150):

			# Backward 
			for l_r in reversed(range(self.p['layers'])):
				
				qnet_z_in = None
				pnet_f_in = enc_prev[l_r][-1] if t > 0 else None 
				qnet_f_in = enc[l_r][-1] 

				# Step LSTMs & sample z
				q_hidden = (q_h[l_r],q_c[l_r])
				z_qparams[l_r], q_h[l_r], q_c[l_r] = self.q_net[l_r](q_hidden, qnet_z_in, qnet_f_in)
				qz_sample = self.q_dist[l_r].sample_normal(z_qparams[l_r], self.training)

				if t > 0:
					#pnet_z_in = cat(pz_sample[l_r], dim=1) if l_r < 2 else None
					pnet_z_in = None
					p_hidden = (p_h[l_r],p_c[l_r])
					z_pparams[l_r], p_h[l_r], p_c[l_r] = self.p_net[l_r](p_hidden, pnet_z_in, pnet_f_in)
					pz_sample = self.p_dist[l_r].sample_normal(z_pparams[l_r], self.training)

				# Decoding - p(x|z)
				#dec_xin = cat(z_qparams, pred[l_r+1], dim=1) if l_r < 2 else qz_sample
				dec_xin = qz_sample

				d_hidden = (d_h[l_r],d_c[l_r])
				pred[l_r], d_h[l_r], d_c[l_r] = self.decoder[l_r](d_hidden, dec_xin)
				if l_r == 0 and plotting:
					prediction_plots.append(pred[l_r])

			# Frame Encoder / ResNet 16
			for l in range(self.p['layers']):
				
				if l == 0:
					if t >= imseq.shape[1]:
						err[l] = self.e_err[l](prediction_plots[-1], pred[l])	
						enc[l] = self.encoder[l](err[l])						
					else:
						err[l] = self.e_err[l](imseq[:,t], pred[l])	
						enc[l] = self.encoder[l](err[l])
				else:
					err[l] = self.e_err[l](enc[l-1][-1], pred[l])
					enc[l] = self.encoder[l]([None,err[l]])

			del l

			enc_prev = enc

			if self.training:
			
				iter_loss, metrics = self.loss(iter, imseq[:,t+1,-self.p['chans']:], pred[0], z_qparams, z_pparams) 
				if iter % 10000:
					self.methandle.extend(metrics)

				self.iter_loss += iter_loss			
				
			del l_r
		if plotting:
				return prediction_plots
		if to_matlab:

			return self.z[0].detach().numpy(), foveated.detach().numpy()
			
		elif eval:
			return z_pparams, pred[0]

		elif self.enc_mode:
			return self.z[0]

class ResNet(Module):
	def __init__(self, p):
		super(ResNet, self).__init__()
		
		self.p = p
		self.methandle = tutils.MetricsHandler() 
		
		self.iter_loss = 0.0		
		self.has_con = self.p['nz_con'][0] > 0 
		self.has_dis = self.p['nz_dis'][0][0] > 0		
				
		# Initialise Distributions
		self.prior_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(p)
		p['z_params']	= self.q_dist.nparams
			 
		self.resnet = ResNet_VAE(CNN_embed_dim=self.p['nz_con'][0])
		self.reset()
		
		if self.p['gpu']:
			self.cuda()
			
	def plot(self, i, input_image, plot_vars, vis):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
		matsdir = os.path.join(self.p['plot_dir'], self.p['model_name'], 'mats')
		
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		os.makedirs(matsdir) if not os.path.exists(matsdir) else None
	
		save_image(pred[0].data.cpu(), pdir+'/p{}.png'.format(i))
		save_image(input_image[0].data.cpu(), pdir+'/b{}.png'.format(i))
		
			
		x_data = self.methandle.metrics['iter']
		y1	   = self.methandle.metrics['recon']
		y2	   = self.methandle.metrics['cont_kl']
		y3	   = self.methandle.metrics['disc_kl']
		
		
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(x_data,y1, label='recon')
		ax.plot(x_data,y2, label='cont_kl')
		ax.plot(x_data,y3, label='disc_kl')
		fig.savefig(os.path.join(pdir,'m{}.png'.format(i)))
		plt.close(fig)

		z	= z.data.cpu().numpy()
		sio.savemat(os.path.join(matsdir,'z_{}.mat'.format(i)), {'r':z})
		vis(i)
	
	def reset(self):
		# clears computation graph for next batch
		self.iter_loss = 0
		self.plot_errs = []

	def decode(self, z, l):
		
		return self.resnet.decode(z)

	def loss(self, curiter, image, pred, z_pc, z_pd, eval=False):
		# MSE = F.mse_loss(recon_x, x, reduction='sum')
		loss			   = 0.		
		train_loss		   = [] 
		train_norm_kl_loss = []
		train_cat_kl_loss  = [] 
		layer_loss		   = 0.

		#err_loss = self.mse(image, pred)	
		flat_dim = np.prod(self.p['imdim'])

		err_loss = F.binary_cross_entropy(pred.view(-1,flat_dim), image.view(-1,flat_dim))
		err_loss *= flat_dim

		if self.has_con:
			kloss_args	= (z_pc,   # mu, sig
						   self.p['z_con_capacity'][0], # anealing params
						   curiter)	# data size
						   
			norm_kl_loss = self.q_dist.calc_kloss(*kloss_args) #/ self.p['b']
		else:
			norm_kl_loss = torch.tensor(0)
		
		if self.has_dis:
			kloss_args	 = (z_pd,  # alpha
							self.p['z_dis_capacity'][0],  # anneling params 
							self.p['nz_dis'][0], # nclasses per categorical dimension
							curiter)	# data size
						  
			cat_kl_loss = self.cat_dist.calc_kloss(*kloss_args) #/ self.p['b']
		
		else:
			cat_kl_loss = torch.tensor(0)
		
		if self.p['elbo_loss']:	
			layer_loss = norm_kl_loss + cat_kl_loss + err_loss
		else:
			layer_loss = err_loss 
		
		loss = layer_loss / flat_dim
		
		metrics = {'iter':curiter, 
				   'recon':err_loss.item(), 
				   'cont_kl':norm_kl_loss.item(), 
				   'disc_kl':cat_kl_loss.item()}

		return loss, metrics
	
	def forward(self, iter, image, actions=None, eval=False, to_matlab=False):
		if to_matlab:
			self.p['gpu'] = False
			self.reset()
			if not isinstance(image, Tensor) or isinstance(image, FloatTensor):
				image = FloatTensor(np.asarray(image)).unsqueeze(0).unsqueeze(0)
		
		x_hat, z, mu, logvar = self.resnet(image.squeeze(0))  # VAE
	
		if self.training:

			iter_loss, metrics = self.loss(iter, image, x_hat, torch.cat([mu, logvar], dim=1), None) 
			if iter % 10000:
				self.methandle.extend(metrics)

			self.iter_loss += iter_loss

		if to_matlab:
			return z.detach().numpy(), foveated.detach().numpy()
			
		elif eval:
			return z, x_hat
	
class ObservationVAE(Module):
	def __init__(self, p):
		super(ObservationVAE,self).__init__()
		
		self.p = p
		self.methandle = tutils.MetricsHandler() 

		self.plot_errs = []
		
		#self.mse  = MSELoss(reduction='sum').cuda() if p['gpu'] else MSELoss()
		
		self.iter_loss = 0.0
			
		# Initialise Distributions
		self.prior_dist, self.q_dist = mutils.discheck(p)
		p['z_params']	= self.q_dist.nparams	 

		# Initialise Modules - learnable
		# change all to encoder, decoder

		#self.f_enc = Resnet_AnimalAI_Encoder(p)
		#self.g_dec = Resnet_AnimalAI_Decoder(p)

		self.f_enc = ConvEncoderAnAI(p)
		self.g_dec = ConvDecoderAnAI(p)
													
		self.iter_loss = 0

		if p['gpu']:
			self.cuda()
		
	def plot(self, i, input_image, plot_vars, vis):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
	
		os.makedirs(pdir) if not os.path.exists(pdir) else None
	
		save_image(pred[0].data.cpu(), pdir+'/p{}.png'.format(i))
		save_image(input_image[0].data.cpu(), pdir+'/b{}.png'.format(i))
			
		vis(i)
			
	def loss(self, curiter, image, pred, z_pc,  eval=False):

		loss			   = 0.		
		train_loss		   = [] 
		train_norm_kl_loss = []

		#err_loss = self.mse(image, pred)	
		flat_dim = np.prod(self.p['imdim'])
		
		err_loss = F.binary_cross_entropy(pred.view(-1,flat_dim), image.view(-1,flat_dim))
		err_loss *= flat_dim

		kloss_args	= (z_pc,   # mu, sig
					   self.p['z_con_capacity'], # anealing params
					   curiter)	# data size
					   
		norm_kl_loss = self.q_dist.calc_kloss(*kloss_args) #/ self.p['b']
		
		layer_loss = norm_kl_loss + err_loss
		
		loss = layer_loss / flat_dim

		metrics = {'iter':curiter, 
				   'recon':err_loss.item(), 
				   'cont_kl':norm_kl_loss.item()}

		return loss, metrics
	
	def decode(self, z, l):
		return self.g_dec(z).data
	
	
	def spatial_decode(self, z, l, matlab=False):
		dis = z[:,-10:]
		con = z[:,0]
		con = con.view(-1, 1, 1,1).repeat(1,1,7,7).view(-1,49)
		z = torch.cat((con,dis),dim=1)
		return self.g_dec(z).data
		
	def encode(self, im, l):
		z_pc = self.f_enc(im)
		z = self.q_dist.sample_normal(params=z_pc, train=self.training)
		return z		
	
	def foveate(self, im, a):
		image, foveated = self.retina.foveate(im, a)
		return foveated.numpy()		
	
	def forward(self, iter, image, actions=None, eval=False, to_matlab=False):
		
		if to_matlab:
			
			self.p['gpu'] = False
			self.reset()
			if not isinstance(image, Tensor) or isinstance(image, FloatTensor):
				image = FloatTensor(np.asarray(image)).unsqueeze(0).unsqueeze(0)

			if not actions is None:
				actions = FloatTensor(actions).unsqueeze(0).unsqueeze(0)
		
		# Encoding - p(z2|x) or p(z1 |x,z2)
		z_pc = self.f_enc(image)
		
		z = self.q_dist.sample_normal(params=z_pc, train=self.training)
		# Decoding - p(x|z)
		pred = self.g_dec(z)

		if self.training:
			
			iter_loss, metrics = self.loss(iter, image, pred, z_pc) 
			if iter % 10000:
				self.methandle.extend(metrics)

			self.iter_loss += iter_loss

		if to_matlab:
			return z.detach().numpy(), pred.detach().numpy()
			
		return z, pred

class PrednetWorldModel(Module):
	def __init__(self, p):
		super(PrednetWorldModel,self).__init__()
		
		self.p = p
		
		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		
		self.mse  = MSELoss(reduction='sum').cuda() if p['gpu'] else MSELoss()
		self.iter_loss = 0.0
		
		#self.xent = CrossEntropyLoss().cuda() if p['gpu'] else CrossEntropyLoss()
		#self.l1 = L1Loss().cuda() if p['gpu'] else L1Loss()
		
		self.brain = BrainParameters(brain_name='Learner',
									 camera_resolutions=[{'height': 84, 'width': 84, 'blackAndWhite': False}],
									 num_stacked_vector_observations=self.p['b'],
									 vector_action_descriptions=['', ''],
									 vector_action_space_size=[3, 3],
									 vector_action_space_type=0,  # corresponds to discrete
									 vector_observation_space_size=3)		
		
		# Initialise Distributions
		self.prior_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(self.p)

		self.q_dist = self.q_dist
		self.cat_dist = self.cat_dist		
		
		p['z_params']	= self.q_dist.nparams
		# Calc RF masks for dataset 
		masks, p = mutils.calc_rf(p)
		full_mask, reduce_mask = masks

		# Initialise Modules - learnable
		self.f_enc = ConvEncoder(p,0)
		self.g_dec = ConvDecoder(p,0)

		# Putting this in a list precludes it from the parent model's graph 
		self.a_net	= [ActionNet(p,0)]

		# Initialise Modules - non-learnable
		self.g_obs = ObsModel(p,masks,0)
		self.e_err = ErrorUnit(p,full_mask,0)
		
		self.lstm  = [DynamicModel(p,0)]
		
		# Initialise Tensors
		self.reset()

		if p['gpu']:
			self.cuda()
			self.a_net[0].cuda()
			self.lstm[0].cuda()
		
	def plot(self, i, input_image, plot_vars):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
		matsdir = os.path.join(self.p['plot_dir'], self.p['model_name'], 'mats')
		
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		os.makedirs(matsdir) if not os.path.exists(matsdir) else None
	
		save_image(pred[0].data.cpu(), pdir+'/p{}.png'.format(i))
		save_image(input_image[0].data.cpu(), pdir+'/b{}.png'.format(i))
	
	def reset(self):
		# reset / init all model variables 
		# call before each batch
		
		# clears computation graph for next batch
		self.iter_loss = 0
		self.plot_errs = []

		# Initialise Distributions
		del self.q_dist, self.cat_dist
		self.prior_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(self.p)

		self.q_dist = self.q_dist
		self.cat_dist = self.cat_dist
		
		#self.lstm[0].reset()
			
	def loss(self, curiter, pred, target, z_pc, z_pd, eval=False):

		loss = 0.
		
		train_loss = []	 
		train_norm_kl_loss = [] 
		train_cat_kl_loss = [] 
		
		#err_loss  = torch.abs(self.error[l]).sum() / np.prod(self.p['ldim'][0][1:])# * layer_weights
		
		err_loss = self.mse(pred, target)
		
		kloss_args	= (z_pc,   # mu, sig
					   self.p['z_con_capacity'][0], # anealing params
					   curiter)	# data size
					   
		norm_kl_loss = self.q_dist.calc_kloss(*kloss_args) #/ self.p['b']
		
		kloss_args	 = (z_pd,  # alpha
						self.p['z_dis_capacity'][0],  # anneling params 
						self.p['nz_dis'][0], # nclasses per categorical dimension
						curiter)	# data size
					  
		cat_kl_loss = self.cat_dist.calc_kloss(*kloss_args) #/ self.p['b']
		
		#err_loss /= (32*32)

		if self.p['elbo_loss']:	
			layer_loss = norm_kl_loss + cat_kl_loss + err_loss
		else:
			layer_loss = err_loss 

		loss += layer_loss #/ np.prod(self.p['imdim'][1:])
		#loss /= self.p['b']#(32*32)

		train_loss.append(err_loss.item())
		train_norm_kl_loss.append(norm_kl_loss.item())
		train_cat_kl_loss.append(cat_kl_loss.item())
		
		metrics = [m/np.prod(self.p['imdim'][1:]) for m in (train_loss[0], train_norm_kl_loss[0], train_cat_kl_loss[0])]

		#loss /= self.p['b']#np.prod(self.p['imdim'][1:])
		#print(err_loss.item(), norm_kl_loss.item(), cat_kl_loss.item())
		#loss /= 32*32
		#return loss, tuple(x[0]/self.p['b'] for x in metrics)

		return loss, metrics
	
	def decode(self, z, l):
		return self.g_dec[l](z).data
	
	def forward(self, iter, image,	actions=None, eval=False, to_matlab=False):
	
		# Encoding - p(z2|x) or p(z1 |x,z2)
		z_pc, z_pd = self.f_enc(image, None)
		
		# Latent Sampling
		latent_sample = []

		# Continuous sampling 
		norm_sample = self.q_dist.sample_normal(params=z_pc, train=self.training)
		latent_sample.append(norm_sample)

		# Discrete sampling
		for ind, alpha in enumerate(z_pd):
			cat_sample = self.cat_dist.sample_gumbel_softmax(alpha, train=self.training)
			latent_sample.append(cat_sample)

		z = torch.cat(latent_sample, dim=-1)

		#self.z[l_r], done = self.lstm[l_r](z, actions[:,t])
		z_pred = self.lstm[0](z.detach(), actions.detach())

		# Decoding - p(x|z)
		pred = self.g_dec(z)

		# Bottom up			
		#obs   = self.g_obs(image)
		#error = self.e_err(obs, pred, None)
			   # this is not a zeroth layer index (see init)
		#return self.a_net[0](self.z[0].detach(), self.lstm.lstm_h.detach(), actions)
		actions = self.a_net[0](z, self.lstm[0].lstm_h.clone(), actions)
		return actions, pred, z_pc, z_pd, z, z_pred