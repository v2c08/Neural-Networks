import os
import glob
import numpy as np
from os.path import join 
from scipy.io import savemat
from torch.optim import Adam
from logging import getLogger
from model import ObservationVAE
from torch import cuda, no_grad, save, load, cat, zeros, argmax
from torch import FloatTensor, tensor, Tensor, zeros_like
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torch.autograd import Variable
from utils import train_utils as tutils
from utils import model_utils as mutils
from utils import data_utils as dutils
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig

class Trainer(object):	

	def __init__(self, p, dataloader):
		
		self.p = p
		self.train_loader = dataloader[0]
		self.test_loader   = dataloader[1]		
		
		self.iteration = 0 
		self.plot_iter = 0
		self.logger = getLogger('train')
	
	def train_epoch(self):
		""" All items from dataloader are passed to forward.
			Handle input data in the model's forward function, 
			or overwrite this function in a subordinate class. """
			
		epoch_loss = 0
		
		for data in self.train_loader:
			
			self.model.iter_loss = 0
			
			if isinstance(data, list):
				data = [x.cuda() if self.p['gpu' ] else x for x in data]	
			else:
				data = [data.cuda() if self.p['gpu'] else data]
			
			# forward 
			self.model(self.iteration, data[0])
			
			# backward
			self.optimizer.zero_grad()
			self.model.iter_loss.backward()
			self.optimizer.step()
			
			epoch_loss += self.model.iter_loss.item()
			
			self.iteration += 1

		epoch_loss = epoch_loss / len(self.train_loader.dataset)
		self.logger.info('Mean Epoch Loss - {}'.format(epoch_loss))
				
	def eval_batch(self, e, force_write=None):
		

		for data in self.test_loader:
			if isinstance(data, list):
				data = [x.cuda() if self.p['gpu' ] else x for x in data]	
			else:
				data = [data.cuda() if self.p['gpu'] else data]
			data = data[0]

			plot_vars = self.model(self.iteration, data, eval=True)
			break 


		if (e % self.p['plot_iter'] == 0) or force_write:
			
			self.model.plot(self.iteration, data, plot_vars, self.vis)

			self.plot_iter += 1


		tutils.save_checkpoint({'model': self.model, 
								'state_dict': self.model.state_dict(),
								'args': self.model.p}, 
								 self.model.p['model_dir'],  
								 self.model.p['model_name'], 0)	
								
			
	def vis(self, e):
		mutils.visualise(self.model.p, self.model, e, self.test_loader)
	
class ObservationTrainer(Trainer):
	""" Inherits from base Trainer class,
		used to train basic vision models """

	def __init__(self, p, dataloader, model):
		super(ObservationTrainer, self).__init__(p, dataloader)
		
		self.model = model
		try:
			self.model.p['datasize']  = len(dataloader[0].dataset)
		except:
			self.model.p['datasize']  = 50000
		self.model.p['n_batches'] = len(dataloader[0])
		self.model.p['n_iter']    = self.model.p['datasize'] * p['e']

		self.optimizer = Adam(self.model.parameters(), lr=p['lr'])		
		
	def train(self):
		
		self.model.train() 

		self.model.p = tutils.set_paths(self.p, 'obs_model')


		self.logger.info('\n Training Observation Model \n ')
		self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))

		#while self.iteration < self.model.p['n_iter']:
		for e in range(self.p['e']):
		#while self.iteration < 1:
			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.model.train()
			self.train_epoch()
			with no_grad():
				self.model.eval()
				self.eval_batch(e)
		self.eval_batch(e, force_write=True)

class SaccadeTrainer(Trainer):
	"""
		Used to train models that can saccade 
		around a visual scene
	"""

	def __init__(self, p, dataloader):
		super(SaccadeTrainer, self).__init__(p, dataloader)
		
		self.obs_model_path = join(p['model_dir'], p['model_name'])
		self.obs_model  = None	
	
	def prep_latent_dataloaders(self, obs_model):
		with no_grad():
			latent_data  = []
			action_data  = []
			_actshape = (-1, self.p['n_steps'], 1, 2)
			indices = list(range(self.p['n_actions']))

			for b, data in enumerate(self.train_loader):

				z   = zeros(self.p['b'], self.p['n_steps'], self.p['z_dim'][0]).cuda()
				act = zeros(self.p['b'], self.p['n_steps'], self.p['n_actions'], self.p['action_dim']).cuda()

				data[0] = data[0].cuda()

				for t in range(self.p['n_steps']):
					
					lr = np.random.choice(self.p['action_dim'])
					ud = np.random.choice(self.p['action_dim'])
					act[:,t, 0, lr] = 1
					act[:,t, 1, ud] = 1
					z[:,t] = obs_model(data[0][:,t], actions=act[:,t])

				latent_data.append(z)
				action_data.append(act)				
				cuda.empty_cache()	

			z_tensor   = cat(latent_data,  dim=0).cuda()
			act_tensor = cat(action_data, dim=0)

			dataset = TensorDataset(*[z_tensor, act_tensor])

		return dutils.train_val_split(self.p, dataset)
		
	def _load_obs_model(self):
		with no_grad():
			obs_model = ObservationModel(self.p).cuda()
			obs_model.load_state_dict(load(self.obs_model_path+'.pth')['state_dict'])
			obs_model.eval()
		return obs_model
			
	def _prep_saccade_model(self):
		
		# load vision model 
		self.p['foveate'] = True
		obs_model  = self._load_obs_model()
		obs_params = obs_model.named_parameters()
		obs_model.enc_mode = True
		self.train_loader, self.test_loader = self.prep_latent_dataloaders(obs_model)
		del obs_model 
		cuda.empty_cache()			
		
		# initialise transition model 
		self.p['use_lstm'] = True ; 
		rnn = TransitionModel(self.p)
		rnn_params = dict(rnn.state_dict())
		
		# copy and freeze modules from vision model
		for name_v, param_v in obs_params:
			if name_v in rnn_params:
				rnn_params[name_v].data.copy_(param_v.data)
				rnn_params[name_v].requires_grad = False
		
		rnn.load_state_dict(rnn_params)
		
		self.model = rnn
				

		self.model.p['datasize']  = len(self.train_loader.dataset)
		self.model.p['n_batches'] = len(self.train_loader)
		self.model.p['n_iter']    = self.p['datasize'] * self.p['e']

	def train(self):
		
		self.p['foveate'] = True
		self._prep_saccade_model()
		self.model.train() 
		self.model.p = tutils.set_paths(self.p, 'saccade_model')
		
		self.optimizer = Adam(self.model.parameters(), lr=self.model.p['lr'], weight_decay=1e-5)

		self.logger.info('\n Training Saccade Model \n ')
		self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))


		#while self.iteration < self.model.p['n_iter']:
		for e in range(self.p['e']):
			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.train_epoch()
		self.plot_loss()

class TransitionTrainer(Trainer):	
	""" Inherits from Trainer class,
		trains dynamic models"""

	def __init__(self, p, dataloader):
		super(TransitionTrainer, self).__init__(p, dataloader)
		
		self.obs_model_path = join(p['model_dir'], p['model_name'])
		self.obs_model  = None	
	
	def prep_latent_dataloaders(self, obs_model):
		with no_grad():
			latent_data  = []
			action_data  = []
			_actshape = (self.p['b'], self.p['n_steps'], self.p['n_actions'], self.p['action_dim'])
						
			for b, data in enumerate(self.train_loader):
				
				z   = zeros(self.p['b'], self.p['n_steps'], self.p['z_dim'][0])
				act = zeros(*_actshape) 
				
				data[0] = data[0].cuda()
				data[1] = data[1].cuda()
				for t in range(self.p['n_steps']):

					z[:,t] = obs_model(data[0][:,t].squeeze(1))
					act[:,t] = data[1][:,t]

				latent_data.append(z)
				action_data.append(act)		
		
			# roll images into future
			z_tensor   = cat(latent_data,  dim=0).cuda()
			z_tensor   = dutils.roll(z_tensor, 1, -1, fill_pad=0).unsqueeze(-2)
			act_tensor = cat(action_data, dim=0)

			dataset = TensorDataset(*[z_tensor, act_tensor])
		
		return dutils.train_val_split(self.p,dataset)
			
	def _load_obs_model(self):
		with no_grad():
			obs_model = ObservationModel(self.p).cuda()
			obs_model.load_state_dict(load(self.obs_model_path+'.pth')['state_dict'])
			obs_model.eval()
		return obs_model
			
	def _prep_transition_model(self):
		
		# load vision model 
		obs_model  = self._load_obs_model()
		obs_params = obs_model.named_parameters()
		obs_model.enc_mode = True
		_train, _test = self.prep_latent_dataloaders(obs_model)
		self.train_loader = _train
		self.test_loader  = _test
		# initialise transition model 
		self.p['use_lstm'] = True ; 
		rnn = TransitionModel(self.p)
		rnn_params = dict(rnn.state_dict())
		
		# copy and freeze modules from vision model
		for name_v, param_v in obs_params:
			if name_v in rnn_params:
				rnn_params[name_v].data.copy_(param_v.data)
				rnn_params[name_v].requires_grad = False
		
		rnn.load_state_dict(rnn_params)
		
		self.model = rnn

		self.model.p['datasize']  = len(self.train_loader.dataset)
		self.model.p['n_batches'] = len(self.train_loader)
		self.model.p['n_iter']    = self.p['datasize'] * self.p['e']

		cuda.empty_cache()			

	def train(self):

		self._prep_transition_model()
		self.model.train() 
		self.model.p = tutils.set_paths(self.p, 'trans_model')
		
		self.optimizer = Adam(self.model.parameters(), lr=self.model.p['lr'], weight_decay=1e-5)

		self.logger.info('\n Training Transition Model \n ')
		self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))
				
		
		#while self.iteration < self.model.p['n_iter']:
		#while self.iteration < 1:
		for e in range(self.p['e']):
			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.train_epoch()
			with no_grad():
				self.eval_batch()
				
class HierarchicalPredNetTrainer(object):	
	""" Inherits from Trainer class,
		supports hierarchical predictive coding,
		hardcoded for Cardiff dataset for now """
		
	def __init__(self, p, model, data_path):
		
		self.p = p
		self.model = model
		self.data_path = data_path
			
		self.iteration = 0 
		self.plot_iter = 0
		self.logger = getLogger('train')
		self.methandle = tutils.MetricsHandler() 

		self.opt = Adam(self.model.parameters(), lr=p['lr'])
		# 0 != a layer index. listing precludes a_net params from prednet graph
		
		for image_data in self.test_loader:
			if self.p['gpu']:
				image_data = image_data[0].cuda()
			else:
				image_data = image_data[0]
			plot_vars = self.model(self.iteration, image_data, eval=True)
			break 

		if (e % self.p['plot_iter'] == 0) or force_write:
			
			self.model.plot(self.iteration, image_data, plot_vars, self.vis)
			self.plot_iter += 1		
		
	def train(self):
		
		self.model.train() 

		self.logger.info('\n Training Observation Model \n ')
		self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))

		for e in range(self.p['e']):

			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.model.train()			
			self.train_epoch()
			with no_grad():
				self.model.eval()
				self.eval_batch()
		self.eval_batch(force_write=True)
		self.plot_loss()
	
	def train_epoch(self):
		""" All items from dataloader are passed to forward.
			Handle input data in the model's forward function, 
			or overwrite this function in a subordinate class. """
		epoch_loss 	    = 0

		all_files = glob.glob(os.path.join(self.data_path,'*.pth')) 
				
		for file in files:


			dataset = CardiffDataset(file, self.p['dynamic'], self.p['chans'])
			train_loader = DataLoader(dataset, batch_size=p['b'], num_workers=0)
		
			for x in train_loader:
				
				self.model.reset()

				if self.p['gpu']:					
					x = x.cuda()

				for t in range(x.shape[1]):
					
					if t < self.p['n_context']:
						if t > 0:
							err, zp, hs = self.model.context(x[:,t], None)
						else:
							err, zp, hs = self.model.context(err,    hs)
					else:
						
						err, zp, hs = self.model(err, hs)

					self.model.opt.zero_grad()			
					loss, metrics = self.model.loss(self.iteration, err, zp)
					loss.backward()
					self.opt.step()				
					self.iteration += 1
					epoch_loss += loss.item()
					print(t)
				

		self.logger.info('\n Epoch Loss  {}'.format(epoch(loss)))
				
	def eval_batch(self, force_write=None):
		
		self.model.reset()
		tutils.save_checkpoint({'model': self.model, 
								'state_dict': self.model.state_dict(),
								'args': self.model.p}, 
								 self.model.p['model_dir'],  
								 self.model.p['model_name'], 0)	

	def vis(self, e):
		mutils.visualise(self.model.p, self.model, e, self.test_loader)

	def plot_loss(self):
		
		err_loc = join(self.p['metrics_dir'], 'err_loss.png')
		plt.plot(self.methandle.metrics['loss'])
		plt.savefig(err_loc)   # save the figure to file
		plt.close()

		cat_loc = join(self.p['metrics_dir'], 'cat_loss.png')
		plt.plot(self.methandle.metrics['disc_kl'])
		plt.savefig(cat_loc)   # save the figure to file
		plt.close()

		kl_loc = join(self.p['metrics_dir'], 'cont_loss.png')
		plt.plot(self.methandle.metrics['cont_kl'])
		plt.savefig(kl_loc)   # save the figure to file
		plt.close()

class InteractiveTrainer(object):	
	""" Inherits from Trainer class,
		supports interaction with the 
		AnimalAI Unity environment  """

	def __init__(self, p, model):
		
		self.p = p
		self.model = model

		logger = getLogger('train')	

		env_name = 'env/AnimalAI'
		self.env = UnityEnvironment( n_arenas=p['b'],
								file_name=env_name)

		default_brain = self.env.brain_names[0]
		self.brain = self.env.brains[default_brain]
				
		self.iteration = 0 
		self.plot_iter = 0
		self.logger = getLogger('train')
		self.methandle = tutils.MetricsHandler() 

		self.vae_opt = Adam(self.model.parameters(), lr=p['lr'])
		self.rnn_opt = Adam(self.model.lstm[0].parameters(), lr=p['lr'])
		self.act_opt   = Adam(self.model.a_net[0].parameters(), lr=p['lr'])

		
	def train(self):
		
		self.model.train() 

		self.model.p = tutils.set_paths(self.p, 'interactive_model')

		self.logger.info('\n Training Observation Model \n ')
		self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))

		for e in range(self.p['e']):
			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.model.train()
			self.model.lstm[0].train()
			self.model.a_net[0].train()
			
			self.train_epoch()
			print(self.iteration)
			with no_grad():
				self.model.eval()
				self.eval_batch()
		self.eval_batch(force_write=True)
		self.plot_loss()
		
	
	def train_epoch(self):
		""" All items from dataloader are passed to forward.
			Handle input data in the model's forward function, 
			or overwrite this function in a subordinate class. """

		epoch_loss 	    = 0
		epoch_reward    = 0
		epoch_anet_loss = 0
		epoch_rnn_loss = 0
		epoch_vae_loss = 0

		all_files = glob.glob("examples/configs/env_configs/*.yaml") # list of all .yaml files in a directory 
		files = []
		for file in all_files:
			#if any(x in file for x in ['1', '2', '4']):
			if any(x in file for x in ['1']):
				files.append(file)
				
		for file in files:

			config = ArenaConfig(file)
			
			obs = self.env.reset(arenas_configurations=config, train_mode=True)['Learner']
			action = {}
				
			init_action = Variable(zeros(self.p['n_actions']*self.p['b'],1), requires_grad=True)
			init_action = init_action.cuda() if self.p['gpu'] else init_action 
			action['Learner'] = init_action
			done = False
			
			self.model.reset()

			self.model.a_net[0].policy_history = Variable(zeros(200, self.p['b'], 2))
			
			i = 0
			while  i < 200:

				self.rnn_opt.zero_grad()	
				info 	 =  self.env.step(vector_action=action)['Learner']
				
				vis_obs  = Variable(FloatTensor(info.visual_observations[0]), requires_grad=True).cuda().permute(0,-1,1,2)

				vel_obs  = tensor(info.vector_observations).cuda()
				text_obs = info.text_observations
				reward 	 = info.rewards		
				epoch_reward += np.mean(reward)
				self.model.a_net[0].reward_episode.append(reward)
				
				self.model.zero_grad()
				actions, pred, z_pc, z_pd, z_real, z_pred = self.model(self.iteration, vis_obs, Variable(tensor(action['Learner']), requires_grad=True).cuda())
				rnn_target = z_real
				acts = argmax(actions.detach(), dim=-1)
				
				self.model.a_net[0].policy_history[i,:] = acts
				
				action['Learner'] = acts.view(self.p['n_actions']*self.p['b'],1)
				
				#if self.model.training and (self.iteration > 10000):
					# learn lstm
				if i > 0:
					rnn_loss = self.model.lstm[0].loss(rnn_input,rnn_target)
					rnn_loss.backward(retain_graph=True)
					self.rnn_opt.step()		
					epoch_rnn_loss += rnn_loss.item()
				
				rnn_input  = z_pred
				# learn vae
				self.vae_opt.zero_grad()			
				vae_loss, metrics = self.model.loss(self.iteration, pred, vis_obs, z_pc, z_pd)
				vae_loss.backward()
				self.vae_opt.step()		
				epoch_vae_loss += vae_loss.item()
				z_pred = z_pred#.detach()
				#del rnn_loss
				#del z_real, rnn_target, rnn_input, pred, vis_obs, z_pc, z_pd, actions, vae_loss, metrics
	
				self.iteration += 1
				i += 1
				self.model.reset()
				#if any(info.local_done):
				##	print('break at {}'.format(i))
				#	self.model.a_net[0].policy_history = self.model.a_net[0].policy_history[:i,:]
				#	break
				#self.model.a_net[0].policy_history = self.model.a_net[0].policy_history[:i,:]			
				
				
			# learn controller
			#if self.model.training and (self.iteration > 1):
			self.act_opt.zero_grad()
			a_loss = self.model.a_net[0].loss(i)
			a_loss.backward()
			self.act_opt.step()	
			epoch_anet_loss += a_loss.item()
			self.model.a_net[0].loss_history.append(a_loss.item())
			self.model.a_net[0].policy_history = Variable(Tensor())
			self.model.a_net[0].reward_episode = []				
			
			# reset lstm
			self.model.lstm[0].reset()

		epoch_vae_loss  = epoch_vae_loss / 1000 / 4
		epoch_anet_loss = epoch_anet_loss / 1000 / 4
		epoch_rnn_loss  = epoch_rnn_loss / 1000 / 4

		self.logger.info('\n Mean Epoch VAE Loss  {}'.format(epoch_vae_loss))
		self.logger.info('Mean Epoch RNN Loss  {}'.format(epoch_rnn_loss))
		self.logger.info('Mean Epoch Action Loss  {}'.format(epoch_anet_loss))
		self.logger.info('Mean Epoch Reward  {} \n'.format(epoch_reward))
		save_image(pred,'{}_pred_{}.png'.format(self.p['model_name'],self.iteration))
				
	def eval_batch(self, force_write=None):
		
		self.model.reset()
		tutils.save_checkpoint({'model': self.model, 
								'state_dict': self.model.state_dict(),
								'args': self.model.p}, 
								 self.model.p['model_dir'],  
								 self.model.p['model_name'], 0)	

		tutils.save_checkpoint({'model': self.model.lstm[0], 
								'state_dict': self.model.lstm[0].state_dict(),
								'args': self.model.p}, 
								 self.model.p['model_dir'],  
								 'lstm_'+self.model.p['model_name'], 0)	
		
		tutils.save_checkpoint({'model': self.model.a_net[0], 
								'state_dict': self.model.a_net[0].state_dict(),
								'args': self.model.p}, 
								 self.model.p['model_dir'],  
								 'anet_'+self.model.p['model_name'], 0)	
		
			
	def vis(self, e):
		mutils.visualise(self.model.p, self.model, e, self.test_loader)

	def plot_loss(self):
		
		err_loc = join(self.p['metrics_dir'], 'err_loss.png')
		plt.plot(self.methandle.metrics['loss'])
		plt.savefig(err_loc)   # save the figure to file
		plt.close()

		cat_loc = join(self.p['metrics_dir'], 'cat_loss.png')
		plt.plot(self.methandle.metrics['disc_kl'])
		plt.savefig(cat_loc)   # save the figure to file
		plt.close()

		kl_loc = join(self.p['metrics_dir'], 'cont_loss.png')
		plt.plot(self.methandle.metrics['cont_kl'])
		plt.savefig(kl_loc)   # save the figure to file
		plt.close()
			