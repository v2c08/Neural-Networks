from os.path import join  
import yaml 
import numpy as np

from utils import train_utils as tutils 
from utils import model_utils as mutils 
from utils import data_utils  as dutils

from model import *
from trainer import * 
 
from pprint import pprint, pformat
from logging import shutdown
from torch.nn import MSELoss, Module, CrossEntropyLoss
from utils import elbo_decomposition 
from torch import cuda, no_grad, isnan, load
from torch.autograd import Variable, set_detect_anomaly, detect_anomaly
from argparse import ArgumentParser
import glob 

def main(args):	
	
	largs = yaml.load(open(args.config), Loader=yaml.SafeLoader)	
	logger, logf = tutils.logs(largs['exp_name'])
	logger.info(args)
	
	argv = tutils.product_dict(largs)
	nrun = len([x for x in tutils.product_dict(largs)])
	f = 0 ; flist = []

	for perm, mvars in enumerate(argv):
		try:
			logger.info('Training')
			logger.info('Reading permutation : {} of {}'.format(perm, nrun))

			# Initialise parameters
			mvars = tutils.arg_check(mvars, perm)
			
			# Initialise Model
			pprint(mvars)
			
			mfile = os.path.join('exps', mvars['exp_name'], 'models', 'obs_model', mvars['model_name']+'.pth')
			if not mvars['overwrite'] and os.path.isfile(mfile):
				obs_model = []
				print('model {} already exists'.format(mvars['model_name']))
				0/0
				
			if mvars['interactive']:
				model = PrednetWorldModel(mvars)
				trainer = InteractiveTrainer(mvars, model)
				trainer.train()
				del trainer, model
					
			else:
				
				if mvars['arch'] == 'resnet':
					obs_model = ResNet(mvars)

				elif mvars['arch'] == 'HVRNN':
					if mvars['prednet']:
						obs_model = HVRNNPrednet(mvars)
					else:
						if mvars['spatial']:
							obs_model = HVRNNSpatial(mvars)
						else:
							obs_model = HVRNN(mvars)

				elif mvars['arch'] == 'VRNN':
					obs_model = VRNN(mvars)
					
				elif mvars['arch'] == 'VAE':
					obs_model = ObservationVAE(mvars)
					
				else:
					print('No Such Architecture')
					0/0
				
				if mvars['gpu']:
					obs_model = obs_model.cuda()
				
				# Initialise Dataloader(s)
				if largs['train_gx']:
					logger.info('Training Observation Model')
					data = dutils.get_dataset(mvars, split='train',static= not mvars['dynamic'])
					gx_trainer  = ObservationTrainer(mvars, data, obs_model)
					gx_trainer.train()
					del gx_trainer, obs_model, data
					

				if largs['train_fx']:
					logger.info('Training Transition Model')
					data = dutils.get_dataset(mvars, split='train', static=True)
					if mvars['dataset'] == 'mnist':
						fx_trainer  = SaccadeTrainer(mvars, data)
					else:
						fx_trainer = TransitionTrainer(mvars, data)
					fx_trainer.train()
					del fx_trainer
						
		except Exception as e:     # most generic exception you can catch
			
			logf.error('Error - config {}'.format(perm))
			logger.exception(e)
			f += 1 ; flist.append(perm)
			torch.cuda.empty_cache()
			del obs_model 

	logf.info('\n Total Evaluation Failures : {} / {} '.format(f, perm+1))	
	logf.info(pprint(flist))
	shutdown() # close logs

if __name__ == '__main__':
	ap = ArgumentParser()
	ap.add_argument("-m", '--config', required=False, default="parameters.yaml", help="Path to model config", dest='config')
	main(ap.parse_args())