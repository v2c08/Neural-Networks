import os
import math
from torch import nn, max
import logging
from torchvision.utils import save_image
import numpy as np
import numpy as np
from itertools import product
from collections.abc import Iterable
import torch


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


class MetricsHandler(object):
	def __init__(self):
		self.metrics = {'iter':[],'recon':[], 'cont_kl':[]}
		
	def extend(self, new_metrics):
		for key, value in self.metrics.items():
			value.append(new_metrics[key])
	
	def _get(self):
		return self.metrics

def save_checkpoint(state, save, exp_name, iter):
	if not os.path.exists(save):
		os.makedirs(save)
	#filename = os.path.join(save, '{}_%04d.pth'.format(exp_name) % iter)
	filename = os.path.join(save, '{}.pth'.format(exp_name))
	
	torch.save(state, filename)
	
def group(number):
	s = '%d' % number
	groups = []
	while s and s[-1].isdigit():
		groups.append(s[-3:])
		s = s[:-3]
	return s + ','.join(reversed(groups))

def logs(exp_dir):

	exp_dir = os.path.join('exps', exp_dir[0], 'logs')
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
	datefmt='%Y-%m-%d:%H:%M:%S',level=logging.DEBUG)	
	logger = logging.getLogger('model')
	handler = logging.FileHandler(os.path.join(exp_dir, 'debug.log'))
	logger.addHandler(handler)
	logf = open(os.path.join(exp_dir, 'summary.log'), "w+")
	logf = logging.getLogger('errlog')
	errhandler = logging.FileHandler(os.path.join(exp_dir, 'err.log'))
	logf.addHandler(errhandler)		
	
	return logger, logf

def arg_check(p, iter=None):
	

	#p['z_con_capacity'] = p['z_dis_capacity']
	p['beta'] = p['z_con_capacity'][1]

	if p['dataset']=='animalai':
		p['imdim'] = (3, 84, 84)
	
	elif p['dataset']=='moving_mnist':
		p['imdim'] = (1,32,32)

	elif p['dataset'] in ['mnist', 'emnist','RFMnist', 'mnist_spatial_z']:
		p['imdim'] = (1, 32, 32)

	elif p['dataset'] == 'mnist_sequences':
		# not correct when sequences also true
		p['t'] = 4
		p['imdim'] = (1,28,28)

	elif p['dataset'] in ['lsun_bedroom', 'celeba']:
		p['imdim'] = (3, 64, 64)
	
	elif p['dataset'] == 'stl10':
		p['imdim'] = (3, 96, 96)

	elif p['dataset'] in ['dsprites', 'freyfaces', '3dfaces']:
		p['imdim'] = (1, 64, 64)
		
	elif p['dataset'] == 'fec':		
		p['imdim'] = (3, 224, 224)

	elif p['dataset']  == 'cardiff':		
		p['imdim'] = (p['chans'], 256, 256)
	elif p['dataset'] == 'car_racing':
		p['imdim'] = (3,64,64)

	else:
		print('no such datset : {}'.format(p['dataset']))
		
	constr = 'c'+'_'.join(map(str, [round(x) for x in p['z_con_capacity']]))
	
	p['model_name'] = '{}_z{}_{}'.format(p['dataset'], p['z_dim'],  constr)

	return p

def set_paths(p, modestr):

	p['save'] = os.path.join('exps', p['exp_name'])
	
	p['metrics_dir'] = os.path.join(p['save'], 'metrics', modestr)
	os.makedirs(p['metrics_dir']) if not os.path.exists(p['metrics_dir']) else None
	
	p['model_dir']	 = os.path.join(p['save'], 'models', modestr)
	os.makedirs(p['model_dir']) if not os.path.exists(p['model_dir']) else None
	
	p['plot_dir']	 = os.path.join(p['save'], 'plots', modestr)
	os.makedirs(p['plot_dir']) if not os.path.exists(p['plot_dir']) else None
	
	return p

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("tensor gcdump" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass		
	print("Total size:", total_size)

def product_dict(args):
	keys = args.keys()
	vals = args.values()

	vals = [[x] if not isinstance(x,Iterable) else x for x in vals]
	
	false_case = 0
	for instance in product(*vals):
		
		arg = dict(zip(keys, instance))

		yield arg

