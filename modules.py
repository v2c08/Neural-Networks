import math 
from utils import model_utils as mutils
import numpy as np
from torch.nn import functional as F
from torch.nn import ModuleList, GroupNorm, ReLU, MaxPool2d, Sigmoid
from torch.nn import Conv2d, ConvTranspose2d, Softmax, Sequential, BatchNorm1d
from torch.nn import Linear, Module, LSTM, Parameter, BatchNorm2d
from torch.autograd import Variable, Function
from torch import zeros,zeros_like, ones_like, cat, ByteTensor, FloatTensor, rand, log, sigmoid
from torch import add,tanh,squeeze,Tensor,float,stack, argmax
from torch import max as torchmax
from torch import min as torchmin
import torchvision.models as models
import torch 
from torch.nn.modules.utils import _pair

class CelebConvEncoder(Module):
	def __init__(self, p, l):
		super(CelebConvEncoder, self).__init__()

		self.conv1 = Conv2d(3,	32, (4,4), stride=2, padding=1)
		self.conv2 = Conv2d(32, 64, (4,4), stride=2, padding=1)
		self.conv3 = Conv2d(64, 128, (4,4), stride=2, padding=1)
		self.conv4 = Conv2d(128, 128, (4,4), stride=2, padding=1)
		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l][0] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim']) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(128 * 4 * 4, enc_h) 

		
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 

		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		h = F.relu(self.fc1(x.view(-1, 128*4*4)))

		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class SpatialEncoder(Module):
	def __init__(self, p, l):
		super(SpatialEncoder, self).__init__()

		self.conv1 = Conv2d(1,	32, (4,4), stride=2, padding=1)
		self.conv2 = Conv2d(32, 64, (4,4), stride=2, padding=1)
		self.conv3 = Conv2d(64, 64, (4,4), stride=2, padding=1)
		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] > 0 
		self.has_dis = p['nz_dis'][l][0] > 0
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['imdim']) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(64 * 4 * 4, enc_h) 
				
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))

		h = F.relu(self.fc1(x.view(self.bs, -1)))

		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class MNISTConvEncoder(Module):
	def __init__(self, p, l):
		super(MNISTConvEncoder, self).__init__()

		self.conv1 = Conv2d(1,	32, (4,4), stride=2, padding=1)
		self.conv2 = Conv2d(32, 64, (4,4), stride=2, padding=1)
		self.conv3 = Conv2d(64, 64, (4,4), stride=2, padding=1)
		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] > 0 
		self.has_dis = p['nz_dis'][l][0] > 0
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['imdim']) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(64 * 4 * 4, enc_h) 
				
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))

		h = F.relu(self.fc1(x.view(-1, 1024)))

		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class ConvLSTMCell(Module):
	"""
	Generate a convolutional LSTM cell
	"""

	def __init__(self, input_size, hidden_size, kernel_size=3, stride=1,padding=1, transpose=False):
		super(ConvLSTMCell, self).__init__()
		
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.kernel_size = kernel_size
		self.padding = padding
		if not transpose:
			self.Gates = Conv2d(input_size + hidden_size, 4*hidden_size, kernel_size, stride=1, padding=1)
		else:
			self.Gates = ConvTranspose2d(input_size + hidden_size, 4*hidden_size, kernel_size, stride=1, padding=1)		

	def forward(self, input_, prev_state):

		# get batch and spatial sizes
		batch_size = input_.data.size()[0]
		spatial_size = input_.data.size()[2:]
		# generate empty prev_state, if None is provided
		print(prev_state)
		print(prev_state is None)
		if prev_state is None:
			state_size = [batch_size, self.hidden_size] + list(spatial_size)
			prev_state = (
				Variable(torch.zeros(state_size)),
				Variable(torch.zeros(state_size))
			)
		
		prev_state = [x.cuda() if input_.is_cuda else x for x in prev_state] 
		prev_hidden, prev_cell = prev_state

		# data size is [batch, channel, height, width]
		stacked_inputs = torch.cat((input_, prev_hidden), 1)
		gates = self.Gates(stacked_inputs)

		# chunk across channel dimension
		in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

		# apply sigmoid non linearity
		in_gate = sigmoid(in_gate)
		remember_gate = sigmoid(remember_gate)
		out_gate = sigmoid(out_gate)

		# apply tanh non linearity
		cell_gate = tanh(cell_gate)
		
		# compute current cell and hidden state
		cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
		hidden = out_gate * tanh(cell)

		return hidden, cell
		
class ResidualBlock(Module):
	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(ResidualBlock, self).__init__()
		
		def conv3x3(in_channels, out_channels, stride=1):
			return Conv2d(in_channels, out_channels, kernel_size=3, 
							 stride=stride, padding=1, bias=False)
		
		
		self.conv1 = conv3x3(in_channels, out_channels, stride)
		self.bn1 = BatchNorm2d(out_channels)
		self.relu = ReLU(inplace=True)
		self.conv2 = conv3x3(out_channels, out_channels)
		self.bn2 = BatchNorm2d(out_channels)
		
		if (stride != 1) or (in_channels != out_channels):
		
			self.downsample = conv3x3(in_channels, out_channels, stride=stride)
			self.downsample_bn = BatchNorm2d(out_channels)
		else:
			self.downsample = None
		
	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample:
			residual = self.downsample(x)
			residual = self.downsample_bn(residual)
		out += residual
		out = self.relu(out)
		return out

class HVRNNPrior(Module):
	def __init__(self, p, l):
		super(HVRNNPrior, self).__init__()
		
		# Add GroupNorms after LSTMS
		self.l = l
		self.dbgflag = 'Prior'
		
		if l == 2:
			#1x1
			self.conv1	  = Conv2d(512, 512, 1, 1, 0)
			self.gnrm1	  = GroupNorm(1, 512)
			
			self.convlstm = ConvLSTMCell(512, 512)
			
			self.conv2	= Conv2d(512, 512*2, 1, 1, 0)
			self.gnrm2	= GroupNorm(1, 512*2)
			
		if l == 1:
			#8x8
			self.conv1	  = Conv2d(512, 512, 1, 1, 0)
			self.gnrm1	  = GroupNorm(16, 512)
			self.convlstm = ConvLSTMCell(512, 512)
			# 512 mu, 512 sig
			self.conv2	  = Conv2d(512, 512*2, 1, 1, 0)
			self.gnrm2	  = GroupNorm(16, 512*2)

		if l == 0:
			# 32x32
			self.conv1	  = Conv2d(128, 128, 1, 1, 0)
			self.gnrm1	  = GroupNorm(16, 128)
			self.convlstm = ConvLSTMCell(128, 128)
			
			# 128 mu, 128, sig
			self.conv2	  = Conv2d(128, 128*2, 1, 1, 0)
			self.gnrm2	  = GroupNorm(16, 128*2)
	
	def forward(self, hs, td, bu):
		
		if not td is None:
			x = cat([td, bu], dim=1)
		else:
			x = bu

		if not hs is None:
			if isinstance(hs, tuple) or isinstance(hs, list):
				h,c = hs
			else:
				h,c = hs.split(hs.shape[1]//2,dim=1)		
		
		x = self.conv1(x)
		x = self.gnrm1(x)
		h, c = self.convlstm(x, (h,c))
		x = self.conv2(h)
		z_p = self.gnrm2(x)
			
		return z_p, h, c
	
class HVRNNPosterior(HVRNNPrior):
		def __init__(self, p, l):
			super(HVRNNPosterior, self).__init__(p,l)
			self.dbgflag = 'Posterior'

class HVRNNSpatialPrior(Module):
	def __init__(self, p, l):
		super(HVRNNSpatialPrior, self).__init__()
		
		# Add GroupNorms after LSTMS
		self.l = l
		self.dbgflag = 'Prior'
		
		if l == 2:
			#1x1
			self.conv1	  = Conv2d(512, 128, 1, 1, 0)
			self.gnrm1	= GroupNorm(1, 128)
			self.convlstm = ConvLSTMCell(128, 1)
			self.conv2	= Conv2d(1, 1*2, 1, 1, 0)
			self.gnrm2	= GroupNorm(1, 1*2)
			
		if l == 1:
			#8x8
			self.conv1	  = Conv2d(1, 1, 1, 1, 0)
			self.gnrm1	  = GroupNorm(16, 512)
			self.convlstm = ConvLSTMCell(1, 1)
			# 512 mu, 512 sig
			self.conv2	  = Conv2d(1, 1*2, 1, 1, 0)
			self.gnrm2	  = GroupNorm(1, 1*2)

		if l == 0:
			# 32x32
			self.conv1	  = Conv2d(1, 1, 1, 1, 0)
			self.gnrm1	  = GroupNorm(16, 1)
			self.convlstm = ConvLSTMCell(1, 1)
			
			# 128 mu, 128, sig
			self.conv2	  = Conv2d(1, 1*2, 1, 1, 0)
			self.gnrm2	  = GroupNorm(16, 1*2)
	
	def forward(self, hs, td, bu):
		
		if not td is None:
			x = cat([td, bu], dim=1)
		else:
			x = bu

		if not hs is None:
			if isinstance(hs, tuple) or isinstance(hs, list):
				h,c = hs
			else:
				hs = hs.split(hs.shape[1]//2,dim=1)		
		else:
			hs = None
		print(x.shape)
		x = self.conv1(x)
		x = self.gnrm1(x)
		h, c = self.convlstm(x, hs)
		x = self.conv2(h)
		z_p = self.gnrm2(x)
			
		return z_p, h, c
	
class HVRNNSpatialPosterior(HVRNNSpatialPrior):
		def __init__(self, p, l):
			super(HVRNNSpatialPosterior, self).__init__(p,l)
			self.dbgflag = 'Posterior'

class SpatialDecoder(Module):
	
	def __init__(self, p, l):
		super(SpatialDecoder, self).__init__()

		# layer configuration 
#		latents = 10 + (7 * 7) 
		hidden	  = p['enc_h'][l] 
		latents = sum(p['z_dim'][l:l+2]) 
		
		im_size = 7
		
		x = torch.linspace(-1,1,32)
		y = torch.linspace(-1,1,32)
		# (7,7)
		x_grid, y_grid = torch.meshgrid(x,y)
		# (1,1,7,7)
		self.register_buffer('x_grid', x_grid.view((1,1)+x_grid.shape))
		self.register_buffer('y_grid', y_grid.view((1,1)+y_grid.shape))
		
		# cont + disc + 2 
		self.dcnv1 = ConvTranspose2d(22, 64, 3, padding=1)
		self.dcnv2 = ConvTranspose2d(64, 1,  3, padding=1)	
		
	def forward(self, z):
		
		batch_size = z.size(0)
		z = z.view(z.shape+(1,1))
		z = z.expand(-1,-1,32,32) 
		
		x = torch.cat((self.x_grid.expand(batch_size,-1,-1,-1),
					   self.y_grid.expand(batch_size,-1,-1,-1), z),dim=1)
		x =  F.relu(self.dcnv1(x))
		x = sigmoid(self.dcnv2(x)) 
		return x
		
class HVRNNSpatialDecoder(Module):
	def __init__(self, p, l):
		super(HVRNNSpatialDecoder, self).__init__()

		# remember order of latents needs to be reversed here
		self.l = l
		
		# Add GroupNorms after LSTMs 
		
		if self.l == 2:
					
			self.lstm1 = ConvLSTMCell(1, 32) 
			self.dcnv1 = ConvTranspose2d(32, 128, 4) # 4x4
			#self.dcnvn = ConvTranspose2d(512, 512, 4, 2, 1) # 4x4
			
			self.lstm2 = ConvLSTMCell(128,256)
			self.dcnv2 = ConvTranspose2d(256, 512, 4, 2, 1) # 8 x 8

		
		elif self.l == 1:
			
			x = torch.linspace(-1,1,8)
			y = torch.linspace(-1,1,8)
			x_grid, y_grid = torch.meshgrid(x,y)

			self.register_buffer('x_grid', x_grid.view((1,1)+x_grid.shape))
			self.register_buffer('y_grid', y_grid.view((1,1)+y_grid.shape))
			
			
			self.lstm1 = ConvLSTMCell(1,32)
			self.dcnv1 = ConvTranspose2d(32, 64, 4, 2, 1) # 16 x 16 
			
			self.lstm2 = ConvLSTMCell(64, 128)
			self.dcnv2 = ConvTranspose2d(128, 128, 4, 2, 1) # 32 x 32
		
		elif self.l == 0: 

			x = torch.linspace(-1,1,32)
			y = torch.linspace(-1,1,32)
			x_grid, y_grid = torch.meshgrid(x,y)

			self.register_buffer('x_grid', x_grid.view((1,1)+x_grid.shape))
			self.register_buffer('y_grid', y_grid.view((1,1)+y_grid.shape))


			self.lstm1 = ConvLSTMCell(128, 128)
			self.dcnv1 = ConvTranspose2d(128, 64, 4, 2, 1) # 64 x 64 
			
			self.lstm2	= ConvLSTMCell(64, 64)
			self.dcnv2 = ConvTranspose2d(64, 64, 4, 2, 1)
			
			self.dcnv3	= ConvTranspose2d(64, 32, 4, 2, 1)
			self.gnrm1	= GroupNorm(16, 32) # ReLu here
			
			self.dcnv4	= ConvTranspose2d(32, p['chans'], 4, 2, 1)	
				
		else:
			print('Unsupported n layers (Decoder)')
			0/0
		
	def forward(self, z, hs):
		
		batch_size = z.size(0)
		z = z.view(z.shape+(1,1))
		z = z.expand(-1,-1,32,32) 
		
		x = torch.cat((self.x_grid.expand(batch_size,-1,-1,-1),
					   self.y_grid.expand(batch_size,-1,-1,-1), z),dim=1)
		x =  F.relu(self.dcnv1(x))
		x = sigmoid(self.dcnv2(x)) 
		return x
		
class HVRNNDecoder(Module):
	def __init__(self, p, l):
		super(HVRNNDecoder, self).__init__()
		
		# remember order of latents needs to be reversed here
		self.l = l
		
		# Add GroupNorms after LSTMs 
		
		if self.l == 2:
			
			self.lstm1 = ConvLSTMCell(512, 512) 
			self.dcnv1 = ConvTranspose2d(512, 512, 4) # 4x4
			#self.dcnvn = ConvTranspose2d(512, 512, 4, 2, 1) # 4x4
			
			self.lstm2 = ConvLSTMCell(512,512)
			self.dcnv2 = ConvTranspose2d(512, 512, 4, 2, 1) # 8 x 8

		
		elif self.l == 1:
			
			self.lstm1 = ConvLSTMCell(512,512)
			self.dcnv1 = ConvTranspose2d(512, 256, 4, 2, 1) # 16 x 16 
			
			self.lstm2 = ConvLSTMCell(256, 256)
			self.dcnv2 = ConvTranspose2d(256, 128, 4, 2, 1) # 32 x 32
			
		
		elif self.l == 0: 

			self.lstm1 = ConvLSTMCell(128, 128)
			self.dcnv1 = ConvTranspose2d(128, 64, 4, 2, 1) # 64 x 64 
			
			self.lstm2	= ConvLSTMCell(64, 64)
			self.dcnv2 = ConvTranspose2d(64, 64, 4, 2, 1)
			
			self.dcnv3	= ConvTranspose2d(64, 32, 4, 2, 1)
			self.gnrm1	= GroupNorm(16, 32) # ReLu here
			
			self.dcnv4	= ConvTranspose2d(32, p['chans'], 4, 2, 1)	
				
		else:
			print('Unsupported n layers (Decoder)')
			0/0
			
	def forward(self, hs, x):

		if hs is None:
			hidden_1 = None
			hidden_2 = None

		elif isinstance(hs, tuple) or isinstance(hs, list):
			h,c = hs
			h1, h2 = h
			c1, c2 = c
			hidden_1 = (h1, c1)
			hidden_2 = (h2, c2)
		else:
			0/0
			h1, c1 = hs[0].split(hs[0].shape[1]//2,dim=1)
			h2, c2 = hs[1].split(hs[1].shape[1]//2,dim=1)
			hidden_1 = (h1, c1)
			hidden_2 = (h2, c2)		

		hidden_1 = self.lstm1(x, hidden_1)
		x = self.dcnv1(hidden_1[0])

		if self.l == 0:
			
			hidden_2 = self.lstm2(x, hidden_2)
			x = F.relu(self.gnrm1(self.dcnv3(x)))
			x = sigmoid(self.dcnv4(x))
		else:
			
			hidden_2= self.lstm2(x, hidden_2)
			
			x = self.dcnv2(hidden_2[0])

		return x, [hidden_1[0], hidden_2[0]], [hidden_1[1], hidden_2[1]]
		
class HVRNNInitStates(Module):
	def __init__(self, p, l):
		super(HVRNNInitStates, self).__init__()
		self.l = l ; self.p = p

		# Need to handle difference between QP/D
		# QP only need every second block

		if l == 2:

			# 1x1
			# Decoder
			self.conv1 = Conv2d(512, 512, 1, 1, 0)
			self.gnrm1 = GroupNorm(16, 512)
			
			# Prior & Posterior 
			self.conv2 = Conv2d(512, 512*2, 1, 1, 0)
			self.gnrm2 = GroupNorm(16, 512*2)
			
			# 4x4
			# Decoder
			self.conv3 = Conv2d(512, 512, 1, 1, 0)
			self.gnrm3 = GroupNorm(1, 512)
			
			# Prior & Posterior
			self.conv4 = Conv2d(512, 512*2, 1, 1, 0)
			self.gnrm4 = GroupNorm(1, 512*2)

		if l == 1:

			# 16x16
			# Decoder
			self.conv1 = Conv2d(256, 256, 1, 1, 0)
			self.gnrm1 = GroupNorm(16, 256)
			# Prior & Posterior 
			self.conv2 = Conv2d(256, 256*2, 1, 1, 0)
			self.gnrm2 = GroupNorm(16, 256*2)
			
			# 8x8
			# Decoder 
			self.conv3 = Conv2d(512, 512, 1, 1, 0)
			self.gnrm3 = GroupNorm(16, 512)
			# Prior & Posterior
			self.conv4 = Conv2d(512, 512*2, 1, 1, 0)
			self.gnrm4 = GroupNorm(16, 512*2)
						
		if l == 0:
			
			# 64x64 
			# Decoder
			self.conv1 = Conv2d(64, 64, 1, 1, 0)
			self.gnrm1 = GroupNorm(16, 64)
			
			# Prior & Posterior
			self.conv2 = Conv2d(64, 64*2, 1, 1, 0)
			self.gnrm2 = GroupNorm(16, 64*2)
			
			# 32x32 
			# Decoder
			self.conv3 = Conv2d(128, 128, 1, 1, 0)
			self.gnrm3 = GroupNorm(16, 128)
			
			# Prior & Posterior 
			self.conv4 = Conv2d(128, 128*2, 1, 1, 0)
			self.gnrm4 = GroupNorm(16, 128*2)

	def forward(self, features):
		
		x1   = F.relu(self.gnrm1(self.conv1(features[0])))
		x1   = self.gnrm2(self.conv2(x1))
		h1, c1 = x1.split(x1.shape[1]//2, dim=1)
		
		x2   = F.relu(self.gnrm3(self.conv3(features[1])))
		x2   = self.gnrm4(self.conv4(x2)) 
		h2, c2 = x2.split(x2.shape[1]//2, dim=1)
		
		return (h1,h2), (c1,c2)
							
class HVRNNEncoder(Module):
	def __init__(self, p, l):
		super(HVRNNEncoder, self).__init__()
		
		self.l = l
		if l == 0:
			# z = 32
			
			self.conv0  = Conv2d(p['chans'],  64, 4, 2, 1)
			#self.conv00 = Conv2d(64, 64, 4, 2, 1)
			#self.conv00 = Conv2d(64, 64, 4, 2, 0)
			#self.res0 = ResidualBlock(6, 64, 1)
			self.max0 = MaxPool2d(2, 2)
			
			# r/g/b/d/of_cam/of_dense
			self.conv1 = Conv2d(64, 64, 3, 1, 1)
			self.res1 = ResidualBlock(64, 64)
			self.max1 = MaxPool2d(2, 2)
			
			self.res2 = ResidualBlock(64, 128)
			self.res3 = ResidualBlock(128, 128)
			self.max2 = MaxPool2d(2, 2)
			
		
		elif l == 1:
			# z = 8
			self.res1 = ResidualBlock(128, 256)
			self.res2 = ResidualBlock(256, 256)
			self.max1 = MaxPool2d(2, 2)
			
			self.res3 = ResidualBlock(256, 512)
			self.res4 = ResidualBlock(512, 512)
			self.max2 = MaxPool2d(2, 2)

			
		elif l == 2:
			# z = 1
			self.res1 = ResidualBlock(512, 512)
			self.res2 = ResidualBlock(512, 512)
			self.max1 = MaxPool2d(2, 2)
			
			self.res3 = ResidualBlock(512, 512)
			self.res4 = ResidualBlock(512, 512)
			self.max2 = MaxPool2d(4, 2)
			
		else:
			print('Unsupported n layers (Encoder)')
			0/0
				
	def forward(self, x, z_q=None):
		
		if self.l == 0:
			#
			#print(x.shape)
			#0/0
			x = self.conv0(x)
			#x = self.max0(x)
			#x = self.conv00(x)
			#x = self.res0(x)
			h1 = self.max0(x)
			
			
			#x = self.conv00(x)
			#x = self.conv01(x)
			x = self.conv1(h1)
			x = self.res1(x)
			x = self.max1(x)
			
			x = self.res2(h1)
			x = self.res3(x)
			h2 = self.max2(x)
		
		elif self.l == 1:
	
			x1 = self.res1(x[-1])
			x1 = self.res2(x1)
			h1 = self.max1(x1)
			
			x2 = self.res3(h1)
			x2 = self.res4(x2)
			h2 = self.max2(x2)			
		
		elif self.l == 2:
		
			x1 = self.res1(x[-1])
			x1 = self.res2(x1)
			h1 = self.max1(x1)
			
			x2 = self.res3(h1)
			x2 = self.res4(x2)
			h2 = self.max2(x2)	
			 
		return (h1, h2)

class STL10ConvEncoder(Module):
	def __init__(self, p, l):
		super(STL10ConvEncoder, self).__init__()	

		self.enc1 = Conv2d(3, 32, 4, stride=2, padding=1)

		self.enc2 = Conv2d(32, 32, 4, stride=2, padding=1)
		self.bn2  = BatchNorm2d(32)

		self.enc3 = Conv2d(32, 64, 4, stride=2, padding=1)
		self.bn3  = BatchNorm2d(64)

		self.enc4 = Conv2d(64, 64, 4, stride=2, padding=1)
		self.bn4  = BatchNorm2d(128)

		self.enc5 = Conv2d(64, 64, 4, stride=2, padding=1)
		self.bn5  = BatchNorm2d(64)

		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim']) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(64 * 3 * 3, enc_h) 

		
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 
		x = F.relu(			self.enc1(x) )
		x = F.relu(self.bn2(self.enc2(x)))
		x = F.relu(self.bn3(self.enc3(x)))
		x = F.relu(self.bn4(self.enc4(x)))
		x = F.relu(self.bn5(self.enc5(x)))
		x = F.relu(self.bn6(self.enc6(x)))
		x = x.view(self.bs, -1)
		h = F.relu(self.fc1(x))
		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class STL10ConvDecoder(Module):
	
	def __init__(self, p, l):
		super(STL10ConvDecoder, self).__init__()

		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		
		self.fc1 = Linear(latents, 64*3*3)
		
		self.dec1 = ConvTranspose2d(64, 64, 4, stride=2, padding=1)
		self.bn1  = BatchNorm2d(256)
		
		self.dec2 = ConvTranspose2d(64, 64, 4, stride=2, padding=1)
		self.bn2  = BatchNorm2d(128)

		self.dec3 = ConvTranspose2d(64, 32, 4, stride=2, padding=1)
		self.bn3  = BatchNorm2d(64)

		self.dec4 = ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.bn4  = BatchNorm2d(32)

		self.dec5 = ConvTranspose2d(32, 3, 4, stride=2, padding=1)
		
		
	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = x.view(-1,64, 3, 3)
		x = F.relu(self.bn1(self.dec1(x)))
		x = F.relu(self.bn2(self.dec2(x)))
		x = F.relu(self.bn3(self.dec3(x)))
		x = F.relu(self.bn4(self.dec4(x)))
		x = sigmoid(		self.dec5(x) )

		return x
	
class Encoder(Module): 
	# Error > latent Encoder 
	def __init__(self, p, l): 
		super(Encoder, self).__init__() 
		 
		# layer specific config 
		#enc_h = p['enc_h'][l]*2 if l < p['layers']-1 else p['enc_h'][l] 
		 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim']) 
		self.constrained = l < p['layers']-1 
		 
		# image to features 
		self.fc1 = Linear(self.imdim, enc_h) 
		#self.fc2 = Linear(enc_h, enc_h) 
		#self.fc3 = Linear(enc_h+p['z_dim'][l+1] if self.constrained else enc_h, enc_h) 

		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
		#self.apply(utils.init_weights) 
			 
	def forward(self, x, z_q=None):			

		latent_dist = {'con':[], 'dis':[]} 
 
		h = F.relu(self.fc1(x.view(-1, self.imdim)))
		#h = F.relu(self.fc2(h))
		if self.constrained: 
			h = cat((h,z_q), dim=-1) 
			h = F.relu(self.fc3(h)) 
		 
		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class ConvEncoder(Module):
	def __init__(self, p, l):
		super(ConvEncoder, self).__init__()
		self.output_dim = sum(p['z_dim'][l:l+2]) * p['z_params']
		self.constrained = l < p['layers']-1
		
		self.has_con = p['nz_con'][l] is not None
		self.has_dis = p['nz_dis'][l] is not None				
		
		self.z_con_dim = 0; self.z_dis_dim = 0;
		if self.has_con:
			self.z_con_dim = p['nz_con'][l] * p['z_params']
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = len(p['nz_dis'][l])
		
		self.z_dim = self.z_con_dim + self.z_dis_dim		

		self.conv1 = Conv2d(p['ldim'][l][0], 32, (4,4), stride=2, padding=1)	# (1) 42 x 42
		self.bn1 = BatchNorm2d(32)
		self.conv2 = Conv2d(32, 32, (4,4), stride=2, padding=1)
		self.bn2 = BatchNorm2d(32)
		self.conv3 = Conv2d(32, 64, (4,4), stride=2, padding=1)
		self.bn3 = BatchNorm2d(64)
		self.conv4 = Conv2d(64, 64, (4,4), stride=2, padding=1)
		self.bn4 = BatchNorm2d(64) 
		self.conv5 = Conv2d(64, 128,  (4,4), stride=2, padding=1)
		self.bn5 = BatchNorm2d(128)
		self.fc1 = Linear(128*2*2, p['enc_h'][l])	
		
		self.fc2 = Linear(p['enc_h'][l], p['enc_h'][l]) 
		
				
		if self.has_con:
			# features to continuous latent 
			self.fc_zp = Linear(p['enc_h'][l], self.z_con_dim)
		if self.has_dis:
			# features to categorical latent
			self.fc_alphas = []
			for a_dim in p['nz_dis'][l]:

				self.fc_alphas.append(Linear(p['enc_h'][l],a_dim))
			self.fc_alphas = ModuleList(self.fc_alphas)		

		# setup the non-linearity
		self.dim = p['ldim'][l]
		self.apply(mutils.init_weights)

	def forward(self, x, z_q=None):
		latent_dist = {'con':[], 'dis':[]}
		
		h = x.view(-1, *self.dim)
		h = F.relu(self.bn1(self.conv1(h)))
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.relu(self.bn4(self.conv4(h)))
		h = F.relu(self.bn5(self.conv5(h)))
		
		h = h.view(x.size(0), -1)
		h = F.relu(self.fc1(h))
		h = F.relu((self.fc2(h)))
		if self.constrained:
			h = cat((h, z_q), dim=-1)
		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))
		

		return latent_dist['con'], latent_dist['dis']

class ConvEncoderAnAI(Module):
	def __init__(self, p):
		super(ConvEncoderAnAI, self).__init__()
		
		self.conv1 = Conv2d(p['imdim'][0], 64, 4, stride=2, padding=1)	# (1) 42 x 42
		#self.conv2 = Conv2d(32, 64, 4, stride=2, padding=1)	# (1) 42 x 42
		#self.conv3 = Conv2d(p['imdim'][0], 64, 4, stride=2, padding=1)	# (1) 42 x 42
		
		self.bn1 = BatchNorm2d(64)
		self.conv2 = Conv2d(64, 128, 4, stride=2, padding=1)
		self.bn2 = BatchNorm2d(128)
		self.conv3 = Conv2d(128, 256, 4, stride=2, padding=1)
		self.bn3 = BatchNorm2d(256)		
		self.conv4 = Conv2d(256, 512, 4, stride=2)
		self.bn4 = BatchNorm2d(512)		
		
		self.fc1 = Linear(512*4*4, 512)	
		
		self.fc2 = Linear(512,256) 
		
		self.fc_zp = Linear(256, p['z_dim'] * p['z_params'])

		# setup the non-linearity
		self.dim = p['imdim']
		self.apply(mutils.init_weights)

	def forward(self, x, z_q=None):
		
		#print(x.shape)
		h = x.view(-1, *self.dim)
		#print(x.shape)
		h = F.relu(self.bn1(self.conv1(h)))
		#print(h.shape)
		h = F.relu(self.bn2(self.conv2(h)))
		#print(h.shape)
		h = F.relu(self.bn3(self.conv3(h)))
		#print(h.shape)
		h = F.relu(self.bn4(self.conv4(h)))
		#print(h.shape)
		h = h.view(h.size(0), -1)
		#print(h.shape)
		h = F.relu(self.fc1(h))
		#print(h.shape)
		h = F.relu((self.fc2(h)))

		return self.fc_zp(h)

class ConvDilationEncoder(Module):
	def __init__(self, p, l):
		super(ConvEncoder, self).__init__()
		self.output_dim = sum(p['z_dim'][l:l+2]) * p['z_params']
		self.constrained = l < p['layers']-1
		
		self.has_con = p['nz_con'][l] is not None
		self.has_dis = p['nz_dis'][l] is not None				
		
		self.z_con_dim = 0; self.z_dis_dim = 0;
		if self.has_con:
			self.z_con_dim = p['nz_con'][l] * p['z_params']
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = len(p['nz_dis'][l])
		
		self.z_dim = self.z_con_dim + self.z_dis_dim		

		self.conv1 = Conv2d(p['ldim'][l][0], 32, 6, 1, dilation=3)	# (1) 42 x 42
		self.bn1 = BatchNorm2d(32)
		self.conv2 = Conv2d(32, 32, 6, 1, dilation=2)	 # (2) 21 x 21
		self.bn2 = BatchNorm2d(32)
		self.conv3 = Conv2d(32, 64, 6, 2, dilation=3)	 # (3) 10 x 10
		self.bn3 = BatchNorm2d(64)
		self.conv4 = Conv2d(64, 64, 6, 1, dilation=2)		 # (4) 4 x 4
		self.bn4 = BatchNorm2d(64) 
		self.conv5 = Conv2d(64, 128, 6, 2)		 # (5) 2 x 2 
		self.bn5 = BatchNorm2d(128)
		self.conv6 = Conv2d(128, p['enc_h'][l], 4)	 # (6)	512 x 1
		
		self.fc1 = Linear(p['enc_h'][l], p['enc_h'][l]) 
		
				
		if self.has_con:
			# features to continuous latent 
			self.fc_zp = Linear(p['enc_h'][l], self.z_con_dim)
		if self.has_dis:
			# features to categorical latent
			self.fc_alphas = []
			for a_dim in p['nz_dis'][l]:

				self.fc_alphas.append(Linear(p['enc_h'][l],a_dim))
			self.fc_alphas = ModuleList(self.fc_alphas)		

		# setup the non-linearity
		self.dim = p['ldim'][l]
		self.apply(mutils.init_weights)

	def forward(self, x, z_q=None):
		latent_dist = {'con':[], 'dis':[]}
		h = x.view(-1, *self.dim)

		h = F.relu(self.bn1(self.conv1(h)))

		h = F.relu(self.bn2(self.conv2(h)))

		h = F.relu(self.bn3(self.conv3(h)))

		h = F.relu(self.bn4(self.conv4(h)))

		h = F.relu(self.bn5(self.conv5(h)))

		h = F.relu((self.conv6(h)))
		h = h.view(x.size(0), -1)
		h = F.relu(self.fc1(h))
		if self.constrained:
			h = cat((h, z_q), dim=-1)


		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))
		

		return latent_dist['con'], latent_dist['dis']

class MNISTConvDecoder(Module):
	
	def __init__(self, p, l):
		super(MNISTConvDecoder, self).__init__()

		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		
		self.fc1 = Linear(latents, hidden)
		self.fc2 = Linear(hidden,  64*4*4)
		
		self.deconv1 = ConvTranspose2d(64, 32, (4,4),stride=2,padding=1)
		self.deconv2 = ConvTranspose2d(32, 32, (4,4),stride=2,padding=1)
		self.deconv3 = ConvTranspose2d(32, 1,  (4,4),stride=2,padding=1)
		
	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.deconv1(x.view(-1,64,4,4)))
		x = F.relu(self.deconv2(x))
		x = sigmoid(self.deconv3(x)) 
		return x

class CelebConvDecoder(Module):
	
	def __init__(self, p, l):
		super(CelebConvDecoder, self).__init__()

		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		
		self.fc1 = Linear(latents, hidden)
		self.fc2 = Linear(hidden,  64*4*4)
		
		self.deconv1 = ConvTranspose2d(64, 64, (4,4),stride=2,padding=1)
		self.deconv2 = ConvTranspose2d(64, 32, (4,4),stride=2,padding=1)
		self.deconv3 = ConvTranspose2d(32, 32, (4,4),stride=2,padding=1)
		self.deconv4 = ConvTranspose2d(32, 3,  (4,4),stride=2,padding=1)
		
		
	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.deconv1(x.view(-1,64,4,4)))
		x = F.relu(self.deconv2(x))
		x = F.relu(self.deconv3(x))
		x = sigmoid(self.deconv4(x)) 
		return x
		
class Decoder(Module): 
	# Input:  sample from z(l) 
	# Output: prediction at z(l) 
 
	def __init__(self, p,l): 
		super(Decoder, self).__init__() 
		 
		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		 
		# nn init 
		self.dim = p['ldim'][l] 
		self.fc1 = Linear(latents, hidden) 
		self.fc2 = Linear(hidden, hidden) 
		#self.fc3 = Linear(hidden, hidden) 
		self.fc4 = Linear(hidden, np.prod(self.dim)) 
		#self.apply(utils.init_weights) 
 
	def forward(self, x):  
		h1 = tanh(self.fc1(x)) 
		h2 = tanh(self.fc2(h1)) 
		#h3 = tanh(self.fc3(h2)) 
		return sigmoid(self.fc4(h1).view(-1, *self.dim))

class ConvDilationDecoder(Module):
	# keep filter dims pow(2) please
	# 
	def __init__(self, p, l):
		super(ConvDecoder, self).__init__()
		latents = sum(p['z_dim'][l:l+2])
		
		self.conv1 = ConvTranspose2d(latents, 512, 4)	# (1) 2 x 2
		self.bn1 = BatchNorm2d(512)
		self.conv2 = ConvTranspose2d(512, 128, 3, 3)  # (2) 4 x 4
		self.bn2 = BatchNorm2d(128)
		self.conv3 = ConvTranspose2d(128, 64, 6, 1, dilation=2)	 # (3) 10 x 10
		self.bn3 = BatchNorm2d(64)
		self.conv4 = ConvTranspose2d(64, 32, 6, 3)	 # (4) 21 x 21 
		self.bn4 = BatchNorm2d(32)
		self.conv5 = ConvTranspose2d(32, 16, 6, 1, dilation=3) # (6) 
		self.bn5 = BatchNorm2d(16)
		
		self.conv_final = ConvTranspose2d(16, p['ldim'][l][0], 1)
		
		# setup the non-linearity
		self.dim = p['ldim'][l]

	def forward(self, z, a=None, vel_obs=None): 
		
		if not a is None and vel_obs is Nonde:
			z = cat((z, a.squeeze(1), vel_obs.squeeze(1)), dim=-1)
		h = z.view(z.size(0), z.size(1), 1, 1)
		h = F.relu(self.bn1(self.conv1(h)))
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.relu(self.bn4(self.conv4(h)))
		h = F.relu(self.bn5(self.conv5(h)))
		mu_img = self.conv_final(h)
		return mu_img

class ConvDecoder64(Module):
	# keep filter dims pow(2) please
	# 
	def __init__(self, p, l):
		super(ConvDecoder64, self).__init__()
		latents = sum(p['z_dim'][l:l+2])
		
		self.conv1 = ConvTranspose2d(latents, 512, (4,4), stride=2, padding=1)
		self.bn1 = BatchNorm2d(512)
		self.conv2 = ConvTranspose2d(512, 128,	(4,4), stride=2, padding=1)
		self.bn2 = BatchNorm2d(128)
		self.conv3 = ConvTranspose2d(128, 64,  (4,4), stride=2, padding=1)
		self.bn3 = BatchNorm2d(64)
		self.conv4 = ConvTranspose2d(64, 32, (4,4), stride=2, padding=1)
		self.bn4 = BatchNorm2d(32)
		self.conv5 = ConvTranspose2d(32,16,	 (4,4), stride=2, padding=1)
		self.bn5 = BatchNorm2d(16)
		self.conv_final = ConvTranspose2d(16, p['ldim'][l][0],	(4,4), stride=2, padding=1)
		

		# setup the non-linearity
		self.dim = p['ldim'][l]

	def forward(self, z, a=None, vel_obs=None): 
		
		if not a is None and vel_obs is Nonde:
			z = cat((z, a.squeeze(1), vel_obs.squeeze(1)), dim=-1)
		h = z.view(z.size(0), z.size(1), 1, 1)
		h = F.relu(self.bn1(self.conv1(h)))
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.relu(self.bn4(self.conv4(h)))
		h = F.relu(self.bn5(self.conv5(h)))
		mu_img = self.conv_final(h)
		return mu_img

class ConvDecoderAnAI(Module):
	# keep filter dims pow(2) please
	# 
	def __init__(self, p):
		super(ConvDecoderAnAI, self).__init__()
		
		latents = p['z_dim']
		hidden	  = 512
		
		self.fc1 = Linear(latents, hidden)
		self.fc2 = Linear(hidden,  256*7*7)		

		self.conv1 = ConvTranspose2d(256, 128, (3,3),1)
		self.bn1 = BatchNorm2d(128)
		self.conv2 = ConvTranspose2d(128, 64,  (4,4), 2)
		self.bn2 = BatchNorm2d(64)
		self.conv3 = ConvTranspose2d(64, 32, (4,4),2)
		self.bn3 = BatchNorm2d(32)
		self.conv4 = ConvTranspose2d(32, 3, (4,4),2, padding=1)
		
		
	def forward(self, z, a=None, vel_obs=None): 
		
		#if not a is None and vel_obs is Nonde:
		#	z = cat((z, a.squeeze(1), vel_obs.squeeze(1)), dim=-1)
		
		#print('z', z.shape)
		x = F.relu(self.fc1(z))
		#print('x', x.shape)
		x = F.relu(self.fc2(x))
		#print(x.shape)
		h = x.view(x.size(0), 256, 7, 7)
		#print(h.shape)
		h = F.relu(self.bn1(self.conv1(h)))
		#print(h.shape)
		h = F.relu(self.bn2(self.conv2(h)))
		#print(h.shape)
		h = F.relu(self.bn3(self.conv3(h)))
		#print(h.shape)
		mu_img = sigmoid(self.conv4(h))
		#print(mu_img.shape)
		return mu_img

class dSpritesEncoder(Module):
	def __init__(self, p, l):
		super(dSpritesEncoder, self).__init__()

		latents = sum(p['z_dim'][l:l+2])

		self.constrained = l < p['layers']-1

		self.has_con = p['nz_con'][l] > 0 
		self.has_dis = p['nz_dis'][l][0] > 0

		self.z_con_dim = 0; self.z_dis_dim = 0;
		if self.has_con:
			self.z_con_dim = p['nz_con'][l] * p['z_params']
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = len(p['nz_dis'][l])

		self.z_dim = self.z_con_dim + self.z_dis_dim

		if self.has_con:
			# features to continuous latent
			self.fc_zp = Linear(256, self.z_con_dim)
		if self.has_dis:
			# features to categorical latent
			self.fc_alphas = []
			for a_dim in p['nz_dis'][l]:

				self.fc_alphas.append(Linear(256,a_dim))
				self.fc_alphas = ModuleList(self.fc_alphas)


		self.conv1 = Conv2d(1,	32, 4, 2, 1)
		self.conv2 = Conv2d(32, 32, 4, 2, 1)
		self.conv3 = Conv2d(32, 32, 4, 2, 1)
		self.conv4 = Conv2d(32, 32, 4, 2, 1)

		self.linear1 = Linear(32*4*4, 256)
		self.linear2 = Linear(256, 256)

	def forward(self, x, z_q=None):

		latent_dist = {'con':[], 'dis':[]}
		x = x.view(-1, 1, 64, 64)
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		h = F.relu(self.conv3(h))
		h = F.relu(self.conv4(h))
		h = h.view(-1, 32 * 4 * 4)

		h = F.relu(self.linear1(h))
		h = F.relu(self.linear2(h))
		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))


		return latent_dist['con'], latent_dist['dis']

class CarRacingEncoder(Module):
	def __init__(self, p, l):
		super(CarRacingEncoder, self).__init__()

		self.has_con = p['nz_con'][l] > 0 
		self.has_dis = p['nz_dis'][l][0] > 0

		self.z_con_dim = 0; self.z_dis_dim = 0;

		if self.has_con:
			self.z_con_dim = p['nz_con'][l] * p['z_params']
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = len(p['nz_dis'][l])

		self.z_dim = self.z_con_dim + self.z_dis_dim

		if self.has_con:
			# features to continuous latent
			self.fc_zp = Linear(256 * 2 * 2, self.z_con_dim)
		if self.has_dis:
			# features to categorical latent
			self.fc_alphas = []
			for a_dim in p['nz_dis'][l]:

				self.fc_alphas.append(Linear(256,a_dim))
				self.fc_alphas = ModuleList(self.fc_alphas)


		self.conv1 = Conv2d(3,	 32, 4, 2, 1)
		self.conv2 = Conv2d(32,	 64, 4, 2, 1)
		self.conv3 = Conv2d(64,	 128, 4, 2, 1)
		self.conv4 = Conv2d(128, 256, 4, 2, 1)

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]}

		x = x.view(-1, 3, 64, 64)
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		h = F.relu(self.conv3(h))
		h = F.relu(self.conv4(h))
		h = h.view(-1, 256 * 2 * 2)

		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))


		return latent_dist['con'], latent_dist['dis']

class CarRacingDecoder(Module):

	def __init__(self, p, l):
		super(CarRacingDecoder, self).__init__()
		latents = sum(p['z_dim'][l:l+2])
		
		self.linear1 = Linear(latents+p['n_actions'], 32 * 4 * 4)

		self.lstm1 = ConvLSTMCell(32, 32)
		self.dcnv1 = ConvTranspose2d(32, 32, 4, 2, 1)
		self.lstm2 = ConvLSTMCell(32, 32)
		self.dcnv2 = ConvTranspose2d(32, 32, 4, 2, 1)
		self.lstm3 = ConvLSTMCell(32, 32)
		self.dcnv3 = ConvTranspose2d(32, 32, 4, 2, 1)
		self.lstm4 = ConvLSTMCell(32, 32)
		self.dcnv4 = ConvTranspose2d(32, 3, 4, 2, 1)


	def forward(self, z, a, hs, vel_obs=None):
		
		if not hs is None:
			[h1,c1,h2,c2,h3,c3,h4,c4] = hs
		else:
			[h1,c1,h2,c2,h3,c3,h4,c4] = [None]*8
			
		# cell state shouldn't go through activation
		
		h = cat((z,a), dim=1)
		h = F.relu(self.linear1(h))
		h = h.view(-1, 32, 4, 4)
		
		h1, c1 = self.lstm1(h,  (h1, c1))
		x1 = F.relu(self.dcnv1(h1))
		h2, c2 = self.lstm2(x1, (h2,c2))
		x2 = F.relu(self.dcnv2(h2))
		h3, c3 = self.lstm3(x2, (h3,c3))
		x3 = F.relu(self.dcnv3(h3))
		h4, c4 = self.lstm4(x3, (h4, c4))
		x4 = sigmoid(self.dcnv4(h4))
		return x4, [h1,c1,h2,c2,h3,c3,h4,c4]
		
class dSpritesDecoder(Module):

	def __init__(self, p, l):
		super(dSpritesDecoder, self).__init__()
		latents = sum(p['z_dim'][l:l+2])


		self.linear1 = Linear(latents, 256)
		self.linear2 = Linear(256, 256)
		self.linear3 = Linear(256, 32*4*4)

		self.conv1 = ConvTranspose2d(32, 32, 4, 2, 1)
		self.conv2 = ConvTranspose2d(32, 32, 4, 2, 1)
		self.conv3 = ConvTranspose2d(32, 32, 4, 2, 1)
		self.conv4 = ConvTranspose2d(32, 1, 4, 2, 1)


	def forward(self, z, a=None, vel_obs=None):

		h = F.relu(self.linear1(z))
		h = F.relu(self.linear2(h))
		h = F.relu(self.linear3(h))
		h = h.view(-1, 32, 4, 4)

		h = F.relu(self.conv1(h))
		h = F.relu(self.conv2(h))
		h = F.relu(self.conv3(h))
		mu_img = self.conv4(h)

		return mu_img

class ConvDecoder(Module):
	# keep filter dims pow(2) please
	# 
	def __init__(self, p, l):
		super(ConvDecoder, self).__init__()
		latents = sum(p['z_dim'][l:l+2])
		
		self.conv1 = ConvTranspose2d(latents, 512,	4, 2)
		self.bn1 = BatchNorm2d(512)
		self.conv2 = ConvTranspose2d(512, 128,	4, 2)
		self.bn2 = BatchNorm2d(128)
		self.conv3 = ConvTranspose2d(128, 64,  4, 2)
		self.bn3 = BatchNorm2d(64)
		self.conv4 = ConvTranspose2d(64, 32, 4, 2, padding=2)
		self.bn4 = BatchNorm2d(32)
		self.conv_final = ConvTranspose2d(32, p['ldim'][l][0],	(4,4), stride=2, padding=1)
		

		# setup the non-linearity
		self.dim = p['ldim'][l]

	def forward(self, z, a=None, vel_obs=None): 
		
		if not a is None and vel_obs is Nonde:
			z = cat((z, a.squeeze(1), vel_obs.squeeze(1)), dim=-1)
		h = z.view(z.size(0), z.size(1), 1, 1)
		h = F.relu(self.bn1(self.conv1(h)))
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.relu(self.bn4(self.conv4(h)))
		mu_img = sigmoid(self.conv_final(h))
		return mu_img

class FECEncoder(Module):

	def __init__(self, p, l):
		super(FECEncoder, self).__init__()

		latents = sum(p['z_dim'][l:l+2])

		self.constrained = l < p['layers']-1

		self.has_con = p['nz_con'][l] > 0 
		self.has_dis = p['nz_dis'][l][0] > 0

		self.z_con_dim = 0; self.z_dis_dim = 0;
		if self.has_con:
			self.z_con_dim = p['nz_con'][l] * p['z_params']
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = len(p['nz_dis'][l])

		self.z_dim = self.z_con_dim + self.z_dis_dim

		self.conv1 = Conv2d(3,	32, 4, 2, 1)
		self.conv2 = Conv2d(32, 32, 4, 2, 1)
		self.conv3 = Conv2d(32, 64, 4, 2, 1)
		self.conv4 = Conv2d(64, 64, 4, 2, 1)
		self.conv5 = Conv2d(64, 128, 4, 2, 1)
		
		self.linear1 = Linear( 128 * 7 * 7, 512)
		
		if self.has_con:
			# features to continuous latent
			self.fc_zp = Linear(512, self.z_con_dim)
		if self.has_dis:
			# features to categorical latent
			self.fc_alphas = []
			for a_dim in p['nz_dis'][l]:

				self.fc_alphas.append(Linear(512,a_dim))
				self.fc_alphas = ModuleList(self.fc_alphas)

				
	def forward(self, x, l):
		
		latent_dist = {'con':[], 'dis':[]}

		x = x.view(-1, 3, 224, 224)
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		h = F.relu(self.conv3(h))
		h = F.relu(self.conv4(h))
		h = F.relu(self.conv5(h))
		h = h.view(-1, 128 * 7 * 7)

		h = F.relu(self.linear1(h))
		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))

		return latent_dist['con'], latent_dist['dis']		
				
class FECDecoder(Module):
	
	def __init__(self, p, l):
		super(FECDecoder, self).__init__()

		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		
		self.fc1 = Linear(latents, hidden)
		self.fc2 = Linear(hidden,  128*7*7)

		self.deconv1 = ConvTranspose2d(128, 64, (4,4),stride=2,padding=1)
		self.deconv2 = ConvTranspose2d(64,	64, (4,4),stride=2,padding=1)
		self.deconv3 = ConvTranspose2d(64, 32,	(4,4),stride=2,padding=1)
		self.deconv4 = ConvTranspose2d(32, 32,	(4,4),stride=2,padding=1)
		self.deconv5 = ConvTranspose2d(32, 3,	(4,4),stride=2,padding=1)
		
	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.deconv1(x.view(-1,128,7,7)))
		x = F.relu(self.deconv2(x))
		x = F.relu(self.deconv3(x))
		x = F.relu(self.deconv4(x))
		x = sigmoid(self.deconv5(x)) 

		return x

class DynamicModel(Module):
	""" x = f(x,v,P)"""
	"""Approximation of the function underlying hidden state dynamics"""
		
	def __init__(self, p, l):

		super(DynamicModel,self).__init__()
		self.hidden_size = p['lstm_h'][l]
		self.n_layers = p['lstm_l']

		self.BS = p['b']
		self.gpu = p['gpu']
		self.p = p
		
		self.has_con = p['nz_con'][l] is not None
		self.has_dis = p['nz_dis'][l] is not None	

		self.mse  = torch.nn.MSELoss().cuda() if p['gpu'] else MSELoss()
		self.xent = torch.nn.CrossEntropyLoss().cuda() if p['gpu'] else CrossEntropyLoss()		
		
		_, self.q_dist, _, self.cat_dist = mutils.discheck(self.p)		
		
		self.z_con_dim = 0; self.z_dis_dim = 0;
		if self.has_con:
			self.z_con_dim = p['nz_con'][l]
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = sum( [x for x in p['nz_dis'][l]] )
		
		
		self.input_size = self.z_con_dim + self.n_dis_z + p['n_actions']

		if self.has_con:
			# features to continuous latent 
			self.fc_zp = Linear(self.hidden_size, self.z_con_dim*2)
		if self.has_dis:
			# features to categorical latent
			self.fc_alphas = []
			for a_dim in p['nz_dis'][l]:
				self.fc_alphas.append(Linear(self.hidden_size,a_dim))
			self.fc_alphas = ModuleList(self.fc_alphas)		

		self.lstm = LSTM(input_size=self.input_size, num_layers=self.n_layers, hidden_size=self.hidden_size, batch_first=True)
		#self.linear_out = Linear(self.hidden_size, sum(p['z_dim'][l:l+2]) + 1 + 1)
		#self.linear_out = Linear(self.hidden_size, sum(p['z_dim'][l:l+2]) )
		self.reset()
		self.apply(mutils.init_weights)
		
	def reset(self):

		self.lstm_h = Variable(next(self.lstm.parameters()).data.new(self.n_layers,	self.BS, self.hidden_size))
		self.lstm_c = Variable(next(self.lstm.parameters()).data.new(self.n_layers, self.BS, self.hidden_size))

		if self.gpu:
			self.lstm_h = self.lstm_h.cuda() ;
			self.lstm_c = self.lstm_c.cuda() ;
			
		self.lstm_h.zero_()
		self.lstm_c.zero_()

	def loss(self, input, target):
		"""
		Input:	l	
		Output: loss, (metrics)
		l [int] - layer for which to calculate loss 
		loss[scalar] - loss for current layer
		metrics [tuple] - loss (detached) for discrete & continuous kl
		"""
		
		splitdim = [self.p['nz_con'][0], sum(self.p['nz_dis'][0])]
		con_pred, cat_pred = torch.split(input.view(-1, sum(splitdim)), splitdim, dim=1) 
		con_target, cat_target = torch.split(target.view(-1, sum(splitdim)), splitdim, dim=1) 
		
		cat_target = torch.max(cat_target, 1)[1].type(torch.LongTensor).cuda()
		xentloss   = self.xent(cat_pred, cat_target)
		mseloss	   = self.mse(con_pred,	 con_target)
	
		return xentloss + mseloss
			

	def forward(self, r, a):
		
		latent_dist = {}

		a = a.view(self.BS, 1, -1)
		r = r.view(r.shape[0], 1, -1)
		lstm_input = cat((r, a), dim=-1)
		if lstm_input.dim() == 2:
			lstm_input = lstm_dim.unsqueeze(1)
		lstm_out, hidden = self.lstm(lstm_input, (self.lstm_h, self.lstm_c))
		self.lstm_h = hidden[0] ; self.lstm_c = hidden[1]
		
		#linear_out = self.linear_out(lstm_out)
		#done	= linear_out[:,:,0]
		#rew	   = linear_out[:,:,1]
		
		zc = self.fc_zp(lstm_out)

		zd = []
		for fc_alpha in self.fc_alphas:
				zd.append(F.softmax(fc_alpha(lstm_out), dim=1))
		latent_sample = []
		# Continuous sampling 
		norm_sample = self.q_dist.sample_normal(params=zc, train=self.training)
		latent_sample.append(norm_sample)

		# Discrete sampling
		for ind, alpha in enumerate(zd):
			cat_sample = self.cat_dist.sample_gumbel_softmax(alpha, train=self.training)
			latent_sample.append(cat_sample)

		z_pred = torch.cat(latent_sample, dim=-1)
		
		z_pred = z_pred.view(self.BS, -1, 1, 1)

		#return latent_dist['con'], cat(latent_dist['dis'], dim=-1)				
		return z_pred #, done, rew

class ActionNet(Module):

	# It takes as inputs both the latent encoding of the current 
	# frame and the hidden state of the MDN-RNN given past latents 
	# and actions and outputs an action.

	def __init__(self, p, l):
		super(ActionNet, self).__init__()
				
		sum(p['z_dim']),sum(p['z_dim']),p['action_dim'], p['lstm_l']
				
		n_actions = 2
		
		latents = sum(p['z_dim'])
		hidden	= sum(p['lstm_h']) 
		action_dim = p['action_dim']
		rnn_l = p['lstm_l']
		self.BS = p['b']
				
		self.fc1 = Linear(latents+hidden*rnn_l+n_actions, hidden)
		self.fc2 = Linear(hidden, hidden)
		self.lr = Linear(hidden, action_dim)
		self.ud = Linear(hidden, action_dim)
		
		self.reward_history = []
		self.policy_history = Variable(torch.Tensor().cuda())
		self.policy_history.requires_grad = True

		self.reward_episode = []
		self.loss_history	= []
		self.gamma = 0.99

	def loss(self, eval=False):

		R = np.asarray([0.] * self.BS)
		rewards = []

		# Discount future rewards back to the present using gamma
		for r in self.reward_episode[::-1]:
			
			r = np.asarray(r)
			R = r + self.gamma * R
			rewards.insert(0,R)
		
		# Scale rewards
		rewards = torch.FloatTensor(rewards)
		rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
		rewards = Variable(rewards)
		print(rewards.shape) 
		print(self.policy_history.shape)
		rewards.requires_grad = True
		# Calculate loss
		loss1 = Variable(torch.sum(torch.mul(self.policy_history[:,:,0], rewards).mul(-1),-1))
		loss2 = Variable(torch.sum(torch.mul(self.policy_history[:,:,1], rewards).mul(-1),-1))
		loss1.requires_grad = True
		loss2.requires_grad = True
		
		loss = torch.sum(torch.add(loss1,loss2))
		#Save and intialize episode history counters
		
		return loss
		
	def sample_gumbel(self, shape, eps=1e-20):
		U = rand(shape).cuda()
		return -Variable(log(-log(U + eps) + eps))

	def gumbel_softmax_sample(self, logits, temperature):
		y = logits + self.sample_gumbel(logits.size())
		return F.softmax(y / temperature, dim=-1)

	def gumbel_softmax(self, logits, temperature):
		"""
		ST-gumple-softmax
		input: [*, n_class]
		return: flatten --> [*, n_class] an one-hot vector
		"""
		y = self.gumbel_softmax_sample(logits, temperature)
		shape = y.size()
		_, ind = y.max(dim=-1)
		y_hard = zeros_like(y).view(-1, shape[-1])
		y_hard.scatter_(1, ind.view(-1, 1), 1)
		y_hard = y_hard.view(*shape)
		y_hard = (y_hard - y).detach() + y
		return y_hard.view(-1,3)		

	def forward(self, cur_z, prev_lstmh, prev_a): 
			
		cur_z	   = cur_z.view(cur_z.shape[0],-1)
		prev_lstmh = prev_lstmh.view(cur_z.shape[0],-1)
		prev_a	   = prev_a.view(cur_z.shape[0], -1)
				
		x = cat((cur_z, prev_lstmh, prev_a), dim=-1)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		lr = self.lr(x)
		ud = self.ud(x)
		lr_action = self.gumbel_softmax(lr,temperature=1.0)
		ud_action = self.gumbel_softmax(ud,temperature=1.0)
		action = stack([lr_action, ud_action], -2).squeeze(0)
		
		return action

class ErrorUnit(Module):
	""" Calculate layer-wise error """
	# Input: prediction, observation at l
	# Output: Error at l

	def __init__(self, p, l, mask=None):
		super(ErrorUnit,self).__init__()
		#self.in_flat = np.prod(p['imdim'][l])
		#self.out_full = p['ldim'][l] 
				
		#if p['use_rf']:
			# apply RF
		#	self.mask = CustomizedLinear(mask['full'][l], bias=None)
		#	self.fc1 = Linear(self.flat, self.flat, bias=None)

	def forward(self, bottom_up, top_down, noise=None):

		e_up   = F.relu(bottom_up - top_down)
		e_down = F.relu(top_down - bottom_up)
		error  = add(e_up,e_down)#.view(-1,*self.out_full)
		if noise:
			noise = noise.data.normal_(0, 0.01)
			error =	 error + noise		

		return	error
	
class Retina(Module):
	# Foveate input at l0
	# Input:  image data x
	# Output: image data x foveated at (x,y)

	def __init__(self, g, k, patch_noise=None):
		super(Retina, self).__init__()
		self.g = g
		self.k = k
		self.patch_noise = patch_noise

	def foveate(self, x, l):

		phi = []
		size = self.g
		full, fov = self.extract_patch(x, l, size)
		return full, fov

	def extract_patch(self, x, l, size):

		# single-batch matlab stuff
		if len(x.shape) == 2:
			x = FloatTensor(x)
			x = x.unsqueeze(0).unsqueeze(0)

		B, C, H, W = x.shape

		full = zeros_like(x)

		if self.patch_noise:
			full = full.uniform_(0, self.patch_noise)


		#l = l * 100

		patch = zeros(B,C,self.g, self.g)
		#l = torchmax(l, -ones_like(l))
		#l = torchmin(l, ones_like(l))
		#print(l)
		#coords = self.denormalize(H, l).view(B,2)
		#coords = (l*1.5).long() -1
		coords = (l).long() -1
		#print(coords)
		#print(H)
		#print(W)
		#print(l)
		#print(self.g)
		#print(size)
		#0/0
		try:
			patch_x = coords[:, 0]
			patch_y = coords[:, 1]
		except:
			patch_x = coords[0]
			patch_y = coords[1]


		for i in range(B):
			im = x[i].unsqueeze(dim=0)
			T = im.shape[-1]

			try:
				from_x, to_x = patch_x[i] , patch_x[i]
				from_y, to_y = patch_y[i] , patch_y[i]
			except:
				from_x, to_x = patch_x - (size/2) , patch_x + (size/2)
				from_y, to_y = patch_y - (size/2) , patch_y + (size/2)

			# cast to ints
			from_x, to_x = from_x.item(), to_x.item()
			from_y, to_y = from_y.item(), to_y.item()

			x_range = [x - size for x in range(from_x, to_x)]
			x_lim = range(0, W)

			y_range = [y - size for y in range(from_y, to_y)]
			y_lim = range(0, H)

			# todo replace with to/from = max(28, to/from)
			for xi in range(self.g):
				for yi in range(self.g):

					if x_range[xi] in x_lim and y_range[yi] in y_lim:
						patch[i,:,xi,yi] = x[:,:,x_range[xi], y_range[yi]]
						full[i,:,x_range[xi], y_range[yi]] = x[:,:,x_range[xi], y_range[yi]]
					else:
						patch[i,:,xi,yi] = 0

		if x.is_cuda:
			return full.cuda(), patch.cuda()
		else:
			return full, patch

	def denormalize(self, t_size, coords):

		return (0.5 * ((coords + 1.0) * t_size)).long()
		#return ((coords + 1.0) * t_size).long()

	def exceeds(self, from_x, to_x, from_y, to_y, T):

		if (
			(from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T)
		):
			return True
		return False

class Resnet_AnimalAI_Encoder(Module):
	def __init__(self,p):
		super(Resnet_AnimalAI_Encoder, self).__init__()		
		z = p['z_dim']
		
		# CNN architechtures
		self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
		self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)		 # 2d kernal size
		self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)		 # 2d strides
		self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)	 # 2d padding		
		
		# encoding components
		resnet = models.resnet18()
		modules = list(resnet.children())[:-1]		# delete the last fc layer.
		self.resnet = Sequential(*modules)
		
		self.fc1 = Linear(resnet.fc.in_features, 512)
		self.bn1 = BatchNorm1d(512, momentum=0.01)
		self.fc2 = Linear(512, 256)
		self.bn2 = BatchNorm1d(256, momentum=0.01)
		# Latent vectors mu and sigma
		self.fc3_mu = Linear(256, z)	   # output = CNN embedding latent variables
		self.fc3_logvar = Linear(256, z)  # output = CNN embedding latent variables

	def forward(self, x):
		
		x = self.resnet(x)	# ResNet
		x = x.view(x.size(0), -1)  # flatten output of conv

		# FC layers
		x = self.bn1(self.fc1(x))
		x = F.relu(x)
		x = self.bn2(self.fc2(x))
		x = F.relu(x)
		# x = F.dropout(x, p=self.drop_p, training=self.training)
		mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
		return torch.cat([mu, logvar], dim=-1)
		
class Resnet_AnimalAI_Decoder(Module):
	def __init__(self,p):
		super(Resnet_AnimalAI_Decoder, self).__init__()
		# Sampling vector
		self.fc4 = Linear(p['z_dim'], 256)
		self.fc_bn4 = BatchNorm1d(256)
		self.fc5 = Linear(256, 64 * 4 * 4)
		self.fc_bn5 = BatchNorm1d(64 * 4 * 4)
		self.relu = ReLU(inplace=True)

		# Decoder
		self.convTrans6 = Sequential(
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2,padding=1),
			BatchNorm2d(32, momentum=0.01),
			ReLU(inplace=True),
		)
		self.convTrans7 = Sequential(
			ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,padding=1),
			BatchNorm2d(32, momentum=0.01),
			ReLU(inplace=True),
		)


		self.convTrans8 = Sequential(
			ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,padding=1),
			BatchNorm2d(32, momentum=0.01),
			ReLU(inplace=True),
		)

		self.convTrans9 = Sequential(
			ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=4, stride=2,padding=1),
			BatchNorm2d(8, momentum=0.01),
			ReLU(inplace=True),
		)


		self.convTrans10 = Sequential(
			ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=2,padding=1),
			BatchNorm2d(3, momentum=0.01),
			Sigmoid())	 # y = (y1, y2, y3) \in [0 ,1]^3)	
	

	def forward(self, z):
		x = F.relu(self.fc_bn4(self.fc4(z)))
		x = F.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
		x = self.convTrans6(x)
		x = self.convTrans7(x)
		x = self.convTrans8(x)
		x = self.convTrans9(x)
		x = self.convTrans10(x)
		x = F.interpolate(x, size=(84, 84), mode='bilinear', align_corners=True)
		return x	
	
class ResNet_VAE(Module):
	def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
		super(ResNet_VAE, self).__init__()

		self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu



	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		x_reconst = self.decode(z)
		

		return x_reconst, z, mu, logvar

class CustomizedLinear(Module):
	def __init__(self, mask, bias=True):
		"""
		extended torch.nn module which mask connection.
		Argumens
		------------------
		mask [torch.tensor]:
			the shape is (n_input_feature, n_output_feature).
			the elements are 0 or 1 which declare un-connected or
			connected.
		bias [bool]:
			flg of bias.
		"""
		super(CustomizedLinear, self).__init__()
		self.input_features = mask.shape[0]
		self.output_features = mask.shape[1]
		if isinstance(mask, Tensor):
			self.mask = mask.type(float).t()
		else:
			self.mask = Tensor(mask, dtype=float).t()

		self.mask = Parameter(self.mask, requires_grad=False)

		# nn.Parameter is a special kind of Tensor, that will get
		# automatically registered as Module's parameter once it's assigned
		# as an attribute. Parameters and buffers need to be registered, or
		# they won't appear in .parameters() (doesn't apply to buffers), and
		# won't be converted when e.g. .cuda() is called. You can use
		# .register_buffer() to register buffers.
		# nn.Parameters require gradients by default.
		self.weight = Parameter(Tensor(self.output_features, self.input_features))

		if bias:
			self.bias = Parameter(Tensor(self.output_features))
		else:
			# You should always register all possible parameters, but the
			# optional ones can be None if you want.
			self.register_parameter('bias', None)
		self.reset_parameters()

		# mask weight
		self.weight.data = self.weight.data * self.mask

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input):
		# See the autograd section for explanation of what happens here.
		return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

	def extra_repr(self):
		# (Optional)Set the extra information about this module. You can test
		# it by printing an object of this class.
		return 'input_features={}, output_features={}, bias={}'.format(
			self.input_features, self.output_features, self.bias is not None
		)
			
class LinearRF(Module):
	# 'A' layer from prednet  
	# apply RF mask retaining dimensions
	
	def __init__(self,p,l):
		super(LinearRF, self).__init__()

		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l][0] is not None 
		
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l])		

		obs_mask, layer_mask, p = mutils.calc_rf(p)

		self.out_full = p['ldim'][l] 
		
		self.rf_algo = p['rf_algo']
		self.use_rf = p['use_rf']
		
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim']) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(self.imdim, enc_h) 
		self.fc2 = Linear(enc_h, enc_h)

		
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas)			
		
		if p['use_rf']:
			if l>0:
				if p['rf_reduce']:
					on_mask	 = layer_mask['on'][int(l)-1]
					off_mask = layer_mask['off'][int(l)-1]
				else:
					on_mask	 = obs_mask['on'][int(l)]
					off_mask = obs_mask['off'][int(l)]					
			else:
				on_mask	 = obs_mask['on'][int(l)]
				off_mask = obs_mask['off'][int(l)]
				
			self.in_flat = on_mask.shape[0]
			self.fc_c = MaskedLinear(self.in_flat, np.prod(self.out_full), on_mask)
			self.fc_s = MaskedLinear(self.in_flat, np.prod(self.out_full), off_mask)
			
		#else:
			
			#self.in_flat  = np.prod(p['ldim'][l])
			#self.out_full = p['ldim'][l]
			#self.fc1  = Linear(self.in_flat, np.prod(self.out_full), bias=None)

		# apply RF
		#self.mask = CustomizedLinear(mask, bias=None)
		#self.fc1  = Linear(self.in_flat, np.prod(self.out_full), bias=None)

	def forward(self, x, l=None):
		#x = self.fc1(self.mask(x.view(-1,self.in_flat)))

		#print(torch.isnan(x).any())
		if self.use_rf:
			if self.rf_algo == 'independent':
				
				C = F.relu(self.fc_c(x))
				S = F.relu(self.fc_s(x))		
				x = C+S
					
			elif self.rf_algo == 'shared':
			
				C = self.fc_c(x)
				S = self.fc_s(x)
				x = F.relu(C + S)
			
			elif self.rf_algo == 'stacked':
				
				C = F.relu(self.fc_c(x)) 
				#print(torch.isnan(C).any())
				S = F.relu(self.fc_s(x))
				#print(torch.isnan(S).any())
				x = F.relu(C + S);
				#print(torch.isnan(x).any())
			
		else:
			x = self.fc1(x.view(-1,self.in_flat))
	
		#print(torch.isnan(x).any())
		latent_dist = {'con':[], 'dis':[]} 
		
		h = F.relu(self.fc1(x))
		h = F.relu(self.fc2(h))
		#print(torch.isnan(h).any())
		
		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		
		#print(torch.isnan(x).any())
		#print(torch.isnan(latent_dist['con']).any())
		#print(torch.isnan(latent_dist['dis']).any())
		return latent_dist['con'], latent_dist['dis']			
			
		#return x.view(-1,*self.out_full)