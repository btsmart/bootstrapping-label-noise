# coding=utf-8
"""
some training utils.
reference:
	https://github.com/ZJULearning/RMI/blob/master/utils/train_utils.py
	https://github.com/zhanghang1989/PyTorch-Encoding

Contact: zhaoshuaimcc@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


class lr_scheduler(object):

	def __init__(self,
						init_lr=0.1,
						num_epochs=100,
						iters_per_epoch=300,
						slow_start_epochs=0,
						slow_start_lr=1e-4,
						end_lr=1e-3,
						):

		self.init_lr = init_lr
		self.now_lr = self.init_lr
		self.end_lr = end_lr

		self.num_epochs = num_epochs
		self.iters_per_epoch = iters_per_epoch
		
		self.slow_start_iters = slow_start_epochs * iters_per_epoch
		self.slow_start_lr = slow_start_lr
		self.total_iters = (num_epochs - slow_start_epochs) * iters_per_epoch
		
		print("INFO:PyTorch: Using learning rate scheduler with"
				" warm-up epochs of {}!".format(slow_start_epochs))

	def __call__(self, optimizer, i, epoch):
		"""call method"""
		T = epoch * self.iters_per_epoch + i

		if self.slow_start_iters > 0 and T <= self.slow_start_iters:
			lr = (1.0 * T / self.slow_start_iters) * (self.init_lr - self.slow_start_lr)
			lr = min(lr + self.slow_start_lr, self.init_lr)
		
		else:
			T = T - self.slow_start_iters
			lr = 0.5 * self.init_lr * (1.0 + math.cos(1.0 * T / self.total_iters * math.pi))

		lr = max(lr, self.end_lr)
		self.now_lr = lr

		# adjust learning rate
		self._adjust_learning_rate(optimizer, lr)

	def _adjust_learning_rate(self, optimizer, lr):
		"""adjust the leaning rate"""
		if len(optimizer.param_groups) == 1:
			optimizer.param_groups[0]['lr'] = lr
		else:
			# BE CAREFUL HERE!!!
			# 0 -- the backbone conv weights with weight decay
			# 1 -- the bn params and bias of backbone without weight decay
			# 2 -- the weights of other layers with weight decay
			# 3 -- the bn params and bias of other layers without weigth decay
			optimizer.param_groups[0]['lr'] = lr
			optimizer.param_groups[1]['lr'] = lr
			for i in range(2, len(optimizer.param_groups)):
				optimizer.param_groups[i]['lr'] = lr * self.multiplier




class lr_scheduler_2(object):

	def __init__(self,
						init_lr=0.1,
						total_iters=100,
						slow_start_iters=0,
						slow_start_lr=1e-4,
						end_lr=1e-3,
						):

		self.init_lr = init_lr
		self.now_lr = self.init_lr
		self.end_lr = end_lr

		self.slow_start_iters = slow_start_iters
		self.slow_start_lr = slow_start_lr
		self.total_iters = total_iters - slow_start_iters
		
		print("INFO:PyTorch: Using learning rate scheduler with"
				" warm-up iters of {}!".format(slow_start_iters))

	def __call__(self, optimizer, i):
		"""call method"""
		T = i

		if self.slow_start_iters > 0 and T <= self.slow_start_iters:
			lr = (1.0 * T / self.slow_start_iters) * (self.init_lr - self.slow_start_lr)
			lr = min(lr + self.slow_start_lr, self.init_lr)
		
		else:
			T = T - self.slow_start_iters
			lr = 0.5 * self.init_lr * (1.0 + math.cos(1.0 * T / self.total_iters * math.pi))

		lr = max(lr, self.end_lr)
		self.now_lr = lr

		# print(f"{i}: {self.now_lr}")

		# adjust learning rate
		self._adjust_learning_rate(optimizer, lr)

	# @TODO: Modify this function when we introduce different weight decays
	def _adjust_learning_rate(self, optimizer, lr):
		"""adjust the leaning rate"""
		if len(optimizer.param_groups) == 1:
			optimizer.param_groups[0]['lr'] = lr
		else:
			# BE CAREFUL HERE!!!
			# 0 -- the backbone conv weights with weight decay
			# 1 -- the bn params and bias of backbone without weight decay
			# 2 -- the weights of other layers with weight decay
			# 3 -- the bn params and bias of other layers without weigth decay
			optimizer.param_groups[0]['lr'] = lr
			optimizer.param_groups[1]['lr'] = lr
			for i in range(2, len(optimizer.param_groups)):
				optimizer.param_groups[i]['lr'] = lr * self.multiplier





class lr_scheduler_3(object):

	def __init__(self,
		init_lr=0.1,
		milestones=[],
		gamma=0.1,
	):

		self.init_lr = init_lr
		self.now_lr = self.init_lr

		self.lr_milestones = milestones
		self.lr_gamma = gamma

		print("INFO:PyTorch: Using multi-step learning rate")

	def __call__(self, optimizer, i, epoch):

		self.now_lr = self.init_lr
		for milestone in self.lr_milestones:
			if epoch >= milestone:
				self.now_lr *= self.lr_gamma

		# adjust learning rate
		self._adjust_learning_rate(optimizer, self.now_lr)

	# @TODO: Modify this function when we introduce different weight decays
	def _adjust_learning_rate(self, optimizer, lr):
		"""adjust the leaning rate"""
		if len(optimizer.param_groups) == 1:
			optimizer.param_groups[0]['lr'] = lr
		else:
			# BE CAREFUL HERE!!!
			# 0 -- the backbone conv weights with weight decay
			# 1 -- the bn params and bias of backbone without weight decay
			# 2 -- the weights of other layers with weight decay
			# 3 -- the bn params and bias of other layers without weigth decay
			optimizer.param_groups[0]['lr'] = lr
			optimizer.param_groups[1]['lr'] = lr
			for i in range(2, len(optimizer.param_groups)):
				optimizer.param_groups[i]['lr'] = lr * self.multiplier



def get_parameter_groups(model, norm_weight_decay=0):
	"""
	Separate model parameters from scale and bias parameters following norm if
	training imagenet
	"""
	model_params = []
	norm_params = []

	for name, p in model.named_parameters():
		if p.requires_grad:
			# if 'fc' not in name and ('norm' in name or 'bias' in name):
			if 'norm' in name or 'bias' in name or 'bn' in name:
				# print(f'Norm: {name}')
				norm_params += [p]
			else:
				# print(f'Other: {name}')
				model_params += [p]

	return [{'params': model_params},
			{'params': norm_params,
				'weight_decay': norm_weight_decay}]
