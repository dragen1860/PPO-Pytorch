import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def normal_entropy(std):
	"""
	compute the entropy of normal distribution.
	please refer to https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived
	for more details.
	:param std: [b, a_dim]
	:return:    [b, 1]
	"""
	var = std.pow(2)
	entropy = 0.5 + 0.5 * torch.log(2 * var * np.pi)
	return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std):
	"""
	x ~ N(mean, std)
	this function will return log(prob(x)) while x belongs to guassian distrition(mean, std)
	:param x:       [b, a_dim]
	:param mean:    [b, a_dim]
	:param log_std: [b, a_dim]
	:return:        [b, 1]
	"""
	std = torch.exp(log_std)
	var = std.pow(2)
	log_density = - torch.pow(x - mean, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std

	return log_density.sum(1, keepdim=True)


class Policy(nn.Module):
	def __init__(self, s_dim, a_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(nn.Linear(s_dim, 128),
		                         nn.ReLU(),
		                         nn.Linear(128, 128),
		                         nn.ReLU(),
		                         nn.Linear(128, a_dim))

		# this is Variable of nn.Module, added to class automatically
		# it will be optimized as well.
		self.a_log_std = nn.Parameter(torch.zeros(1, a_dim))

	def forward(self, s):
		# [b, s_dim] => [b, a_dim]
		a_mean = self.net(s)

		# [1, a_dim] => [b, a_dim]
		a_log_std = self.a_log_std.expand_as(a_mean)

		return a_mean, a_log_std

	def select_action(self, s):
		"""

		:param s:
		:return:
		"""
		# forward to get action mean and log_std
		# [b, s_dim] => [b, a_dim]
		a_mean, a_log_std = self.forward(s)

		# randomly sample from normal distribution, whose mean and variance come from policy network.
		# [b, a_dim]
		a = torch.normal(a_mean, torch.exp(a_log_std))

		return a

	def get_log_prob(self, s, a):
		"""

		:param s:
		:param a:
		:return:
		"""
		# forward to get action mean and log_std
		# [b, s_dim] => [b, a_dim]
		a_mean, a_log_std = self.forward(s)

		# [b, a_dim] => [b, 1]
		log_prob = normal_log_density(a, a_mean, a_log_std)

		return log_prob
