from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory:

	def __init__(self):
		self.memory = []

	def push(self, *args):

		self.memory.append(Transition(*args))

	def sample(self, batchsz=None):
		if batchsz is None:
			return Transition(*zip(*self.memory))
		else:
			random_batch = random.sample(self.memory, batchsz)
			return Transition(*zip(*random_batch))

	def append(self, new_memory):
		self.memory += new_memory.memory

	def __len__(self):
		return len(self.memory)
