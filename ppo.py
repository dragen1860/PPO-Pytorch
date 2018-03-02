import numpy as np
import torch
from torch.autograd import Variable
from torch import multiprocessing
from torch.multiprocessing import Queue

from policy import Policy
from value import Value
from replay_memory import ReplayMemory



def sampler(pid, queue, env, policy, batchsz):
	"""

	:param pid:
	:param queue:
	:param env:
	:param policy:
	:param batchsz:
	:return:
	"""
	buff = ReplayMemory()

	# we need to sample batchsz of (state, action, next_state, reward, mask)
	# each trajectory contains `trajectory_len` num of items, so we only need to sample
	# `batchsz//trajectory_len` num of trajectory totally
	# the final sampled number may be larger than batchsz.

	sampled_num = 0
	sampled_trajectory_num = 0
	trajectory_len = 20

	while sampled_num < batchsz:

		# for each trajectory, we reset the env and get initial state
		s = env.reset()
		# [s_dim] => [1, s_dim]
		s = Variable(torch.Tensor(s).unsqueeze(0))
		for t in range(trajectory_len):

			# [1, s_dim] => [1, a_dim]
			a = policy.select_action(s)

			# interact with env
			# [1, a_dim] => [a_dim]
			next_s, reward, done, _ = env.step(a.data[0].numpy())

			# a flag indicates ending or not
			mask = 0 if done else 1

			# save to queue
			buff.push(s, a, mask, next_s, reward)

			#
			s = next_s

		sampled_num += trajectory_len
		sampled_trajectory_num += 1

	# when sampling is over, push all buff data into queue
	queue.put([pid, buff])



class PPO:

	# discounted factor
	gamma = 0.99

	# l2 regulazier coefficient
	l2_reg = 1e-3

	# learning rate
	lr = 3e-4

	# clip epsilon of ratio r(theta)
	epsilon = 0.2

	# tau for generalized advantage estimation
	tau = 0.95


	def __init__(self, thread_num, batchsz, env_cls):
		self.thread_num = thread_num

		# we use a dummy env instance to get state dim and action dim etc. information.
		dummy_env = env_cls()

		self.s_dim = dummy_env.observation_shape.shape[0]

		# if continuous action, the action_space.shape(n,) indicates number of continuous action,
		# otherwise, action_space.n stands for number of discrete action.
		is_discrete_action = len(dummy_env.action_space.shape)
		if is_discrete_action == 0:
			self.a_dim = dummy_env.action_space.n
			self.is_discrete_action = True
		else:
			self.a_dim = dummy_env.action_space.shape[0]
			self.is_discrete_action = False



		# initialize envs for each thread
		self.env_list = []
		for _ in range(thread_num):
			self.env_list.append(env_cls())


		# construct policy and value network
		self.policy = Policy(self.s_dim, self.a_dim)
		self.value = Value(self.s_dim)

	def sample(self, batchsz):
		"""
		Given batchsz number of task, the batchsz will be splited equally to each threads
		and when threads return, it merge all data and return
		:param batchsz:
		:return:
		"""

		# batchsz will be splitted into each thread,
		# final batchsz maybe larger than batchsz parameters
		thread_batchsz = np.ceil(batchsz // self.thread_num)
		# buffer to save all data
		queue = Queue()

		# start threads for pid in range(1, threadnum)
		# if threadnum = 1, this part will be ignored.
		threads = []
		for i in range(self.thread_num - 1):
			thread_args = (i + 1, queue, self.env_list[i + 1], self.policy, thread_batchsz)
			threads.append(multiprocessing.Process(target=sampler, args=thread_args))
		for t in threads:
			t.start()

		# start pid = 0
		# all [pid, buff] will be saved in queue
		sampler(0, queue, self.env_list[0], self.policy, thread_batchsz)


		pid, buff0 = queue.get()
		buff = []
		for _ in range(1, self.thread_num):
			pid, buff_ = queue.get()
			buff.append(buff_)
		# buff contains a series of ReplayMemory objects and we use ReplayMemory0 to merage others ReplayMemory objs.
		buff0.append(buff)
		buff = buff0

		# sample a batch from buff
		# here sample all since we assign the total number of threads is batchsz
		batch = buff.sample()

		return batch



	def update(self, batch):
		"""
		update the policy and value network based on current batch data
		:param batch: []
		:return:
		"""
		



