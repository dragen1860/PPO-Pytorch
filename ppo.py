import numpy as np
import torch
from torch.autograd import Variable
from torch import multiprocessing
from torch.multiprocessing import Queue
from torch import optim
from torch import nn

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
			# [s_dim] = [1, s_dim]
			next_s = Variable(torch.Tensor(next_s).unsqueeze(0))

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


	def __init__(self, env_cls, thread_num):
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
		self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)
		self.value_optim = optim.Adam(self.value.parameters(), lr=self.lr, weight_decay=self.l2_reg)

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


	def estimate_advantage(self, batch, v):
		"""

		:param batch:
		:return:
		"""

		# batch: [(s, a, mask, next_s, reward)]
		# s, next_s, a: Variable
		# mask, reward: scalar
		# batch: namedtuple('state':[b, s_dim], 'next_state':[b, s_dim],...)
		batchsz = batch.state.size(0)
		s = batch.state
		next_s = batch.next_state
		mask = batch.mask
		reward = batch.reward
		a = batch.action

		Q_sa = torch.Tensor(batchsz, 1)
		delta = torch.Tensor(batchsz, 1)
		A_sa = torch.Tensor(batchsz, 1)


		prev_Q_sa = 0
		prev_value = 0
		prev_A_sa = 0
		for t in reversed(range(batchsz)):
			# TODO: why no trajectory information saved?
			# mask here indicates a end of trajectory
			Q_sa[t] = reward[t] + self.gamma * prev_Q_sa * mask[t]

			# please refer to : https://arxiv.org/abs/1506.02438
			# for generalized adavantage estimation
			delta[t] = reward[t] + self.gamma * prev_value * mask[t] - v[t]

			#
			A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

			# update previous
			prev_Q_sa = Q_sa[t, 0]
			prev_value = v[t, 0]
			prev_A_sa = A_sa[t, 0]

		# normalize A_sa
		A_sa = (A_sa - A_sa.mean()) / A_sa.std()

		return A_sa, Q_sa



	def update(self, batch):
		"""
		update the policy and value network based on current batch data
		:param batch: []
		:return:
		"""

		# buff.push(s, a, mask, next_s, reward)
		# s,a,next_s: Variable
		# mask, reward: scalar
		s = batch.state
		a = batch.action
		v = self.value(s)
		# log(PI_old(a|s))
		log_pi_old_sa = self.policy.get_log_prob(s, a)


		A_sa, Q_sa = self.estimate_advantage(batch, v)

		for _ in range(5):

			# 1. update value network
			loss = torch.pow(v - Q_sa, 2).mean()
			self.value_optim.zero_grad()
			loss.backward()
			self.value_optim.step()

			# 2. update policy network
			log_pi_sa = self.policy.get_log_prob(s, a)
			# ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
			# we use log_pi for stability of numerical operation
			ratio = torch.exp(log_pi_sa - log_pi_old_sa)
			surrogate1 = ratio * A_sa
			surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa
			surrogate = - torch.min(surrogate1, surrogate2).mean()

			# backprop
			self.policy_optim.zero_grad()
			surrogate.backward()
			# gradient clipping, for stability
			nn.utils.clip_grad_norm(self.policy.parameters(), 40)
			self.policy_optim.step()







