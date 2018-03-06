import os, time
import numpy as np

import torch
from torch import optim
from torch import multiprocessing
from torch.autograd import Variable

from policy import Policy
from value import Value
from replay_memory import ReplayMemory


def sampler(pid, queue, env, policy, batchsz):
	"""
	This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
	threads.
	:param pid: thread id
	:param queue: multiprocessing.Queue, to collect sampled data
	:param env: environment instance
	:param policy: policy network, to generate action from current policy
	:param batchsz: total sampled items
	:return:
	"""
	buff = ReplayMemory()

	# we need to sample batchsz of (state, action, next_state, reward, mask)
	# each trajectory contains `trajectory_len` num of items, so we only need to sample
	# `batchsz//trajectory_len` num of trajectory totally
	# the final sampled number may be larger than batchsz.

	sampled_num = 0
	sampled_traj_num = 0
	traj_len = 200
	real_traj_len = 0
	avg_reward = []

	while sampled_num < batchsz:
		# total reward per trajectory
		traj_reward = 0
		# for each trajectory, we reset the env and get initial state
		s = env.reset()

		for t in range(traj_len):

			# [1, s_dim] => [1, a_dim]
			# convert numpy data to Variable and then convert back to numpy
			a = policy.select_action(Variable(torch.Tensor(s).unsqueeze(0))).data[0].numpy()

			# interact with env
			# [1, a_dim] => [a_dim]
			next_s, reward, done, _ = env.step(a)

			# a flag indicates ending or not
			mask = 0 if done else 1

			# save to queue
			buff.push(s, a, mask, next_s, reward)

			# update per step
			s = next_s
			traj_reward += reward
			real_traj_len = t

			if done:
				break

		# this is end of one trajectory
		sampled_num += real_traj_len
		sampled_traj_num += 1
		# t indicates the valid trajectory lenght
		avg_reward.append(traj_reward)

	# this is end of sampling all batchsz of items.
	avg_reward = sum(avg_reward) / len(avg_reward)
	# when sampling is over, push all buff data into queue
	queue.put([pid, buff, avg_reward])


class PPO:
	# discounted factor
	gamma = 0.99

	# l2 regulazier coefficient
	l2_reg = 0

	# learning rate
	lr = 3e-4

	# clip epsilon of ratio r(theta)
	epsilon = 0.2

	# tau for generalized advantage estimation
	tau = 0.95

	def __init__(self, env_cls, thread_num):
		"""

		:param env_cls: env class or function, not instance, as we need to create several instance in class.
		:param thread_num:
		"""

		self.thread_num = thread_num
		self.env_cls = env_cls

		# we use a dummy env instance to get state dim and action dim etc. information.
		dummy_env = env_cls()

		self.s_dim = dummy_env.observation_space.shape[0]

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
		self.value_optim = optim.Adam(self.value.parameters(), lr=self.lr)


		# this lock is to keep policy network weighted untouched when executing render thread.
		# The render thread depends on policy network to get latest policy and in order to make sure the policy network
		# unchanged when rendering one trajectory, we use this lock to lock the weights of policy network.
		# TODO: lock can not work yet.
		self.lock = multiprocessing.Lock()


	def est_adv(self, r, v, mask):
		"""
		we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
		:param r: reward, Tensor, [b]
		:param v: estimated value, Tensor, [b]
		:param mask: indicates ending for 0 otherwise 1, Tensor, [b]
		:return: A(s, a), V-target(s), both Tensor
		"""
		batchsz = v.size(0)

		# v_target is worked out by Bellman equation.
		v_target = torch.Tensor(batchsz)
		delta = torch.Tensor(batchsz)
		A_sa = torch.Tensor(batchsz)

		prev_v_target = 0
		prev_v = 0
		prev_A_sa = 0
		for t in reversed(range(batchsz)):
			# mask here indicates a end of trajectory
			# this value will be treated as the target value of value network.
			# mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
			# formula: V(s_t) = r_t + gamma * V(s_t+1)
			v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

			# please refer to : https://arxiv.org/abs/1506.02438
			# for generalized adavantage estimation
			# formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
			delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

			# formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
			# here use symbol tau as lambda, but original paper uses symbol lambda.
			A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

			# update previous
			prev_v_target = v_target[t]
			prev_v = v[t]
			prev_A_sa = A_sa[t]

		# normalize A_sa
		A_sa = (A_sa - A_sa.mean()) / A_sa.std()

		return A_sa, v_target

	def update(self, batchsz):
		"""
		firstly sample batchsz items and then perform optimize algorithms.
		:param batchsz:
		:return:
		"""
		# 1. sample data asynchronously
		batch = self.sample(batchsz)

		# data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
		# batch.action: ([1, a_dim], [1, a_dim]...)
		# batch.reward/ batch.mask: ([1], [1]...)
		s = torch.from_numpy(np.stack(batch.state))
		a = torch.from_numpy(np.stack(batch.action))
		r = torch.Tensor(np.stack(batch.reward))
		mask = torch.Tensor(np.stack(batch.mask))
		batchsz = s.size(0)

		# 2. get estimated V(s) and PI_old(s, a),
		# actually, PI_old(s, a) can be saved when interacting with env, so as to save the time of one forward elapsed.
		# v: [b, 1] => [b]
		v = self.value(Variable(s)).data.squeeze()
		log_pi_old_sa = self.policy.get_log_prob(Variable(s), Variable(a)).data

		# 3. estimate advantage and v_target according to GAE and Bellman Equation
		A_sa, v_target = self.est_adv(r, v, mask)


		# 4. backprop.
		# the following code episode will involve in neural network forward/backward, hence we convert related variable
		# into Variable
		v_target = Variable(v_target)
		A_sa = Variable(A_sa)
		s = Variable(s)
		a = Variable(a)
		log_pi_old_sa = Variable(log_pi_old_sa)

		for _ in range(5):

			# 4.1 shuffle current batch
			perm = torch.randperm(batchsz)
			# shuffle the variable for mutliple optimize
			v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = v_target[perm], A_sa[perm], s[perm], a[perm], \
			                                                               log_pi_old_sa[perm]

			# 4.2 get mini-batch for optimizing
			optim_batchsz = 4096
			optim_chunk_num = int(np.ceil(batchsz / optim_batchsz))
			# chunk the optim_batch for total batch
			v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
			                                                               torch.chunk(A_sa_shuf, optim_chunk_num), \
			                                                               torch.chunk(s_shuf, optim_chunk_num), \
			                                                               torch.chunk(a_shuf, optim_chunk_num), \
			                                                               torch.chunk(log_pi_old_sa_shuf,
			                                                                           optim_chunk_num)
			# 4.3 iterate all mini-batch to optimize
			for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
			                                                         log_pi_old_sa_shuf):
				# print('optim:', batchsz, v_target_b.size(), A_sa_b.size(), s_b.size(), a_b.size(), log_pi_old_sa_b.size())
				# 1. update value network
				v_b = self.value(s_b)
				loss = torch.pow(v_b - v_target_b, 2).mean()
				self.value_optim.zero_grad()
				loss.backward()
				# nn.utils.clip_grad_norm(self.value.parameters(), 4)
				self.value_optim.step()

				# 2. update policy network by clipping
				# [b, 1]
				log_pi_sa = self.policy.get_log_prob(s_b, a_b)
				# ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
				# we use log_pi for stability of numerical operation
				# [b, 1] => [b]
				ratio = torch.exp(log_pi_sa - log_pi_old_sa_b).squeeze(1)
				surrogate1 = ratio * A_sa_b
				surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
				# this is element-wise comparing.
				# we add negative symbol to convert gradient ascent to gradient descent.
				surrogate = - torch.min(surrogate1, surrogate2).mean()

				# backprop
				self.policy_optim.zero_grad()
				surrogate.backward(retain_graph=True)
				# gradient clipping, for stability
				torch.nn.utils.clip_grad_norm(self.policy.parameters(), 10)
				# self.lock.acquire() # retain lock to update weights
				self.policy_optim.step()
				# self.lock.release() # release lock

	def sample(self, batchsz):
		"""
		Given batchsz number of task, the batchsz will be splited equally to each threads
		and when threads return, it merge all data and return
		:param batchsz:
		:return: batch
		"""

		# batchsz will be splitted into each thread,
		# final batchsz maybe larger than batchsz parameters
		thread_batchsz = np.ceil(batchsz / self.thread_num).astype(np.int32)
		# buffer to save all data
		queue = multiprocessing.Queue()

		# start threads for pid in range(1, threadnum)
		# if threadnum = 1, this part will be ignored.
		# when save tensor in Queue, the thread should keep alive till Queue.get(),
		# please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847/2
		evt = multiprocessing.Event()
		threads = []
		for i in range(self.thread_num):
			thread_args = (i, queue, self.env_list[i], self.policy, thread_batchsz)
			threads.append(multiprocessing.Process(target=sampler, args=thread_args))
		for t in threads:
			# set the thread as daemon, and it will be killed once the main thread is stoped.
			t.daemon = True
			t.start()

		# we need to get the first ReplayMemory object and then merge others ReplayMemory use its append function.
		pid0, buff0, avg_reward0 = queue.get()
		avg_reward = [avg_reward0]
		for _ in range(1, self.thread_num):
			pid, buff_, avg_reward_ = queue.get()
			buff0.append(buff_) # merge current ReplayMemory into buff0
			avg_reward.append(avg_reward_)

		# now buff saves all the sampled data and avg_reward is the average reward of current sampled data
		buff = buff0
		avg_reward = np.array(avg_reward).mean()

		print('avg reward:', avg_reward)

		return buff.sample()

	def render(self, interval=8):
		"""
		call this function to start render thread.
		The function will return when the render thread started.
		:return:
		"""
		thread = multiprocessing.Process(target=self.render_, args=(interval,))
		thread.start()

	def render_(self, interval):
		"""
		This function should be called by render()
		:param interval:
		:return:
		"""
		env = self.env_cls()
		s = env.reset()

		while True:
			# [s_dim] => [1, s_dim]
			s = Variable(torch.Tensor(s)).unsqueeze(0)
			# [1, s_dim] => [1, a_dim] => [a_dim]
			a = self.policy.select_action(s).squeeze().data.numpy()
			# interact with env
			s, r, done, _ = env.step(a)

			env.render()

			if done:
				s = env.reset()
				time.sleep(interval)

	def save(self, filename='ppo'):

		torch.save(self.value.state_dict(), filename + '.val.mdl')
		torch.save(self.policy.state_dict(), filename + '.pol.mdl')

		print('saved network to mdl')

	def load(self, filename='ppo'):
		value_mdl = filename + '.val.mdl'
		policy_mdl = filename + '.pol.mdl'
		if os.path.exists(value_mdl):
			self.value.load_state_dict(torch.load(value_mdl))
			print('loaded checkpoint from file:', value_mdl)
		if os.path.exists(policy_mdl):
			self.policy.load_state_dict(torch.load(policy_mdl))

			print('loaded checkpoint from file:', policy_mdl)
