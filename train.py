from ppo import PPO
import gym
import torch


def make_env():

	# env = gym.make('Humanoid-v1')
	env = gym.make('Hopper-v1')
	return env




def main():

	torch.set_default_tensor_type('torch.DoubleTensor')

	batchsz = 1024
	ppo = PPO(make_env, 3)

	# load model from checkpoint
	ppo.load()
	# comment this line to close evaluaton thread, to speed up training process.
	# ppo.render()

	for i in range(10000):

		ppo.update(batchsz)

		if i % 100 == 0 and i:
			ppo.save()






if __name__ == '__main__':
    main()

