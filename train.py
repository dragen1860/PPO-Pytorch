from ppo import PPO
import gym



def make_env():

	env = gym.make('Humanoid-v1')

	return env




def main():

	ppo = PPO(make_env, 2)
	batchsz = 1024

	ppo.render()

	for i in range(10000):

		ppo.update(batchsz)






if __name__ == '__main__':
    main()

