from ppo import PPO
import gym



def make_env():

	env = gym.make('Humanoid-v2')

	return env




def main():

	ppo = PPO(make_env, 4)
	batchsz = 1024

	for i in range(10000):

		batch = ppo.sample(batchsz)
		ppo.update(batch)






if __name__ == '__main__':
    main()

