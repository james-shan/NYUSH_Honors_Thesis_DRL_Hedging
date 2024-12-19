import gymnasium as gym 
import numpy as np
from tqdm import tqdm

class NormalizeObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low = env.observation_space.low
        self.high = env.observation_space.high

    def normalize_observation(self, observation):
        # Normalize the observation to the range [0, 1]
        normalized_obs = (observation - self.low) / (self.high - self.low)
        return normalized_obs

    def step(self, action):
        observation, reward, done, _, info = self.env.step(action)
        normalized_observation = self.normalize_observation(observation)
        return normalized_observation, reward, done, _, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        normalized_observation = self.normalize_observation(observation)
        return normalized_observation, info
    

def calculate_t_statistic(data):
    n = len(data)  # Number of simulations
    t_statistic = data  / np.std(data, ddof=1) * np.sqrt(n)# Calculate the t-statistic
    return t_statistic

def evaluate_delta_policy(env, agent, num_episodes=1000, pnl=True):
    rewards = np.zeros(num_episodes)
    costs = np.zeros(num_episodes)
    rewards_vol = np.zeros(num_episodes)
    for i in tqdm(range(num_episodes)):
        episode_rewards = np.zeros(env.num_steps)
        state,info = env.reset()
        done = False
        step=0
        while not done:
            action = agent.act(state, info['delta'])
            state, reward, done, _ ,info = env.step(action)
            episode_rewards[step] = reward
            step+=1
        rewards[i] = episode_rewards.sum()
        costs[i] = env.cost
        if pnl:
            rewards_vol[i] = np.std(episode_rewards, ddof=1)
    print("reward mean:{},\n, reward std:{}, \n, costs mean:{}, costs std:{}"\
          .format(rewards.mean(),rewards.std(), costs.mean(), costs.std()))
    return rewards, costs, rewards_vol


def evaluate_drl_policy(env, model, num_episodes=1000, pnl=True):
    rewards = np.zeros(num_episodes)
    costs = np.zeros(num_episodes)
    rewards_vol = np.zeros(num_episodes)
    for i in tqdm(range(num_episodes)):
        episode_rewards = np.zeros(env.num_steps)
        obs ,_= env.reset()
        done = False
        step=0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward ,done ,_,_= env.step(action)
            episode_rewards[step] = reward
            step+=1
        rewards[i] = episode_rewards.sum()
        costs[i] = env.cost
        if pnl:
            rewards_vol[i] = np.std(episode_rewards, ddof=1)
    print("reward mean:{},\n reward std:{}, \n costs mean:{}, \n costs std:{}."\
          .format(rewards.mean(),rewards.std(), costs.mean(), costs.std()))
    return rewards, costs, rewards_vol

    

    