import numpy as np
import gym
import os

env = gym.make("CartPole-v1")
obs = env.reset()
print(obs)
print(env.action_space)
print(env.observation_space)

