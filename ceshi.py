import numpy as np
import gym
import os
import dmc2gym

env = gym.make("CartPole-v1")
# obs = env.reset()
# print(obs)
print(f"The action space of CartPole in gym is {env.action_space}.")
print(f"The obs space of CartPole in gym is {env.observation_space}.")
# print(env.dt)

env2 = dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, from_pixels=False)
print(f"The action space of CartPole in dm_control is {env2.action_space}.")
print(f"The obs space of CartPole in dm_control is {env2.observation_space}.")
obs = env2.reset()
# print(env2.dt)

