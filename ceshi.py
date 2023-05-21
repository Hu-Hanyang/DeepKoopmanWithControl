import numpy as np
import gym
import os
import dmc2gym
from utility.Utility import data_collecter


# env = gym.make("CartPole-v1")
# # obs = env.reset()
# # print(obs)
# print(f"The action space of CartPole in gym is {env.action_space}.")
# print(f"The obs space of CartPole in gym is {env.observation_space}.")
# # print(env.dt)

env2 = dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, from_pixels=False)
print(f"The action space of CartPole in dm_control is {env2.action_space}.")
print(f"The obs space of CartPole in dm_control is {env2.observation_space}.")
obs = env2.reset()  # obs.shape = 5
print(obs.shape)    # 5
print(f"The obs is {obs}.")
# print(f"The results of step is {len(env2.step(env2.action_space.sample()))}")

a = env2.physics.get_state() # a.shape = 4???
print(a.shape)
print(f"The get_state is {a}.")
b = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

# env2.physics.set_state(b)
# print(env2.physics.get_state())

# physics_state = env2._env.physics.get_state()
# print(f"The physics_state is {physics_state}.")  # different every time
# # qpos, qvel, act
# env2._env.physics.set_state(physics_state)

# data_collect1 = data_collecter("CartPole-dm")
# print(f"The number of dimensions of the data_collect1 is {data_collect1.Nstates}")
# Ktest_data = data_collect1.collect_koopman_data(5, 1, mode="eval")
# Ktest_samples = Ktest_data.shape[1]
# in_dim = Ktest_data.shape[-1] - 1
# print(f"The in_dim is {in_dim}")


# data_collect2 = data_collecter("CartPole-v1")
# print(f"The number of dimensions of the data_collect2 is {data_collect2.Nstates}")
