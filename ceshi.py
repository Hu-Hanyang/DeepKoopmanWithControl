import numpy as np
import gym
import os
import dmc2gym

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
print(obs.shape)  
print(f"The obs is {obs}.")

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

