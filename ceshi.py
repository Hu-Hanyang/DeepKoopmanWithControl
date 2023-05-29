import numpy as np
import gym
import os
import dmc2gym
from utility.Utility import data_collecter
from dm_control import suite
from PIL import Image, ImageDraw
from utility.control_swingup import ulqr, upid, rebuild_state
import imageio
# import scipy.linalg as linalg
# lqr = linalg.solve_continuous_are

# from utility.control_swingup import ulqr

# # 创建图像帧
# frames = []
# duration = 0.2  # 每个图像帧的显示时间（以秒为单位）

# # 创建一些简单的图像帧（红色和蓝色的交替矩形）
# for i in range(10):
#     # 创建新的图像帧
#     # image = images[i].transpose(1, 2, 0)
#     # image = Image.fromarray(image, "RGB")
#     img = Image.new('RGB', (200, 200), color=(255, 255, 255))
#     draw = ImageDraw.Draw(img)

#     # 交替绘制红色和蓝色的矩形
#     if i % 2 == 0:
#         draw.rectangle([(50, 50), (150, 150)], fill=(255, 0, 0))
#     else:
#         draw.rectangle([(50, 50), (150, 150)], fill=(0, 0, 255))

#     # 将当前帧添加到帧列表中
#     frames.append(img)

# # 将帧列表保存为GIF文件
# imageio.mimsave('animation.gif', frames, duration=duration)



# env = gym.make("CartPole-v1")
# # obs = env.reset()
# # print(obs)
# print(f"The action space of CartPole in gym is {env.action_space}.")
# print(f"The obs space of CartPole in gym is {env.observation_space}.")
# # # print(env.dt)

# env2 = dmc2gym.make(doxmain_name='cartpole', task_name='swingup', seed=2022, from_pixels=False)  # 'swingup', 'balance'
# env2 = dmc2gym.make(domain_name='cartpole', task_name='swingup', visualize_reward=False, seed=2022, from_pixels=True)  # 'swingup', 'balance'
env2 = dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, from_pixels=False)
# print(f"The action space of CartPole in dm_control is {env2.action_space}.")  # .shape[0]
# print(f"The obs space of CartPole in dm_control is {env2.observation_space}.")
# # print(f"The lower bound of the state are {env2.observation_space.low}")
obs = env2.reset()  # obs.shape = 5
print(f"The obs.shape is {obs.shape}.")
physics_state = env2.physics.get_state()
print(f"The physics_state is {physics_state}.")  # different every time
# # print(f"The type of the physics_state is {physics_state.shape}")
# np.random.seed(2023)
# x = np.random.uniform(low=-1e-3, high=1e-3)
# theta = np.random.uniform(low=0.0, high=2*np.pi)
# v = np.random.uniform(low=-1e-3, high=1e-3)
# omega = np.random.uniform(low=-1e-2, high=1e-2)
# new_physics_state = np.array([x, theta, v, omega])
# print(f"The expected new_physics_state is {new_physics_state}.")
# env2.physics.set_statex(physics_state)
# print(f"The new physics_state is {env2.physics.get_state()}")

# for i in range(10):
#     u = np.random.uniform(env2.action_space.low, env2.action_space.high)
#     next_obs, r, done, info = env2.step(u)
#     rebuild_obs = rebuild_state(env2.physics.get_state()) 
#     print(f"The next_obs is {next_obs}")
#     print(f"The rebuild obs is {rebuild_obs}")
# # print(obs)
# print(f"The new env obs space is {dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, from_pixels=False).observation_space}")
# print(obs.shape)    # 5
# # print(f"The obs is {obs}.")
# # print(f"The type of the obs is {type(obs)}.")
# # 根据\theta 和 角速度来做 pid controller
# # obs_new = np.array([obs[0], np.arccos(obs[1]), obs[3], obs[4]])
# # print(obs_new.shape)
# # print(obs_new)
# # u = ulqr(obs)
# # print(u)
# # print(env2.action_space.low) # Box(-1.0, 1.0, (1,), float32)
# # # print(f"The results of step is {len(env2.step(env2.action_space.sample()))}")

# a = env2.physics.get_state() # a.shape = 4
# # print(a.shape)
# print(f"The get_state is {a}.")
# # print(f"The type of the get_state is {type(obs)}.")
# rebuild_obs = np.array([a[0], np.cos(a[1]), np.sin(a[1]), a[2], a[3]])
# print(f"The new rebuild obs from get_state is {rebuild_obs}.")

# b = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

# env2.physics.set_state(b)
# print(env2.physics.get_state())

# env3 = suite.load(domain_name="cartpole", task_name="swingup")
# action_spec = env3.action_spec()
# time_step = env3.reset()
# # action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
# print(time_step.observation)
# b = env3.physics.get_state()
# print(b.shape)
# print(b)


# gravity = 9.8
# masscart = 1.0
# masspole = 0.1
# total_mass = (masspole + masscart)
# length = 0.5  # actually half the pole's length
# polemass_length = (masspole * length)
# force_mag = 10.0
# tau = 0.02


# H = np.array([
# 	[1, 0, 0, 0],
# 	[0, total_mass, 0, - polemass_length],
# 	[0, 0, 1, 0],
# 	[0, - polemass_length, 0, (2 * length)**2 * masspole / 3]
# 	])

# Hinv = np.linalg.inv(H)

# A = Hinv @ np.array([
#     [0, 1, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 1],
#     [0, 0, - polemass_length * gravity, 0]
# 	])
# B = Hinv @ np.array([0, 1.0, 0, 0]).reshape((4, 1))
# Q = np.diag([0.01, 0.1, 100.0, 0.5])
# R = np.array([[0.1]])

# P = lqr(A, B, Q, R)
# Rinv = np.linalg.inv(R)
# K =  - Rinv @ B.T @ P


# def ulqr(x):
# 	x1 = np.copy(x)
# 	x1[2] = np.sin(x1[2])
# 	# x1 = x1.unsqueeze()
# 	return np.dot(K, x1)

# obs = env.reset()
# # print(obs)
# # u = ulqr(obs)
# # print(u)
# # next_obs, r, done, info = env.step(u)
# # print(next_obs)
# done = False

# while not done:
#     env.render()
#     u = ulqr(obs)
#     next_obs, r, done, inf = env.step(u)
#     print(f"The next_obs is {next_obs}")
#     obs =  next_obs
# env.close()


# def ce(x):
#     if (0 < x and x < 1.57) or (4.71 < x and x < 6.28):
#         print("1")
#     else:
#         print("2")

# ce(2.0)