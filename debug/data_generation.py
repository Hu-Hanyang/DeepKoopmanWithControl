import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
# import sys
# sys.path.append("../utility/")
from scipy.integrate import odeint
from utility.Utility import data_collecter
import time
import numpy as np
import gym
import random
from scipy.integrate import odeint
import scipy.linalg
from copy import copy
from utility.rbf import rbf
from gym import spaces
import dmc2gym
import sys
from utility.control_swingup import ulqr, upid, rebuild_state
from PIL import Image, ImageDraw
import imageio
from controller.PPO import PPO


class data_generator():
    def __init__(self,env_name, pixel=True) -> None:
        self.env_name = env_name
        self.pixel = pixel
        np.random.seed(2022)
        random.seed(2022)
        if pixel == True:
            # generate RGB data in shape of (80, 80, 3)
            print(f"The environment is the dm_control CartPole with pixel.")
            self.env = dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, height=80, width=180, camera_id=0, visualize_reward=False, from_pixels=True)  # seed=2022, visualize_reward=False, from_pixels=True
            self.udim = self.env.action_space.shape[0]
            self.Nstates = 5  # 5
            self.umin = self.env.action_space.low  # -1
            self.umax = self.env.action_space.high  # +1
            self.observation_space = dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, from_pixels=False).observation_space
            self.dt = 0.02  # from gym CartPole self.tau, seconds between state updates
        else:
            print(f"The environment is the dm_control CartPole with state.")
            self.env = dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, from_pixels=False)  # seed=2022, visualize_reward=False, from_pixels=True
            self.udim = self.env.action_space.shape[0]
            self.Nstates = 5  # 5
            self.umin = self.env.action_space.low  # -1
            self.umax = self.env.action_space.high  # +1   
            self.observation_space = self.env.observation_space
            self.dt = self.env.dt 
        self.env.reset()

    def collect_koopman_data(self, traj_num, steps, mode="train", controller=None):  # add controller method as input
        # traj_num = 50000, steps = 15
        # the input of the controller is the state (5, ), and the output is a scalar 
        self.controller = controller
        train_data = np.empty((steps+1,traj_num,self.Nstates+self.udim))  
        if self.pixel == True:
            print("Generate visible training data for CartPole-dm now!")
            for traj_i in range(traj_num):
                frames = []
                duration = 0.2
                image0 = self.env.reset()
                frames.append(Image.fromarray(image0.transpose(1, 2, 0), "RGB"))
                s0 = rebuild_state(self.env.physics.get_state()) # s0.shape = (5,)
                u10 = self.controller.select_action(s0)  # for PPO: self.controller.select_action(s0); others: self.controller(s0)
                train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
                for i in range(1,steps+1):
                    next_image, r, done,_ = self.env.step(u10)
                    frames.append(Image.fromarray(next_image.transpose(1, 2, 0), "RGB"))
                    s0 = rebuild_state(self.env.physics.get_state())
                    u10 = self.controller.select_action(s0)  # for PPO: self.controller.select_action(s0); others: self.controller(s0)
                    train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
                path = f'/localhome/hha160/projects/DeepKoopmanWithControl/debug/under_{self.controller}control_{steps}'
                if not os.path.exists(path):
                    os.makedirs(path)
                imageio.mimsave(f'{path}/traj{traj_i}.gif', frames, duration=duration)
                
        else:  
            print("Generate invisible training data for CartPole-dm now!")
            for traj_i in range(traj_num):
                frames = []
                duration = 0.2
                s0 = self.env.reset()
                u10 = self.controller(s0)  # np.random.uniform(self.umin, self.umax)
                train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
                for i in range(1,steps+1):
                    s0, r, done,_ = self.env.step(u10)
                    u10 = self.controller(s0)  # np.random.uniform(self.umin, self.umax)
                    train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)

        return train_data



# data collecter
data_Generator = data_generator("CartPole-dm", pixel=True)

# choose PPO controller
# initialize a PPO agent
ppo_controller = PPO(state_dim=5, action_dim=1, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2, has_continuous_action_space=True)
random_seed = 0             #### set this to load a particular checkpoint trained on random seed
run_num_pretrained = 200000      #### set this to load a particular checkpoint num
directory = "/localhome/hha160/projects/DeepKoopmanWithControl/controller" + '/' 
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format("CartPole-dm", random_seed, run_num_pretrained)
print("loading network from : " + checkpoint_path)
ppo_controller.load(checkpoint_path)

# generate training data
training_data = data_Generator.collect_koopman_data(traj_num=10, steps=350, mode="train", controller=ppo_controller)
in_dim =training_data.shape[-1] - 1
print(in_dim)