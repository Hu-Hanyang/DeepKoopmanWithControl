import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import os
import matplotlib.pyplot as plt
import random
import argparse
from collections import OrderedDict
from copy import copy
import scipy
import scipy.linalg
from utility.Utility import data_collecter
import dmc2gym
from utility.lqr import *
import cv2
from PIL import Image
import imageio
import glob
from utility.control_swingup import ulqr, upid, rebuild_state
import optuna



Methods = ["KoopmanDerivative","KoopmanRBF",\
            "KNonlinear","KNonlinearRNN","KoopmanU",\
            "KoopmanNonlinearA","KoopmanNonlinear",\
                ]

# select the method and the env
method_index = 4
# Step1: choose the env and suffix
# suffix = "test"
# env_name = "CartPole-v1"
# suffix = "dm_control_mixed_PPO"  # test, test1
suffix = "dm_control_mixed_PPO_final"
env_name = "CartPole-dm"
# suffix = "dm_control_cheetah1"
# env_name = "Cheetah-dm"
# suffix = "Pendulum1_26"
# env_name = "Pendulum-v1"
# suffix = "5_2"
# env_name = "DampingPendulum"
# suffix = "MountainCarContinuous1_26"
# env_name = "MountainCarContinuous-v0"

method = Methods[method_index]
root_path = "/localhome/hha160/projects/DeepKoopmanWithControl/Data/"+suffix
print(f"The control method is {method}")
if method.endswith("KNonlinear"):
    import train.Learn_Knonlinear as lka
elif method.endswith("KNonlinearRNN"):
    import train.Learn_Knonlinear_RNN as lka
elif method.endswith("KoopmanNonlinear"):
    import train.Learn_KoopmanNonlinear_with_KlinearEig as lka
elif method.endswith("KoopmanNonlinearA"):
    import train.Learn_KoopmanNonlinearA_with_KlinearEig as lka
elif method.endswith("KoopmanU"):
    import train.Learn_Koopman_with_KlinearEig as lka
for file in os.listdir(root_path):
    if file.startswith(method+"_") and file.endswith(".pth"):
        model_path = file  

model_path = "KoopmanU_CartPole-dmlayer3_edim20_eloss0_gamma0.8_aloss1_samples500019999th_step.pth"
print(f"The model_path is {model_path}.")
print(f"The model we choose is {root_path} + / {model_path}")
Data_collect = data_collecter(env_name)
udim = Data_collect.udim
Nstate = Data_collect.Nstates  # Data_collect.Nstates
layer_depth = 3
layer_width = 128
dicts = torch.load(root_path+"/"+model_path)
state_dict = dicts["model"]
if method.endswith("KNonlinear"):
    Elayer = dicts["Elayer"]
    net = lka.Network(layers=Elayer,u_dim=udim)
elif method.endswith("KNonlinearRNN"):
    net = lka.Network(input_size=udim+Nstate,output_size=Nstate,hidden_dim=layer_width, n_layers=layer_depth-1)
elif method.endswith("KoopmanNonlinear") or method.endswith("KoopmanNonlinearA"):
    layer = dicts["layer"]
    blayer = dicts["blayer"]
    NKoopman = layer[-1]+Nstate
    print(f"The NKoopman is {NKoopman}")
    net = lka.Network(layer,blayer,NKoopman,udim)
elif method.endswith("KoopmanU"):  # use this 
    layer = dicts["layer"]
    NKoopman = layer[-1]+Nstate
    net = lka.Network(layer,NKoopman,udim)  
net.load_state_dict(state_dict)
device = torch.device("cpu")
net.cpu()
net.double()

def Psi_o(s,net): # Evaluates basis functions Ψ(s(t_k))
    psi = np.zeros([NKoopman,1])
    ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
    psi[:NKoopman,0] = ds
    return psi

def Prepare_LQR(env_name):
    x_ref = np.zeros(Nstate)
    if env_name.startswith("CartPole"):  # "CartPole-v1", "CartPole-dm"
        if env_name == "CartPole-dm":
            # the state dimension is 5
            x_ref = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
            Q = np.zeros((NKoopman,NKoopman))
            # Q = 0.1 * np.ones((NKoopman,NKoopman))
            # Best para
            Q[0,0] = 10.0  # 84.12  # 
            Q[1,1] = 60.0  # 62.07  # 50.0
            Q[2,2] = 60.0  # 65.79  # 1.0
            Q[3,3] = 10 # 0.04 # 15.0
            Q[4,4] = 10#   # 10.0            
            # # Manual para1
            # Q[0,0] =0.01  # 84.12  # 
            # Q[1,1] = 10.0  # 62.07  # 50.0
            # Q[2,2] = 10.0  # 65.79  # 1.0
            # Q[3,3] = 0.01 # 0.04 # 15.0
            # Q[4,4] = 0.01 #   # 10.0
            # # Manual para2
            # Q[0,0] =10.0  # 84.12  # 
            # Q[1,1] = 50.0  # 62.07  # 50.0
            # Q[2,2] = 50.0  # 65.79  # 1.0
            # Q[3,3] = 1.0 # 0.04 # 15.0
            # Q[4,4] = 1.0 #   # 10.0
            R = 0.001*np.eye(1)
            reset_state=  [0.0, 0.96,-0.3, 0, 0]  # [cart位置，角度cos， 角度sin，cart速度，pole角速度]
        else:  # "CartPole-v1"
            Q = np.zeros((NKoopman,NKoopman))
            Q[1,1] = 0.01
            Q[2,2] = 5.0
            Q[3,3] = 0.01
            R = 0.001*np.eye(1)
            reset_state=  [0.0, 1.0,-0.3, 0.3]  #  [cart position, pole angle, cart velocity,  pole angular velocity], original: [0.0, 0.0,-0.3, 0]
    elif env_name == "Cheetah-dm":
        pass
    elif env_name.startswith("Pendulum"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 5.0
        Q[1,1] = 0.01
        R = 0.001*np.eye(1)
        reset_state = [-3.0,0.5]
    elif env_name.startswith("DampingPendulum"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 5.0
        Q[1,1] = 0.01
        R = 0.08*np.eye(1)
        reset_state = [-2.5,0.1]   
    elif env_name.startswith("MountainCarContinuous"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 5.0
        Q[1,1] = 0.01
        R = 0.001*np.eye(1)
        reset_state = [0.5,0.0]  
        x_ref[0] = 0.45
    Q = np.matrix(Q)
    R = np.matrix(R)
    return Q,R,reset_state,x_ref

def build_LQR(paras):
    # the state dimension is 5
    x_ref = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    Q = np.zeros((NKoopman,NKoopman))
    Q[0,0] = paras[0]
    Q[1,1] = paras[1]
    Q[2,2] = paras[2]
    Q[3,3] = paras[3]
    Q[4,4] = paras[4]
    # Q[5,5] = 0.01
    # Q[4,4] = 0.01  # I add this
    R = 0.001*np.eye(1)
    return Q, R, x_ref

def cal_rewards(env, Kopt, x_ref, steps=300):
    image0 = env.reset()  # for "CartPole-dm"
    observation = rebuild_state(env.physics.get_state()) # s0.shape = (5,)
    x0 = np.matrix(Psi_o(observation,net))
    x_ref_lift = Psi_o(x_ref, net)
    rewards = 0.0
    for i in range(steps):
        u = -Kopt*(x0 - x_ref_lift)
        next_image, reward, done, info = env.step(u[0,0])
        # frames.append(Image.fromarray(next_image.transpose(1, 2, 0), "RGB"))
        observation = rebuild_state(env.physics.get_state()) # s0.shape = (5,)
        rewards += reward
        x0 = np.matrix(Psi_o(observation, net))
    return rewards
    

print("Let the control task begin: \n")
Ad = state_dict['lA.weight'].cpu().numpy()
Bd = state_dict['lB.weight'].cpu().numpy()

# Step2: name the video
env = Data_collect.env
# env = dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, visualize_reward=False, from_pixels=True)  # seed=2022, visualize_reward=False, from_pixels=True
# env = dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, from_pixels=False)  # seed=2022, visualize_reward=False, from_pixels=True
# env = gym.wrappers.RecordVideo(env, video_folder='videos_dm', video_length=200, name_prefix="dm")  # DKUC
env.reset()

Ad = np.matrix(Ad)
Bd = np.matrix(Bd)
print(f"The shape of the Ad is {Ad.shape}")



Q, R, reset_state, x_ref = Prepare_LQR(env_name)
print(f"The reference state is {x_ref}")
print(f"The shape of the Q is {Q.shape}")

Kopt = lqr_regulator_k(Ad, Bd, Q, R)
observation_list = []

# Step3: choose the observation to circumvent the reset_state() function
image0 = env.reset()  # for "CartPole-dm"
observation = rebuild_state(env.physics.get_state()) # s0.shape = (5,)

print(f"The shape of the observation is {observation.shape}")

x0 = np.matrix(Psi_o(observation,net))
print(f"The shape of latent state is {x0.shape}")
x_ref_lift = Psi_o(x_ref, net)
observation_list.append(x0[:Nstate].reshape(-1,1))
# print(Kopt)
u_list = []
steps = 300
# umax = 100
rewards = 0.0

# Step 4: to save the image of "CartPole-dm"
# frames = []  
# duration = 0.2
# frames.append(Image.fromarray(image0.transpose(1, 2, 0), "RGB"))

for i in range(steps):
    # images.append(env.render(mode="rgb_array"))
    u = -Kopt*(x0 - x_ref_lift)
    # u = max(-umax,min(umax,u[0,0]))
    # print(type(u[0,0]),type(u))
    observation, reward, done, info = env.step(u[0,0])
    # frames.append(Image.fromarray(next_image.transpose(1, 2, 0), "RGB"))
    # observation = rebuild_state(env.physics.get_state()) # s0.shape = (5,)

    rewards += reward
    x0 = np.matrix(Psi_o(observation, net))
    # x0 = Ad*x0+Bd*u
    # observation_list.append(x0[:Nstate].reshape(-1,1))
    u_list.append(u)
    # time.sleep(0.1)
env.close()

observations = np.concatenate(observation_list,axis=1)
u_list = np.array(u_list).reshape(-1)
# print(u_list)
time_history = np.arange(steps+1)*0.02  # env.dt
print(f"In {steps} steps, the total rewards is {rewards}.")

# for i in range(Nstate):
#     plt.plot(time_history, observations[i,:].reshape(-1,1), label="x{}".format(i))
# plt.grid(True)
# plt.title("LQR Regulator")
# plt.legend()
# plt.show()

# # Step 5: generate video while using "CartPole-dm"
# path = f'/localhome/hha160/projects/DeepKoopmanWithControl/control_py/ControlResults'
# if not os.path.exists(path):
#     os.makedirs(path)
# imageio.mimsave(f'{path}/result{steps}.gif', frames, duration=duration)

# def objective(trial):
#     a = trial.suggest_float("a", 0.0, 100.0)
#     b = trial.suggest_float("b", 0.0, 100.0)
#     c = trial.suggest_float("c", 0.0, 100.0)
#     d = trial.suggest_float("d", 0.0, 100.0)
#     e  = trial.suggest_float("e", 0.0, 100.0)

#     paras = [a, b, c, d, e]

#     Q, R, x_ref = build_LQR(paras)
#     Kopt = lqr_regulator_k(Ad, Bd, Q, R)
#     rewards = cal_rewards(env, Kopt, x_ref, steps=300)
#     return rewards


# study = optuna.create_study(study_name='koop', direction='maximize')
# study.optimize(objective, n_trials=2000, n_jobs=1, show_progress_bar=True)
# print(f"The best parameters are {study.best_params}")
# print(f"The best value is {study.best_value}")
