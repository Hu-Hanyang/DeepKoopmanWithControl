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


Methods = ["KoopmanDerivative","KoopmanRBF",\
            "KNonlinear","KNonlinearRNN","KoopmanU",\
            "KoopmanNonlinearA","KoopmanNonlinear",\
                ]

# select the method and the env
method_index = 4
# Step1: choose the env and suffix
# suffix = "test"
# env_name = "CartPole-v1"
suffix = "dm_control_mixed_PPO"  # test, test1
env_name = "CartPole-dm"
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
            x_ref = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
            Q = np.zeros((NKoopman,NKoopman))
            Q[1,1] = 0.01
            Q[2,2] = 5.0
            Q[3,3] = 5.0
            Q[4,4] = 0.01
            Q[5,5] = 0.01
            # Q[4,4] = 0.01  # I add this
            R = 0.001*np.eye(1)
            reset_state=  [0.0, 0.96,-0.3, 0, 0]  # [cart位置，角度sin，角度cos，cart速度，pole角速度]
        else:  # "CartPole-v1"
            Q = np.zeros((NKoopman,NKoopman))
            Q[1,1] = 0.01
            Q[2,2] = 5.0
            Q[3,3] = 0.01
            R = 0.001*np.eye(1)
            reset_state=  [0.0, 1.0,-0.3, 0.3]  #  [cart position, pole angle, cart velocity,  pole angular velocity], original: [0.0, 0.0,-0.3, 0]
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

print("Let the control task begin: \n")
Ad = state_dict['lA.weight'].cpu().numpy()
Bd = state_dict['lB.weight'].cpu().numpy()

# Step2: name the video
env = Data_collect.env
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
# observation = env.reset_state(reset_state)  # for "CartPole-v1"
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
frames = []  # Step 4: to save the image of "CartPole-dm"
duration = 0.2
frames.append(Image.fromarray(image0.transpose(1, 2, 0), "RGB"))

for i in range(steps):
    # images.append(env.render(mode="rgb_array"))
    u = -Kopt*(x0 - x_ref_lift)
    # u = max(-umax,min(umax,u[0,0]))
    # print(type(u[0,0]),type(u))
    next_image, reward, done, info = env.step(u[0,0])
    frames.append(Image.fromarray(next_image.transpose(1, 2, 0), "RGB"))
    observation = rebuild_state(env.physics.get_state()) # s0.shape = (5,)

    rewards += reward
    x0 = np.matrix(Psi_o(observation, net))
    # x0 = Ad*x0+Bd*u
    observation_list.append(x0[:Nstate].reshape(-1,1))
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

# Step 5: generate video while using "CartPole-dm"
path = f'/localhome/hha160/projects/DeepKoopmanWithControl/control_py/ControlResults'
if not os.path.exists(path):
    os.makedirs(path)
imageio.mimsave(f'{path}/result{steps}.gif', frames, duration=duration)
                
# print(images[0].shape)
# save images
# for i in range(len(images)):
#     image = images[i]
#     image = Image.fromarray(image, "RGB")
#     path = "/localhome/hha160/projects/DeepKoopmanWithControl/dm_images"
#     if not os.path.exists(path):
#         os.makedirs(path)
#     image.save(f"{path}/x{i}.png")

# img_array = []
# for filename in glob.glob('/localhome/hha160/projects/DeepKoopmanWithControl/dm_images/*.png'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)
# out = cv2.VideoWriter('videos_dm/CartPole-dm.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()