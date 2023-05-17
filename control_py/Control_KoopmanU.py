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
from utility.lqr import *

Methods = ["KoopmanDerivative","KoopmanRBF",\
            "KNonlinear","KNonlinearRNN","KoopmanU",\
            "KoopmanNonlinearA","KoopmanNonlinear",\
                ]

# select the method and the env
method_index = 4
suffix = "test"
env_name = "CartPole-v1"
# suffix = "Pendulum1_26"
# env_name = "Pendulum-v1"
# suffix = "5_2"
# env_name = "DampingPendulum"
# suffix = "MountainCarContinuous1_26"
# env_name = "MountainCarContinuous-v0"

method = Methods[method_index]
root_path = "/localhome/hha160//projects/DeepKoopmanWithControl/Data/"+suffix
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
Nstate = Data_collect.Nstates
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
    net = lka.Network(layer,blayer,NKoopman,udim)
elif method.endswith("KoopmanU"):
    layer = dicts["layer"]
    NKoopman = layer[-1]+Nstate
    net = lka.Network(layer,NKoopman,udim)  
net.load_state_dict(state_dict)
device = torch.device("cpu")
net.cpu()
net.double()

def Prepare_Region_LQR(env_name,precision = 0.1):
    x_ref = np.zeros(Nstate)
    if env_name.startswith("CartPole"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[1,1] = 0.01
        Q[2,2] = 5.0
        Q[3,3] = 0.01
        R = np.eye(1)
        theta_region = np.arange(-2.0,2.0,precision)
        dtheta_region = np.arange(-2.0,2.0,precision)
        reset_state_list = []
        for theta in theta_region:
            for dtheta in dtheta_region:
                reset_state_list.append([0.0,0.0,theta,dtheta])
    elif env_name.startswith("Pendulum"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 5.0
        Q[1,1] = 0.01
        R = np.eye(1)
        theta_region = np.arange(-6.0,6.0,precision)
        dtheta_region = np.arange(-6.0,6.0,precision)
        reset_state_list = []
        for theta in theta_region:
            for dtheta in dtheta_region:
                reset_state_list.append([theta,dtheta])
    elif env_name.startswith("DampingPendulum"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 5.0
        Q[1,1] = 0.01
        R = np.eye(1)
        theta_region = np.arange(-4.0,4.0,precision)
        dtheta_region = np.arange(-4.0,4.0,precision)
        reset_state_list = []
        for theta in theta_region:
            for dtheta in dtheta_region:
                reset_state_list.append([theta,dtheta])
    elif env_name.startswith("MountainCarContinuous"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 5.0
        Q[1,1] = 0.1
        R = np.eye(1)
        x_region = np.arange(-1.2,0.6,precision)
        dx_region = np.arange(-1.0,1.0,precision)
        reset_state_list = []
        for x in x_region:
            for dx in dx_region:
                reset_state_list.append([x,dx])
        x_ref[0] = 0.45
    Q = np.matrix(Q)
    R = np.matrix(R)
    return Q,R,reset_state_list,x_ref

def Psi_o(s,net): # Evaluates basis functions Ψ(s(t_k))
    psi = np.zeros([NKoopman,1])
    ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
    psi[:NKoopman,0] = ds
    return psi

def Done(env_name,state):
    if env_name.startswith("CartPole"):
        done = (abs(state[2]) >= np.pi)
    if env_name.startswith("Pendulum"):
        done = (abs(state[0]) >= 2*np.pi)
    if env_name.startswith("DampingPendulum"):
        done = (abs(state[0]) >= 2*np.pi)
    if env_name.startswith("MountainCarContinuous"):
        done = (state[0]>0.7 or state[0]<-1.3)
    return done 

def exp(env,env_name,net,Ad,Bd,Q,R,reset_state,x_ref):
    Kopt = lqr_regulator_k(Ad,Bd,Q,R)
    observation_list = []
    observation = np.array(env.reset_state(reset_state))
    x0 = np.matrix(Psi_o(observation,net)).reshape(NKoopman,1)
    x_ref_lift = Psi_o(x_ref,net).reshape(NKoopman,1)
    observation_list.append(x0[:Nstate].reshape(-1,1))
    u_list = []
    steps = 200
    flag = False
    for i in range(steps):
        u = -Kopt*(x0-x_ref_lift)
        observation, reward, done, info = env.step(u[0,0])
        done = Done(env_name,observation)
        if done:
            flag = True
            break
        x0 = np.matrix(Psi_o(observation,net)).reshape(NKoopman,1)
        observation_list.append(x0[:Nstate].reshape(-1,1))
        u_list.append(u)
    u_list = np.array(u_list).reshape(-1)
    observations = np.concatenate(observation_list,axis=1)
    return observations,u_list,flag

def criterion(env_name,observations,flag):
    if flag:
        return 0
    elif env_name.startswith("CartPole"):
        err = np.mean(abs(observations[2:,195:]))
        good = int(err <= 1e-2)
    elif env_name.startswith("Pendulum"):
        err = np.mean(abs(observations[:,195:]))
        good = int(err <= 1e-2)        
    elif env_name.startswith("DampingPendulum"):
        err = np.mean(abs(observations[:,195:]))
        good = int(err <= 1e-2)    
    elif env_name.startswith("MountainCarContinuous"):
        err = np.mean(abs(observations[0,195:]-0.45))+np.mean(abs(observations[1,195:]))
        good = int(err <= 1e-2)      
    return good


def Err(env_name,observations,flag):
    if flag:
        return None
    elif env_name.startswith("CartPole"):
        err = np.mean(abs(observations[2:,195:]))
    elif env_name.startswith("Pendulum"):
        err = np.mean(abs(observations[:,195:]))
    elif env_name.startswith("DampingPendulum"):
        err = np.mean(abs(observations[:,195:]))
    elif env_name.startswith("MountainCarContinuous"):
        err = np.mean(abs(observations[0,195:]-0.45))+np.mean(abs(observations[1,195:]))
    return err

def Cost(observations,u_list,Q,R,x_ref):
    steps = observations.shape[1]
    loss = 0
    for s in range(steps):
        if s!=steps-1:
            ucost = np.dot(np.dot(u_list[s].T,R),u_list[s])
            loss += ucost[0,0]
        xcost = np.dot(np.dot((observations[:,s]-x_ref).T,Q),(observations[:,s]-x_ref))
        loss += xcost[0,0]
    return loss

# print("Plot the safe region:\n")
# Ad = state_dict['lA.weight'].cpu().numpy()
# Bd = state_dict['lB.weight'].cpu().numpy()
# env = Data_collect.env
# env.reset()
# Ad = np.matrix(Ad)
# Bd = np.matrix(Bd)
# Results = {}
# precision = 0.1
# Results_all = {}
# Q,R,reset_state_list,x_ref = Prepare_Region_LQR(env_name,precision=precision)
# for u_val in [0.001,0.01,0.1,1,2,5,10,20,50,100]:
#     Results = {}
#     for reset_state in reset_state_list:
#             obs,_, flag  = exp(env,env_name,net,Ad,Bd,Q,u_val*R,reset_state,x_ref)
#             result = criterion(env_name,obs,flag)
#             Results[(reset_state[-2],reset_state[-1])] = result
#     Results_all[u_val]= Results
#     points = []
#     for key in Results:
#         if Results[key]:
#             points.append(np.array([key[0],key[1]]))
#     points = np.array(points).reshape(-1,2)
#     plt.plot(points[:,0],points[:,1],'r*')
#     plt.savefig("/localhome/hha160/projects/DeepKoopmanWithControl/control_py/ControlResults/"+env_name+"_KoopmanU_saferegion_uval{}.png".format(u_val))
#     plt.show()
#     # print(points.shape)
#     np.save("/localhome/hha160/projects/DeepKoopmanWithControl/control_py/ControlResults/"+env_name+"_KoopmanU_saferegion_uval{}.npy".format(u_val),points)

# points = []
# for key in Results:
#     if Results[key]:
#         points.append(np.array([key[0],key[1]]))
# points = np.array(points).reshape(-1,2)
# plt.plot(points[:,0],points[:,1],'r*')
# plt.savefig("/localhome/hha160/projects/DeepKoopmanWithControl/control_py/ControlResults/"+env_name+"_KoopmanU_saferegion.png")
# plt.show()
# # print(points.shape)
# np.save("/localhome/hha160/projects/DeepKoopmanWithControl/control_py/ControlResults/"+env_name+"_KoopmanU_saferegion.npy",points)

def Psi_o(s,net): # Evaluates basis functions Ψ(s(t_k))
    psi = np.zeros([NKoopman,1])
    ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
    psi[:NKoopman,0] = ds
    return psi

def Prepare_LQR(env_name):
    x_ref = np.zeros(Nstate)
    if env_name.startswith("CartPole"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[1,1] = 0.01
        Q[2,2] = 5.0
        Q[3,3] = 0.01
        R = 0.001*np.eye(1)
        reset_state=  [0.0,0.0,-0.3,0]
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

env = Data_collect.env

# env = gym.make("CartPole-v1")  # , render_mode="human"
# env.seed(2022)

env = gym.wrappers.RecordVideo(env, video_folder='videos', video_length=200, name_prefix="DKUC")

env.reset()
Ad = np.matrix(Ad)
Bd = np.matrix(Bd)
Q,R,reset_state,x_ref = Prepare_LQR(env_name)
# print(f"The Q matrix is: \n {Q}; \n The R metrix is {R}. \n")
Kopt = lqr_regulator_k(Ad,Bd,Q,R)
observation_list = []
observation = env.reset_state(reset_state)  # convert x0 to z0
x0 = np.matrix(Psi_o(observation,net))
x_ref_lift = Psi_o(x_ref,net)
observation_list.append(x0[:Nstate].reshape(-1,1))
# print(Kopt)
u_list = []
steps = 200
# umax = 100
rewards = 0.0
for i in range(steps):
    # env.render()
    u = -Kopt*(x0-x_ref_lift)
    # u = max(-umax,min(umax,u[0,0]))
    # print(type(u[0,0]),type(u))
    observation, reward, done, info = env.step(u[0,0])
    rewards += reward
    x0 = np.matrix(Psi_o(observation,net))
    # x0 = Ad*x0+Bd*u
    observation_list.append(x0[:Nstate].reshape(-1,1))
    u_list.append(u)
    # time.sleep(0.1)
env.close()

observations = np.concatenate(observation_list,axis=1)
u_list = np.array(u_list).reshape(-1)
time_history = np.arange(steps+1)*env.dt
print(f"In {steps} steps, the total rewards is {rewards}.")
for i in range(Nstate):
    plt.plot(time_history, observations[i,:].reshape(-1,1), label="x{}".format(i))
plt.grid(True)
plt.title("LQR Regulator")
plt.legend()
plt.show()