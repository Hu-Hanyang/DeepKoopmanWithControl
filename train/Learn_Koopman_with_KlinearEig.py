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
#define network
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega
    
class Network(nn.Module):
    def __init__(self,encode_layers,Nkoopman,u_dim):
        super(Network,self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers)-1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i],encode_layers[layer_i+1])
            if layer_i != len(encode_layers)-2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman,Nkoopman,bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim,Nkoopman,bias=False)

    def encode_only(self,x):
        return self.encode_net(x)

    def encode(self,x):
        return torch.cat([x,self.encode_net(x)],axis=-1)
    
    def forward(self,x,u):
        return self.lA(x)+self.lB(u)

def K_loss(data,net,u_dim=1,Nstate=4):
    steps,train_traj_num,Nstates = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_current = net.encode(data[0,:,u_dim:])
    max_loss_list = []
    mean_loss_list = []
    for i in range(steps-1):
        X_current = net.forward(X_current,data[i,:,:u_dim])
        Y = data[i+1,:,u_dim:]
        Err = X_current[:,:Nstate]-Y
        max_loss_list.append(torch.mean(torch.max(torch.abs(Err),axis=0).values).detach().cpu().numpy())
        mean_loss_list.append(torch.mean(torch.mean(torch.abs(Err),axis=0)).detach().cpu().numpy())
    return np.array(max_loss_list),np.array(mean_loss_list)


#loss function
def Klinear_loss(data,net,mse_loss,u_dim=1,gamma=0.99,Nstate=4,all_loss=0):
    steps,train_traj_num,NKoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_current = net.encode(data[0,:,u_dim:])
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1,dtype=torch.float64).to(device)
    Augloss = torch.zeros(1,dtype=torch.float64).to(device)
    for i in range(steps-1):
        X_current = net.forward(X_current,data[i,:,:u_dim])
        beta_sum += beta
        if not all_loss:
            loss += beta*mse_loss(X_current[:,:Nstate],data[i+1,:,u_dim:])
        else:
            Y = net.encode(data[i+1,:,u_dim:])
            loss += beta*mse_loss(X_current,Y)
        X_current_encoded = net.encode(X_current[:,:Nstate])
        Augloss += mse_loss(X_current_encoded,X_current)
        beta *= gamma
    loss = loss/beta_sum
    Augloss = Augloss/beta_sum
    return loss+0.5*Augloss

def Stable_loss(net,Nstate):
    x_ref = np.zeros(Nstate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_ref_lift = net.encode_only(torch.DoubleTensor(x_ref).to(device))
    loss = torch.norm(x_ref_lift)
    return loss

def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs()-torch.ones(1,dtype=torch.float64).to(device)
    mask = c>0
    loss = c[mask].sum()
    return loss

def train(env_name, train_steps = 20000,suffix="",all_loss=0,\
            encode_dim = 12, layer_depth=3, e_loss=1, gamma=0.5, Ktrain_samples=5000):
    # Ktrain_samples = 1000
    # Ktest_samples = 1000
    Ktrain_samples = Ktrain_samples
    Ktest_samples = 2000 # 20000
    Ksteps = 300  # 15
    Kbatch_size = 100
    res = 1
    normal = 1
    #data prepare
    print(f"The total training steps is {train_steps}; the number of the training samples is {Ktrain_samples}; the number of the testing samples is {Ktest_samples}.")
    data_collect = data_collecter(env_name)
    u_dim = data_collect.udim
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples, Ksteps, mode="eval")
    Ktest_samples = Ktest_data.shape[1]
    print("test data ok!,shape:",Ktest_data.shape)  # (16, 20000, 5)
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples, Ksteps, mode="train")
    print("train data ok!,shape:",Ktrain_data.shape)  # shape: (16, 50000, 5)
    Ktrain_samples = Ktrain_data.shape[1]  # Ktrain_samples = 50000
    in_dim = Ktest_data.shape[-1] - u_dim
    Nstate = in_dim
    # layer_depth = 4
    layer_width = 128
    layers = [in_dim]+[layer_width]*layer_depth+[encode_dim]
    Nkoopman = in_dim+encode_dim
    print("layers:",layers)
    net = Network(layers, Nkoopman, u_dim)
    # print(net.named_modules())
    eval_step = 1000
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda() 
    net.double()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                    lr=learning_rate)
    for name, param in net.named_parameters():
        print("model:",name,param.requires_grad)
    #train
    eval_step = 1000
    best_loss = 1000.0
    current_state_dict = {}
    logdir = "../Data/"+suffix+"/KoopmanU_"+env_name+"layer{}_edim{}_eloss{}_gamma{}_aloss{}_samples{}".format(layer_depth, encode_dim, e_loss,gamma,all_loss,Ktrain_samples)
    if not os.path.exists( "../Data/"+suffix):
        os.makedirs( "../Data/"+suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    start_time = time.process_time()
    for i in range(train_steps):
        #K loss
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:,Kindex[:Kbatch_size],:]
        Kloss = Klinear_loss(X,net,mse_loss,u_dim,gamma,Nstate,all_loss)
        Eloss = Eig_loss(net)
        loss = Kloss+Eloss if e_loss else Kloss
        # loss = Kloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        writer.add_scalar('Train/Kloss',Kloss,i)
        writer.add_scalar('Train/Eloss',Eloss,i)
        writer.add_scalar('Train/loss',loss,i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i+1) % eval_step ==0:
            #K loss
            with torch.no_grad():
                Kloss = Klinear_loss(Ktest_data,net,mse_loss,u_dim,gamma,Nstate,all_loss=0)
                Eloss = Eig_loss(net)
                loss = Kloss
                Kloss = Kloss.detach().cpu().numpy()
                Eloss = Eloss.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy()
                writer.add_scalar('Eval/Kloss',Kloss,i)
                writer.add_scalar('Eval/Eloss',Eloss,i)
                # writer.add_scalar('Eval/best_loss',best_loss,i)
                writer.add_scalar('Eval/loss',loss,i)
                # if loss<best_loss:
                # best_loss = copy(Kloss)
                current_state_dict = copy(net.state_dict())
                Saved_dict = {'model':current_state_dict,'layer':layers}
                torch.save(Saved_dict, logdir + f"{i}th_step.pth")
                print("Step:{} Eval-loss{} K-loss:{} ".format(i,loss,Kloss))
            # print("-------------END-------------")
        # writer.add_scalar('Eval/best_loss',best_loss,i)
        # if (time.process_time()-start_time)>=210*3600:
        #     print("time out!:{}".format(time.clock()-start_time))
        #     break
    torch.save(net.state_dict(), logdir + "final_step.pth")
    print("Finish training!")
    

def main():
    train(args.env,suffix=args.suffix,all_loss=args.all_loss,\
        encode_dim=args.encode_dim,layer_depth=args.layer_depth,\
            e_loss=args.e_loss,gamma=args.gamma,\
                Ktrain_samples=args.K_train_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="DampingPendulum")
    parser.add_argument("--suffix",type=str,default="5_2")
    parser.add_argument("--all_loss",type=int,default=1)
    parser.add_argument("--K_train_samples",type=int,default=5000)  # 50000
    parser.add_argument("--e_loss",type=int,default=0)
    parser.add_argument("--gamma",type=float,default=0.8)
    parser.add_argument("--encode_dim",type=int,default=20)
    parser.add_argument("--layer_depth",type=int,default=3)
    args = parser.parse_args()
    main()

