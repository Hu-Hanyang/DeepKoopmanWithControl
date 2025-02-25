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
import os

sys.path.append("../franka")

class RBFLiftFunc():
    def __init__(self,env_name,Nstate,udim,Nrbf,observation_space,type="thinplate",center=None) -> None:
        self.env_name = env_name
        self.Nstate = Nstate
        self.udim = udim        
        self.Nrbf = Nrbf
        self.NKoopman = self.Nstate+self.Nrbf
        lift_val = 4
        if env_name.startswith("Pendulum"):
            lift_val = 1
        elif env_name.startswith("Reacher"):
            lift_val = 8
        elif env_name.startswith("DoublePendulum"):
            lift_val = 8
        self.lift_low = np.clip(observation_space.low,-lift_val,lift_val)
        self.lift_high = np.clip(observation_space.high,-lift_val,lift_val)
        self.type = type
        if center is None:
            self.center = np.random.uniform(low=self.lift_low,high=self.lift_high,size=(self.Nrbf,self.Nstate)).T
        else:
            self.center = center
            
    def Psi_s(self,s):
        #s (N,Nstate)
        s = s.reshape(-1,self.Nstate)
        N,_ = s.shape
        psi = np.zeros([N,self.NKoopman])
        psi[:,:self.Nstate] = s  
        lift_space = rbf(s.T,self.center,self.type).T
        psi[:,self.Nstate:] = lift_space
        return psi  

    def Psi_su(self,s,u):
        s = s.reshape(-1,self.Nstate)
        u = u.reshape(-1,self.udim)
        N,_ = s.shape
        psi = np.zeros([N,self.NKoopman+self.udim])
        psi[:,:self.NKoopman] = self.Psi_s(s)
        psi[:,self.NKoopman:] = u
        return psi


class DerivativeLiftFunc():
    def __init__(self,env_name,Nstate,udim) -> None:
        self.env_name = env_name
        self.Nstate = Nstate
        self.udim = udim
        if self.env_name.startswith("DampingPendulum"):
            self.g = 9.8
            self.l = 1.0
            self.m = 1.0    
            self.b = 1
            self.NKoopman = self.Nstate+2
        elif self.env_name.startswith("Pendulum"):
            self.g = 10.0 # gravitational constant
            self.l = 1.0 # pendulum length
            self.m = 1.0
            self.NKoopman = self.Nstate+2
        elif self.env_name.startswith("CartPole"):
            self.g = 9.8 
            self.mc = 1.0
            self.mp = 0.1
            self.mt = self.mc+self.mp
            self.l = 0.5
            self.lp = self.mp*self.l
            self.NKoopman = self.Nstate+2
        elif self.env_name.startswith("MountainCarContinuous"):
            self.NKoopman = self.Nstate+2

    def Psi_s(self,s):
        psi = np.zeros(self.NKoopman)
        s = s.reshape(self.Nstate)
        # u = u.reshape(self.udim)
        psi[:self.Nstate] = s
        # psi[-self.udim:] = u
        if self.env_name.startswith("DampingPendulum"):
            theta,dtheta = s
            psi[self.Nstate] = -self.g/self.l * np.sin(theta) -self.b*self.l*dtheta/self.m
            psi[self.Nstate+1] = -self.g/self.l * np.cos(theta)*dtheta -self.b*self.l*psi[self.Nstate]/self.m
        elif self.env_name.startswith("Pendulum"):
            theta,dtheta = s
            psi[self.Nstate] = 3.0*self.g/(2.0*self.l) * np.sin(theta)
            psi[self.Nstate+1] = 3.0*self.g/(2.0*self.l) * np.cos(theta) * dtheta
        elif self.env_name.startswith("CartPole"):
            x, x_dot, theta, theta_dot = s
            sintheta = np.sin(theta)
            costheta = np.cos(theta)
            temp = (self.lp*theta_dot**2*sintheta)/self.mt
            psi[self.Nstate] = (self.g*sintheta-costheta*temp)/(self.l*(4.0/3.0-self.mp*costheta**2/self.mt))
            psi[self.Nstate+1] = temp - self.lp*psi[4]*costheta/self.mt
        elif self.env_name.startswith("MountainCarContinuous"):
            x,dx = s
            psi[self.Nstate] = 0.0025*np.cos(3*x)
            psi[self.Nstate+1] = -0.0025*np.sin(3*x)*dx
        return psi

    def Psi_su(self,s,u):
        psi = np.zeros(self.NKoopman+self.udim)
        s = s.reshape(self.Nstate) 
        u = u.reshape(self.udim)
        psi[:self.NKoopman] = self.Psi_s(s)
        psi[self.NKoopman:] = u
        return psi

    
class DoublePendulum():
    def __init__(self) -> None:
        self.g = 9.8
        self.l1 = 1.0
        self.l2 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.dt = 0.01
        self.s0 = np.zeros(4)
        self.Nstates = 4
        self.umin = np.array([-6,-6])
        self.umax = np.array([6,6])
        self.b = 2
        low = np.array([-np.pi,-np.pi,-8,-8],dtype=np.float32)
        self.observation_space = spaces.Box(low,-low,dtype=np.float32)

    def reset(self):
        th0 = random.uniform(-0.1*np.pi, 0.1*np.pi)
        dth0 = random.uniform(-1, 1)
        th1 = random.uniform(-0.1*np.pi, 0.1*np.pi)
        dth1 = random.uniform(-1, 1)
        self.s0 = np.array([th0,th1,dth0, dth1])
        return self.s0
    
    def reset_state(self,s):
        self.s0 = s
        return self.s0

    def dynamics(self,y,t, u):
        th1,th2,dth1,dth2 = y
        u = np.array(u).reshape(2,1)
        f = np.zeros(4)
        g = self.g
        l1 = self.l1
        l2 = self.l2
        m1 = self.m1
        m2 = self.m2
        c2 = np.cos(th2)
        s2 = np.sin(th2)
        M = np.zeros((2,2))
        M[0,0] = m1*l1**2+m2*(l1**2+2*l1*l2*c2+l2**2)
        M[0,1] = m2*(l1*l2*c2+l2**2)
        M[1,0] = m2*(l1*l2*c2+l2**2)
        M[1,1] = m2*l2**2
        C = np.zeros((2,1))
        C[0,0] = -m2*l1*l2*s2*(2*dth1*dth2+dth2**2)
        C[1,0] = m2*l1*l2*dth1**2*s2
        G = np.zeros((2,1))
        G[0,0] = (m1+m2)*l1*g*np.cos(th1)+m2*g*l2*np.cos(th1+th2)
        G[1,0] = m2*g*l2*np.cos(th1+th2)
        Minv = scipy.linalg.pinv2(M)
        ddth = np.dot(Minv,(u-C-G)).reshape(-1)
        f[0] = dth1
        f[1] = dth2
        f[2] = ddth[0]
        f[3] = ddth[1]
        return f
        
    def step(self,u):
        sn = odeint(self.dynamics, self.s0, [0, self.dt], args=(u,))
        self.s0  = sn[-1,:]
        r =0
        done = False
        return self.s0 ,r,done,{}

class SinglePendulum():
    def __init__(self) -> None:
        self.g = 9.8
        self.l = 1.0
        self.m = 1.0
        self.dt = 0.02
        self.s0 = np.zeros(2)
        self.Nstates = 2
        self.umin = np.array([-8])
        self.umax = np.array([8])
        self.b = 1
        low = np.array([-np.pi,-8],dtype=np.float32)
        self.observation_space = spaces.Box(low,-low,dtype=np.float32)

    def reset(self):
        th0 = random.uniform(-0.1*np.pi, 0.1*np.pi)
        dth0 = random.uniform(-1, 1)
        self.s0 = np.array([th0, dth0])
        return self.s0
    
    def reset_state(self,s):
        self.s0 = s
        return self.s0

    def single_pendulum(self,y,t, u):
        theta, dtheta = y
        # f = asarray([dtheta, -g/l * sin(theta) +  u*cos(theta)/(m*l)])
        f = np.asarray([dtheta, -self.g/self.l * np.sin(theta) -self.b*self.l*dtheta/self.m+  np.cos(theta)*u/(self.m*self.l)])
        return f
        
    def step(self,u):
        u = np.array(u).reshape(1)
        sn = odeint(self.single_pendulum, self.s0, [0, self.dt], args=(u[0],))
        self.s0  = sn[-1,:]
        r =0
        done = False

        return self.s0 ,r,done,{}

def FrankaObs(o):
    return np.concatenate((o[:3],o[7:]),axis=0)

class data_collecter():
    def __init__(self,env_name) -> None:
        self.env_name = env_name
        np.random.seed(2022)
        random.seed(2022)
        if self.env_name.startswith("DampingPendulum"):
            self.env = SinglePendulum()
            self.Nstates = self.env.Nstates
            self.umax = self.env.umax
            self.umin = self.env.umin
            self.udim = 1
        elif self.env_name.startswith("Snake"):
            self.Nstates = 12
            self.udim = 5
            data = np.load("/mnt/d/github/DeepKoopmanWithControl/Data/SnakeData.npy")
            mean = np.mean(data.reshape(-1,18),axis=0)
            std = np.std(data.reshape(-1,18),axis=0)
            self.data = (data-mean)/std
        elif self.env_name.startswith("DoublePendulum") or self.env_name.startswith("TwoLinkRobot"):
            self.env = DoublePendulum()
            self.Nstates = self.env.Nstates
            self.umax = self.env.umax
            self.umin = self.env.umin
            self.udim = 2            
        elif self.env_name.endswith("Franka"):
            from franka_env import FrankaEnv
            self.env =  FrankaEnv(render = False)
            self.Nstates = 17
            self.uval = 0.12
            self.udim = 7
            self.reset_joint_state = np.array(self.env.reset_joint_state)
        elif self.env_name.endswith("FrankaForce"):
            from franka_env_force import FrankaEnv
            self.env =  FrankaEnv(render = False)
            self.Nstates = 17
            self.uval = 20
            self.udim = 7
            self.reset_joint_state = np.array(self.env.reset_joint_state)
        elif self.env_name == "CartPole-dm":
            # add the CartPole from dm_control
            print(f"The environment is the CartPole from dm_control.")
            self.env = dmc2gym.make(domain_name='cartpole', task_name='swingup', seed=2022, from_pixels=False)  # seed=2022, visualize_reward=False, from_pixels=True
            self.udim = self.env.action_space.shape[0]
            self.Nstates = 5  # 5
            self.umin = self.env.action_space.low  # -1
            self.umax = self.env.action_space.high  # +1
        elif self.env_name == "Cheetah-dm":
            print(f"The environment is the dm_control CartPole with state.")
            self.env = dmc2gym.make(domain_name="cheetah", task_name="run", seed=2022, from_pixels=False)  # seed=2022, visualize_reward=False, from_pixels=True
            self.udim = self.env.action_space.shape[0]
            self.Nstates = self.env.observation_space.shape[0]
            self.umin = self.env.action_space.low  # -1
            self.umax = self.env.action_space.high  # +1           
        else:  # CartPole-v1 from gym
            self.env = gym.make(env_name)
            self.env.seed(2022)
            self.udim = self.env.action_space.shape[0]
            self.Nstates = self.env.observation_space.shape[0]
            self.umin = self.env.action_space.low
            self.umax = self.env.action_space.high
        if not self.env_name.endswith("Snake"):
            # self.observation_space = self.env.observation_space
            self.env.reset()
            if self.env_name == "CartPole-dm":
                self.observation_space = self.env.observation_space
                print(f"The observation space is {self.observation_space}")
                self.dt = 0.02  # from gym CartPole self.tau, seconds between state updates
            else:
                self.observation_space = self.env.observation_space
                self.dt = 0.02  # self.env.dt

    def random_state(self):
        if self.env_name.startswith("DampingPendulum"):
            th0 = random.uniform(-2*np.pi, 2*np.pi)
            dth0 = random.uniform(-8, 8)
            s0 = np.array([th0, dth0])
        elif self.env_name.startswith("Pendulum"):
            th0 = random.uniform(-2*np.pi, 2*np.pi)
            dth0 = random.uniform(-8, 8)
            s0 = [th0, dth0]  
        elif self.env_name.startswith("CartPole"):
            x0 = random.uniform(-4,4)
            dx0 = random.uniform(-8,8)
            th0 = random.uniform(-0.418, 0.418)
            dth0 = random.uniform(-8, 8)
            s0 = [x0,dx0,th0,dth0]
        elif self.env_name.startswith("MountainCarContinuous"):
            x0 = random.uniform(-0.1, 0.1)
            th0 = random.uniform(-0.5, 0.5)
            s0 = [x0, th0]  
        elif self.env_name.startswith("InvertedDoublePendulum"):
            x0 = random.uniform(-0.1,0.1)
            th0 = random.uniform(-0.3,0.3)
            th1 = random.uniform(-0.3,0.3)
            dx0 = random.uniform(-1,1)
            dth0 = random.uniform(-6,6)
            dth1 = random.uniform(-6,6)
            s0 = np.array([x0,th0,th1,dx0,dth0,dth1]) 
        return np.array(s0)

    def collect_koopman_data(self, traj_num, steps, mode="train"):  # add controller method as input
        # traj_num = 50000, steps = 15
        train_data = np.empty((steps+1,traj_num,self.Nstates+self.udim))
        if self.env_name.startswith("Franka"):
            for traj_i in range(traj_num):
                noise = (np.random.rand(7)-0.5)*2*0.2
                joint_init = self.reset_joint_state+noise
                joint_init = np.clip(joint_init,self.env.joint_low,self.env.joint_high)
                s0 = self.env.reset_state(joint_init)
                s0 = FrankaObs(s0)
                u10 = (np.random.rand(7)-0.5)*2*self.uval
                train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
                for i in range(1,steps+1):
                    s0 = self.env.step(u10)
                    s0 = FrankaObs(s0)
                    u10 = (np.random.rand(7)-0.5)*2*self.uval
                    train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)            
        elif self.env_name.startswith("Snake"):
            if mode.startswith("train"):
                real_data = self.data[:500,:,:]
            elif mode.startswith("eval"):
                real_data = self.data[500:700,:,:]
            else:
                real_data = self.data[700:,:,:]
            traj,traj_len,dim = real_data.shape
            train_data = []
            for i in range(traj):
                traj_data = real_data[i,:,:]
                end_index = np.where(traj_data[:,-1]==1)[0]
                if end_index.shape[0]>0:
                    end_index = end_index[0]
                else:
                    end_index = traj_len-1
                traj_num = (end_index+1)//(steps+1)
                train_data_now = np.empty((steps+1,traj_num,17))
                for j in range(traj_num):
                    train_data_now[:,j,:] = traj_data[j*(steps+1):(j+1)*(steps+1),:17]
                train_data.append(train_data_now)
            train_data = np.concatenate(train_data,axis=1)   
        elif self.env_name == "CartPole-dm" :
            print("Generate Koopman data for CartPole-dm using PPO controller now!")
            ppo_controller = PPO(state_dim=5, action_dim=1, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2, has_continuous_action_space=True)
            random_seed = 0             #### set this to load a particular checkpoint trained on random seed
            run_num_pretrained = 200000      #### set this to load a particular checkpoint num
            directory = "/localhome/hha160/projects/DeepKoopmanWithControl/controller" + '/' 
            checkpoint_path = directory + "PPO_{}_{}_{}.pth".format("CartPole-dm", random_seed, run_num_pretrained)
            # print("loading network from : " + checkpoint_path)
            ppo_controller.load(checkpoint_path)
            for traj_i in range(traj_num):
                # if traj_i < (traj_num / 3) :
                #     run_num_pretrained = 3e5
                # elif (traj_num / 3) <= traj_i < (2*traj_num / 3):
                #     run_num_pretrained = 1800000
                # else:
                #     run_num_pretrained = 3000000
                # load PPO controller
                
                # frames = []
                # duration = 0.2
                s0 = self.env.reset()
                # frames.append(Image.fromarray(image0.transpose(1, 2, 0), "RGB"))
                # s0 = rebuild_state(self.env.physics.get_state()) # s0.shape = (5,)
                # s0 = self.random_state()
                u10 = ppo_controller.select_action(s0)  # np.random.uniform(self.umin, self.umax)
                # print(u10)
                # self.env.reset_state(s0)
                train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
                for i in range(1,steps+1):
                    s0, r, done,_ = self.env.step(u10)
                    # frames.append(Image.fromarray(next_image.transpose(1, 2, 0), "RGB"))
                    # s0 = rebuild_state(self.env.physics.get_state())
                    u10 = ppo_controller.select_action(s0)  # np.random.uniform(self.umin, self.umax)
                    train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
                if (traj_i+1) % 100 == 0:
                    print(f"Has generated {traj_i} trajectories.")
                # path = f'/localhome/hha160/projects/DeepKoopmanWithControl/dm_train_data/under_mixed_PPO_final_{steps}'
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # if traj_i % 1000 == 0:
                #     imageio.mimsave(f'{path}/traj{traj_i}.gif', frames, duration=duration)
        elif self.env_name == "Cheetah-dm":
            print(f"Generate invisible training data for {self.env_name} now!")
            for traj_i in range(traj_num):
                # frames = []
                # duration = 0.2
                # image0 = self.env.reset()
                s0 = self.env.reset()
                # frames.append(Image.fromarray(image0.transpose(1, 2, 0), "RGB"))
                # s0 = self.env.physics.get_state() # s0.shape = (17,)
                u10 = np.random.uniform(self.umin, self.umax)  # for PPO: self.controller.select_action(s0); others: self.controller(s0)
                train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
                for i in range(1, steps+1):
                    s0, r, done,_ = self.env.step(u10)
                    # frames.append(Image.fromarray(next_image.transpose(1, 2, 0), "RGB"))
                    # s0 = self.env.physics.get_state()
                    u10 = np.random.uniform(self.umin, self.umax)  # for PPO: self.controller.select_action(s0); others: self.controller(s0)
                    train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
                # path = f'/localhome/hha160/projects/DeepKoopmanWithControl/dm_train_data/{self.env_name}_random_{steps}'
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # imageio.mimsave(f'{path}/traj{traj_i}.gif', frames, duration=duration)
                
        else:  # "CartPole"
            for traj_i in range(traj_num):
                s0 = self.env.reset()
                # s0 = self.random_state()
                u10 = np.random.uniform(self.umin, self.umax)
                # self.env.reset_state(s0)
                train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
                for i in range(1,steps+1):
                    s0,r,done,_ = self.env.step(u10)
                    u10 = np.random.uniform(self.umin, self.umax)
                    train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
        return train_data

    def collect_detivative_data(self,traj_num,steps):
        train_data = np.empty((steps+1,traj_num,self.Nstates+self.udim))
        for traj_i in range(traj_num):
            # s0 = self.env.reset()
            s0 = self.random_state()
            u10 = np.random.uniform(self.umin, self.umax)
            self.env.reset_state(s0)
            # print(s0,np.array(u10))
            # print(s0,u10)
            train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
            for i in range(1,steps+1):
                s0,r,done,_ = self.env.step(u10)
                u10 = np.random.uniform(self.umin, self.umax)
                train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
        return train_data
