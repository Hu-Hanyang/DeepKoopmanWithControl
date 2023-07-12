# Test
python Learn_Koopman_with_KlinearEig.py  # it works

# Others
Add setup.py file to add the python path.

# DampingPendulum
python Learn_Koopman_with_KlinearEig.py  # default method, in paper is DKUC(KPUC)

# "CartPole-v1"
ATTENTION: the version of gym could not be too new due to the difference of functions like "env.step()".

Replace the gym env file with files in folder ./gym_env/.  # done, gym==0.22.0

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-v1" --suffix="test"

tensorboard --logdir=Data/test

# "CartPole-dm"
tmux: KoopCartpoleDM

1. run with original codes

ATTENTION:
I have to block out the limitation of control signal in "/localhome/hha160/anaconda3/envs/koopman/lib/python3.7/site-packages/dmc2gym/wrappers.py" line 147 and line 149.

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control"

tensorboard --logdir=Data/dm_control

2. change the method collect_koopman_data in Utility.py to generate good control inputs

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_u"

change the K to:
tmux a -t KoopTest
K = - Rinv @ B.T @ P  # control_swingup.py, line 45
python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_u2"

3. Use the pid controller  (control_swingup.py) and save the training data as images and gif

tmux a -t KoopTest

change the Ksteps = 15 to 250

ATTENTION for shortening time, shrink the size 100 times:
train_steps = 200000, Ktrain_samples=50000,Ktest_samples = 20000

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_test" 

change the Ksteps = 15 to 250

ATTENTION: change the  hyperparameters: train_steps = 20000, Ktrain_samples=5000, Ktest_samples = 2000

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_test1" 


4. Use PPO controller to generate the training and testing data

4.1 In the main branch, terminal 1:Python (train)
Use the pure PPO models to generate the training data.

Shrink the data size: train_steps = 20000; Ktrain_samples=5000; Ktest_samples = 2000; Ksteps = 300 

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_PPO"

Notice: the training data is saved in "dm_train_data/under_PPO_control_300" 

4.2 In the main branch, terminal 2 Python (train)
Use the mixed PPO models (200000) to generate the training data.

Shrink the data size: train_steps = 20000; Ktrain_samples=5000; Ktest_samples = 2000; Ksteps = 300 

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_mixed_PPO"

Notice: the training data is saved in "dm_train_data/under_mixed_PPO_control_300" 

<!-- 5.0 debugging
python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_debug" -->
<!-- 
6.0 train the cheetah with 
train (train_steps = 100000, envode_dim = 30, Ktrain_samples = 10000, Ktest_samples = 5000)
python Learn_Koopman_with_KlinearEig.py  --env="Cheetah-dm" --suffix="dm_control_cheetah1" -->

Final: saving model during training
train_steps = 20000, Ktrain_samples=5000, Ktest_samples = 2000, Ksteps = 300
python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_mixed_PPO_final"

5th June tasks:
Traing model1 (step 2000th): KoopmanU_CartPole-dmlayer3_edim20_eloss0_gamma0.8_aloss1_samples50001999th_step.pth  
Best value: 19.15
Best para: 
{'a': 0.08, 'b': 85.32, 'c': 2.57, 'd': 0.13, 'e': 0.04}
Manual value1: 18.86
Manual para1:
{'a': 0.01, 'b': 10.0, 'c': 10.0, 'd': 0.01, 'e': 0.01}
Manual value2: 17.84
Manual para2:
{'a': 10.0, 'b': 50.0, 'c': 50.0, 'd': 1.0, 'e': 1.0}

Traing model2 (step 4000th): KoopmanU_CartPole-dmlayer3_edim20_eloss0_gamma0.8_aloss1_samples50003999th_step.pth  
Best value: 19.49
Best para: 
{'a': 10.33, 'b': 71.31, 'c': 0.03, 'd': 0.14, 'e': 0.06}
Manual value1: 13.24
Manual para1:
{'a': 0.01, 'b': 10.0, 'c': 10.0, 'd': 0.01, 'e': 0.01}
Manual value2: 12.38
Manual para2: 12.38
{'a': 10.0, 'b': 50.0, 'c': 50.0, 'd': 1.0, 'e': 1.0}

Traing model3 (step 6000th): KoopmanU_CartPole-dmlayer3_edim20_eloss0_gamma0.8_aloss1_samples50005999th_step.pth  
Best value: 32.63
Best para: 
{'a': 18.46, 'b': 68.45, 'c': 1.75, 'd': 10.79, 'e': 0.01}
Manual value1: 27.91
Manual para1:
{'a': 0.01, 'b': 10.0, 'c': 10.0, 'd': 0.01, 'e': 0.01}
Manual value2: 20.42
Manual para2:
{'a': 10.0, 'b': 50.0, 'c': 50.0, 'd': 1.0, 'e': 1.0}

Traing model4 (step 8000th): KoopmanU_CartPole-dmlayer3_edim20_eloss0_gamma0.8_aloss1_samples50007999th_step.pth  
Best value: 39.71
Best para: 
{'a': 38.13, 'b': 91.22, 'c': 78.16, 'd': 1.44, 'e': 0.09}
Manual value1: 27.91
Manual para1:
{'a': 0.01, 'b': 10.0, 'c': 10.0, 'd': 0.01, 'e': 0.01}
Manual value2: 20.42
Manual para2:
{'a': 10.0, 'b': 50.0, 'c': 50.0, 'd': 1.0, 'e': 1.0}

Final model: "KoopmanU_CartPole-dmlayer3_edim20_eloss0_gamma0.8_aloss1_samples500019999th_step.pth"
Best value: 57.00
Best para: 
{'a': 84.12, 'b': 62.07, 'c': 65.79, 'd': 0.04, 'e': 0.04}
Manual value1: 33.10
Manual para1:
{'a': 0.01, 'b': 10.0, 'c': 10.0, 'd': 0.01, 'e': 0.01}
Manual value2: 40.00
Manual para2:
{'a': 10.0, 'b': 50.0, 'c': 50.0, 'd': 1.0, 'e': 1.0}
Manual value3: 21.41
Manual para3:
{'a': 10.0, 'b': 60.0, 'c': 60.0, 'd': 0.01, 'e': 0.01}
Manual value4: 39.00
Manual para4:
{'a': 10.0, 'b': 60.0, 'c': 60.0, 'd': 1.0, 'e': 1.0}
Manual value5: 37.61
Manual para5:
{'a': 10.0, 'b': 60.0, 'c': 60.0, 'd': 10, 'e': 10}
Manual value6: 37.82
Manual para6:
{'a': 10.0, 'b': 60.0, 'c': 60.0, 'd': 10, 'e': 10 with other diagnals=0.1} 