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

5.0 debugging
python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_debug"

5.1 change the gamma from 0.99 to 1.00
python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_debug1"