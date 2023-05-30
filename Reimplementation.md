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

ATTENTION for shortening time, shrink the size 100 times:
train_steps = 200000, Ktrain_samples=50000,Ktest_samples = 20000

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_test" 

4. In the new branch "experiment", remove the gif expression and use the original environment (not camera) to generate data

tmux a -t KoopTest

ATTENTION: restored the original hyperparameters:
train_steps = 200000, Ktrain_samples=50000,Ktest_samples = 20000

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control_test" 

Control Performances: 
ATTENTION: the control must be implemented in the original (not tmux mode)
when 
Q[1,1] = 0.01 Q[2,2] = 5.0 Q[3,3] = 5.0 Q[4,4] = 0.01
rewards/200 steps = 5.489
