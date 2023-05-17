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
ATTENTION:
I have to block out the limitation of control signal in "/localhome/hha160/anaconda3/envs/koopman/lib/python3.7/site-packages/dmc2gym/wrappers.py" line 147 and line 149.

python Learn_Koopman_with_KlinearEig.py  --env="CartPole-dm" --suffix="dm_control"
