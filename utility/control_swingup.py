import numpy as np
import scipy.linalg as linalg
from simple_pid import PID

gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5  # actually half the pole's length
polemass_length = (masspole * length)
force_mag = 10.0
tau = 0.02

lqr = linalg.solve_continuous_are


# def E(x):
# 	return 1/2 * masspole * (2 * length)**2 / 3 * x[3]**2 + np.cos(x[2]) * polemass_length * gravity


# def u(x):
# 	# print(E(x)-Ed)
# 	return 1.0 * (E(x)-Ed) * x[3] * np.cos(x[2])


H = np.array([
	[1, 0, 0, 0],
	[0, total_mass, 0, - polemass_length],
	[0, 0, 1, 0],
	[0, - polemass_length, 0, (2 * length)**2 * masspole / 3]
	])

Hinv = np.linalg.inv(H)

A = Hinv @ np.array([
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, - polemass_length * gravity, 0]
	])
B = Hinv @ np.array([0, 1.0, 0, 0]).reshape((4, 1))
Q = np.diag([0.01, 0.1, 100.0, 0.5])
R = np.array([[0.1]])

P = lqr(A, B, Q, R)
Rinv = np.linalg.inv(R)
K = - Rinv @ B.T @ P  


def ulqr(x):
	# x.shape = (5,)
	x1 = np.copy(x)
	x1 = np.array([x1[0], np.arccos(x1[1]), x1[3], x1[4]])
	x1[2] = np.sin(x1[2])
	return np.dot(K, x1)

def upid(x, p=20):
	# x = [x, cos, sin, v, w]
	# choose different control based on the angle
	# theta = np.arccos(x[1])  # in radians
	# if (0 < theta and theta < 1.57) or (4.71 < theta and theta < 6.28):
	# 	u = np.random.uniform(-1, 1)
	# else:
	# 	u =  - p * x[3]
	if np.abs(x[3]) >= 1.0:
		u = - p * x[3]
	else:
		u = p * x[3]
	return u

def rebuild_state(get_state):
	state = np.array([get_state[0], np.cos(get_state[1]), np.sin(get_state[1]), get_state[2], get_state[3]])
	return state