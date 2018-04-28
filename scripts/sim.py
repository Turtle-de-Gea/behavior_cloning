#/usr/bin/python
"""
Simulator for running SLAM -- simulate a simple 2d environment
"""
from numpy import pi, array, sin, cos, eye, zeros, matrix, sqrt
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
import time

def get_covariance_ellipse(mu, sig, conf):
    n_pts = 100
    D, V = np.linalg.eig(sig);
    scale = chi2.ppf(conf, len(mu))
    axis_1 = V[:, 0]*sqrt(scale * D[0])
    axis_2 = V[:, 1]*sqrt(scale * D[1])
    axis_1 = axis_1.reshape(-1,1)
    axis_2 = axis_2.reshape(-1,1)
    points = zeros(shape=(2, n_pts));
    for i in range(n_pts):
        th = i / (n_pts-1) * 2 * pi
        pt = sin(th) * axis_2 + cos(th) *axis_1 + mu
        points[:, i] = pt[:,0]
    return points


plt.ion()
n_steps = 50
update_rate = 5
n_map_pts = 80;
map_lb = array([-8, -3])
map_ub = array([8, 8])
max_msmt_dist = 2
circle_rad = 5
dt=0.1

nominal_u = array([[circle_rad *pi / n_steps /dt],[pi/n_steps/dt]])

Sig_n = array([[0.1, 0.001],[0.001, 0.1]])
sig_pt = array([[0.01, 0.001], [0.001, 0.01]])

## Initial state (True and estimate)
x_true = array([circle_rad, 0, pi/2]).reshape(3, 1)
x_t = array([circle_rad, 0, pi/2]).reshape(3, 1)
Sig_t = 0.01*np.eye(3)

## Generate Map Points
map_pts = np.random.rand(2, n_map_pts)

for i in range(n_map_pts):
    map_pts[:, i] = map_lb + np.multiply(map_pts[:, i], map_ub - map_lb)

seen_pt = zeros(shape=(1, n_map_pts))

#fig = plt.figure()
#ax = fig.add_axes([map_lb[0], map_lb[1], map_ub[0], map_ub[1]])
# ax.axis('equal')
ax = plt.gca()
plt.axis((-8, 8, -3, 8))
plot, = ax.plot(x_true[0], x_true[1])
print(x_true[0:-1, 0])
elipse = get_covariance_ellipse(x_true[0:-1, 0].reshape(-1,1), Sig_n, 0.9)
print(elipse)
plot2, = ax.plot(elipse[0,:], elipse[1,:], 'r')

X_true = zeros(shape=(3,n_steps))
cov_pts_x = []
trajectory = x_true

for step in range(n_steps):
    # propagate true state
    u_true = nominal_u + np.random.multivariate_normal([0,0], Sig_n).reshape(-1,1)
    v, omega = u_true
    x_true_new = array([x_true[0]+v*dt*cos(x_true[2]), x_true[1]+v*dt*sin(x_true[2]), x_true[2] + dt*omega])
    x_true = x_true_new
    trajectory = np.concatenate((trajectory, x_true), axis=1)
    # print(trajectory.shape)
    # print('x_true', x_true)
    plot.set_data(trajectory[0,:], trajectory[1,:])

    elipse = get_covariance_ellipse(x_true[0:-1, 0].reshape(-1,1), Sig_n, 0.9)
    plot2.set_data(elipse[0,:], elipse[1,:])

    plt.draw()
    plt.pause(0.1)
    # X_true[:, step] = x_true

plt.plot(trajectory[0,:], trajectory[1,:])
plt.show()
