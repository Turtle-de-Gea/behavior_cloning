#/usr/bin/python
"""
Simulator for running SLAM -- simulate a simple 2d environment
"""
from numpy import pi, array, sin, cos, eye, zeros, matrix, sqrt
from scipy.stats import chi2
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import time
from SLAM import propagate_state

def get_covariance_ellipse(mu, sig, conf):
    print("SIG IS :", sig)
    n_pts = 100
    D, V = np.linalg.eig(sig);
    print("D", D)
    print("V", V)
    scale = chi2.ppf(conf, len(mu))
    axis_1 = V[:, 0]*sqrt(scale * D[0])
    axis_2 = V[:, 1]*sqrt(scale * D[1])
    axis_1 = axis_1.reshape(2,1)
    axis_2 = axis_2.reshape(2,1)
    # print('ax1', axis_1)
    # print('ax2',axis_2)
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
P = 0.01*np.eye(3)

## Generate Map Points
map_pts = np.random.rand(2, n_map_pts)

for i in range(n_map_pts):
    map_pts[:, i] = map_lb + np.multiply(map_pts[:, i], map_ub - map_lb)

seen_pt = zeros(n_map_pts)


## INITIALIZE PLOTTING SHIT
ax = plt.gca()
plt.axis((-8, 8, -3, 8))
# robot pose
plot, = ax.plot(x_true[0], x_true[1])
# estimation ellipse
elipse = get_covariance_ellipse(x_true[0:-1, 0].reshape(-1,1), Sig_n, 0.9)
plot2, = ax.plot(elipse[0,:], elipse[1,:], 'r')
# landmarks
plot3 = ax.scatter(map_pts[0,:], map_pts[1,:])
print(plot3)

cov_pts_x = []
trajectory = x_true

X = x_true

for step in range(n_steps):
    ## PROPAGATE TRUE STATE
    u_true = nominal_u + np.random.multivariate_normal([0,0], Sig_n).reshape(-1,1)
    v, omega = u_true
    x_true_new = array([x_true[0]+v*dt*cos(x_true[2]), x_true[1]+v*dt*sin(x_true[2]), x_true[2] + dt*omega])
    x_true = x_true_new
    trajectory = np.concatenate((trajectory, x_true), axis=1)

    ## PROPAGATE STATE ESTIMATE
    X, P = propagate_state(X, P, nominal_u, dt, Sig_n)

    ## SENSOR UPDATE
    all_msmts = []
    if step % update_rate ==0:
        # OBSERVE
        for i in range(n_map_pts):
            if norm(map_pts[:, i] - x_true[0:1, 0]) < max_msmt_dist:
                seen_pt[i] = 1;
                msmt = None
                all_msmts.append(msmt) # append the measurement
        # EKF update
        pass #TODO

    ## PLOT SHIT
    #> plot true trajectory
    plot.set_data(trajectory[0,:], trajectory[1,:])

    #> replot landmarks



    #> plot estimate ellipse
    elipse = get_covariance_ellipse(X[0:2, 0].reshape(-1,1), P[0:2, 0:2], 0.9)
    plot2.set_data(elipse[0,:], elipse[1,:])

    plt.draw()
    plt.pause(0.1)

plt.plot(trajectory[0,:], trajectory[1,:])
plt.show()
