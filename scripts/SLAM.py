#/usr/bin/python
import numpy as np
from numpy import eye, zeros, cos, sin, array
from numpy import linalg as la

## STATE PROPAGATION STEP
def propagate_state(X, P, u, dt, R):

    M = len(X)
    N = (M-3)/3  # The number of landmarks currently in our state vector

    x = X[0:3]
    v = u[0]
    omega = u[1]

    # F is a matrix we can multiply our entire state vector by for the updates below
    # note that F mostly just Zeros out the state vector
    F = eye(M)
    F[0:3, 0:3] = array([[1, 0, -v*dt*sin(x[2])],
                         [0, 1, v*dt*cos(x[2])],
                         [0, 0, 1]])

    # G, like F, is for zeroing out terms in our state vector
    G = zeros(shape=(M, 2));
    G[0:3, 0:2] = array([[-dt*cos(x[2]), 0],
                         [-dt*sin([x[2]]), 0],
                         [0, -dt]])

    update = array([v*dt*cos(x[2]), v*dt*sin(x[2]), dt*omega]).reshape(-1,1)
    X[0:3] = x + update

    # print('p',P.shape)
    # print('f',F.shape)
    # print('g',G.shape)
    # print('r', R.shape)
    A = F.dot(P).dot(F.T)
    B = G.dot(R).dot(G.T)
    P = A+B
    assert(P.shape == (M,M))

    return X, P


## KALMAN UPDATE STEP
def kalman_update(X, P, Z, R):
    print("n_msmts: ", len(Z))

    for z in Z:
        p_r = X[0:2]
        theta = X[2]
        C = array([cos(theta), -sin(theta), sin(theta), cos(theta)]).reshape(2,2)
        H_L = C.T

        # Delta
        p_z = z[0:2]
        delta = (p_z - p_r).reshape(2,1)  # landmark global - robot global
        z_est = C.T.dot(delta)

        # residual
        r = p_z - z_est
        print('residual:', r)

        # H map

        # innovation

        # Kalman gain

        # residual

        # Kalman state update

        # Kalman covariance update


    return X, P
