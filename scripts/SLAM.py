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
def kalman_update(X, P, Z, pts, R):
    """
    X - state vector
    P - Covariance matrix
    Z - a LIST containing all the observed measurements
    pts - a list containing the actual positions (global, known-map locations) -- We will want to get rid of this later
    R - the measurement noise covariance matrix
    """
    print("n_msmts: ", len(Z))
    J = array([0, -1, 1, 0]).reshape(2,2)

    for i in range(len(Z)):
        zi = Z[i][0:2].reshape(2,1)  # the (known) measurement (relative position)
        p_l = pts[i].reshape(2,1)  # the (known) global position of the landmark
        p_r = X[0:2].reshape(2,1)  # the (estimated) global position of the robot
        print("zi", zi.T)
        print("pl", p_l.T)
        print("pr(est)", p_r.T)

        theta = X[2, 0]
        theta = 0
        C = array([[cos(theta), -sin(theta)],
                   [sin(theta), cos(theta)]])
        H_L = C.T

        # Delta
        delta = (p_l - p_r).reshape(2,1)  # landmark global - robot global
        z_est = C.T.dot(delta)

        # residual
        r = zi - z_est
        print("residual", r)

        # H map
        # H = zeros(shape=(2, len(P)))
        A = - C.T
        B = -C.T.dot(J.dot(delta))
        H = np.concatenate((A, B), axis=1)
        print(H)
        # innovation
        S = H.dot(P).dot(H.T) + R

        # Kalman gain
        K = P.dot(H.T).dot(np.linalg.inv(S))

        # Kalman state update
        X = X+K.dot(r)

        # Kalman covariance update
        P = P - K.dot(H).dot(P)
    return X, P
