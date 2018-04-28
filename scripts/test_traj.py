import numpy as np
from numpy import sin, cos
from matplotlib import pyplot as plt
#plt.ion()
pos_init = np.zeros(shape=(3, 1))  # column vector
pos = pos_init
trajectory = pos
# Read control data
traj_file = open("../data/traPolygon.txt", "r")
stream_ = traj_file.readlines()
stream_ = [x.strip() for x in stream_]

dt = 1/10

# f = plt.figure()
# ax = f.add_axes([-10, -10, 10, 10])
#ax = plt.gca()
#plot, = ax.plot(pos[0], pos[1])

for i in range(1, len(stream_)):
    tokens_ = stream_[i].split(' ')
    v, omega = float(tokens_[0]), float(tokens_[1])
    pos = np.array([pos[0]+v*dt*cos(pos[2]), pos[1]+v*dt*sin(pos[2]), pos[2]+dt*omega]).reshape(-1, 1)
    trajectory = np.concatenate((trajectory, pos), axis=1)
    #print(trajectory.shape)
    #plot.set_data(trajectory[0,:], trajectory[1,:])
    # ax.relim()
    # ax.autoscale_view(True,True,True)
    #plt.draw()
    #plt.pause(0.001)

plt.plot(trajectory[0,:], trajectory[1,:])
plt.show()
