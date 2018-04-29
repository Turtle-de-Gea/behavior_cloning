import numpy as np
import matplotlib.pyplot as plt

pos, theta = np.zeros(3), 0.0

traj_file = open("../data/tra.txt", "r")

stream_ = traj_file.readlines()
stream_ = [x.strip() for x in stream_] 

t1, t2 = [], []

for i in range(1, len(stream_)):
	token_ = stream_[i].split(' ')
	vx, th, dt = float(token_[0]), float(token_[1]), float(token_[2])
	print (vx, th, dt)
	delta_x = (vx * np.cos(th)) * dt;
	delta_y = (vx * np.sin(th)) * dt;
	delta_th = th * dt;

	pos = np.array([pos[0]+delta_x, pos[1]+delta_y, pos[2]+delta_th])
	t1.append(pos[0])
	t2.append(pos[1])	
		
plt.figure(1)
plt.plot(t2, t1, 'ro')
plt.show()

