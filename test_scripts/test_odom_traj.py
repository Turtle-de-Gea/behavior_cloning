import numpy as np
import matplotlib.pyplot as plt

pos, theta = np.zeros(2), 0.0

traj_file = open("../data/odoms_pol.txt", "r")

stream_ = traj_file.readlines()
stream_ = [x.strip() for x in stream_] 

t1, t2 = [], []

for i in range(1, len(stream_)):
	token_ = stream_[i].split(' ')
	pc = np.array([float(token_[0]), float(token_[1])])
	pos = pc 
	print pos
	t1.append(pos[0])
	t2.append(pos[1])	
		
plt.figure(1)
plt.plot(t2, t1, 'ro')
plt.show()
