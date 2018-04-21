import numpy as np

pos, theta = np.zeros(3), 0.0


traj_file = open("../data/traPolygon.txt", "r")

stream_ = traj_file.readlines()
stream_ = [x.strip() for x in stream_] 
for i in range(1, len(stream_)):
	token_ = stream_[i].split(' ')
	v_x, theta = float(token_[0]), float(token_[1])
	pos = np.array([pos[0]+v_x*np.cos(theta), pos[1]+v_x*np.sin(theta), pos[2]+theta])
	print pos
		
