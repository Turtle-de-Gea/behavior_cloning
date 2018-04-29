termianl 0
roscore

termianl 1
cd code/catkin_dir/
catkin_make
source devel/setup.bash
roslaunch behavior_cloning turtle_astra_ar.launch


termianl 2
cd code/catkin_dir/
catkin_make
source devel/setup.bash
rosrun behavior_cloning go_leader.py
or 
rosrun behavior_cloning go_follower.py
