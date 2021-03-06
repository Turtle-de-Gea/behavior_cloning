#! /usr/bin/env python
from __future__ import print_function, division
import sys
import os
import argparse
import cv2
import rospy
import roslib
from std_msgs.msg import String, Float32MultiArray
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Quaternion
from numpy import pi, array, sin, cos, eye, zeros, matrix, sqrt
from scipy.stats import chi2
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import time
from KF_update import kalman_update

class TrajectoryFollower:
    def __init__(self):
        self.original, self.depth = None, None
        self.bench_test, self.publish_image = False, False
        rospy.init_node('turtle_follower', anonymous=True)
        self.r = rospy.Rate(10)
        self.i = 0
        self.vel = 0.2 # 2 m/s linear veocity
        self.STATES = {'0':'INIT', '1':'TURNING', '2':'FINISHED_TURNING', '3':'MOVING', '9':'DONE'}
	self.initialized = False
        self.G_tags = {'1': (1.777, 0.9755), '2': (2.0146, 0.8662),  '3': (2.5591, 0.5296),  '4': (2.5393, -1.0148), '5': (4.4995, 0.3613) }

        self.bridge = CvBridge()
        im_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.imageCallBack, queue_size=5)
        depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depthCallBack, queue_size=5)
        tag_pose_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.tagPoseCallback, queue_size=5)
        odom_sub = rospy.Subscriber("/odom", Odometry, self.OdometryCallback, queue_size=5)

        self.target_pub = rospy.Publisher('target_info', String, queue_size=5)
        self.cmd_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=5)

        if self.publish_image:
            self.ProcessedRaw = rospy.Publisher('/follow/out_image', Image, queue_size=5)

        self.pos = zeros(2)  # robot x, y estimate
        self.theta = zeros(2)  # robot theta estimate: [radians, degrees]
        self.quat = zeros(4)
        self.P = 0.01*eye(3)  # state uncertainty
        self.R = array([[0.01, 0.001], [0.001, 0.01]])  # Measurement noise
        self.target, self.target_theta = zeros(2), [0.0, 0.0]
        self.setpoints = []
        self.sp_file = open("src/behavior_cloning/data/setpoints.txt", "r")
        self.curr_sp_ptr = 10
        self.getSetpoints()

        self.curr_time = rospy.Time.now()
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Rospy Sping Shut down")

    @property
    def X(self):
        # construct X as is needed for input into SLAM functions
        return np.append(self.pos, self.theta[0]).reshape(3, 1)

    def getSetpoints(self):
        stream_ = self.sp_file.readlines()
        stream_ = [x.strip() for x in stream_]
        for i in range(len(stream_)): #len(stream_)
            token_ = stream_[i].split(' ')
            s_p = (float(token_[0]), float(token_[1]))
            self.setpoints.append(s_p)
        self.pos[0], self.pos[1] = self.setpoints.pop(0)
        rospy.loginfo("Loaded setpoints %s", str(self.setpoints))
        rospy.loginfo("initial pos: (%s, %s)", self.pos[0], self.pos[1])
     

    def imageCallBack(self, rgb_im):
        try:
            im_array = self.bridge.imgmsg_to_cv2(rgb_im, "bgr8")
        except CvBridgeError as e:
            print(e)
        if im_array is None:
            print ('frame dropped, skipping tracking')
        else:
            self.original = np.array(im_array)

    def OdometryCallback(self, odom_data):
	self.pos[0] = odom_data.pose.pose.position.x
	self.pos[1] = odom_data.pose.pose.position.y
	self.quat = odom_data.pose.pose.orientation
	_, _, theta_t = euler_from_quaternion((self.quat.x, self.quat.y, self.quat.z, self.quat.w))
	self.theta = [theta_t, theta_t * 180 / math.pi]
	self.makemove()

    def depthCallBack(self, d_im):
        try:
            d_array = self.bridge.imgmsg_to_cv2(d_im, "32FC1")
        except CvBridgeError as e:
            print(e)
        if d_array is None:
            print ('frame dropped, skipping tracking')
        else:
            self.depth = np.array(d_array)

    def tagPoseCallback(self, msg):
        if self.original is not None and self.depth is not None:
            if msg.markers:
                self.tag_msg = msg.markers
                self.get_tag_poses()

            if self.bench_test:
                self.showFrame(self.original, 'input_image')

        if self.publish_image:
            msg_frame = CvBridge().cv2_to_imgmsg(self.original, encoding="bgr8")
            self.ProcessedRaw.publish(msg_frame)


    def get_tag_poses(self):
        if self.tag_msg is not None:
            N_tags = len(self.tag_msg)
            tag_poses, tag_orients, tag_ids, tag_globals = [], [], [], []
            for i in xrange(N_tags):
		cur_tag_id = self.tag_msg[i].id
		tag_ids.append(cur_tag_id)
		if str(cur_tag_id) in self.G_tags.keys():
		        landmark_pose = self.tag_msg[i].pose.pose.position
		        lx = landmark_pose.z
		        ly = -landmark_pose.x
		        pl = array([lx, ly]).reshape(2,1)
		        tag_poses.append(pl)
		        tag_orients.append(self.tag_msg[i].pose.pose.orientation)
		        tag_globals.append(array(self.G_tags[str(cur_tag_id)]).reshape(2,1))

            #print("Found tags ", tag_ids)
            #print("Tag globals ", tag_globals)
            #print("Tag poses: ", tag_poses)

        X, P = kalman_update(self.X, self.P, tag_poses, tag_globals, self.R)
        X = X.ravel()
        self.pos[0] = X[0]
        self.pos[1] = X[1]
        self.theta[0], self.theta[1] = X[2], X[2] * 180 / math.pi
        self.P = P
	if not self.initialized:	
		#self.target[0], self.target[1] = self.setpoints.pop(0)
		#self.target_bearing = (self.target-self.pos)
                #theta_temp = math.atan2(self.target_bearing[1], self.target_bearing[0])
                #self.target_theta =  [theta_temp, theta_temp * 180 / math.pi]
		#print (self.target_theta)
		self.initialized = True
		self.target_theta = [0.0, 0.0]
		self.theta_dis = self.theta[1] - self.target_theta[1]
		turn_val = np.abs(math.radians(self.theta_dis))
		print (turn_val, self.theta[1])
		self.curr_sp_ptr = 1

	print ('Current state: ', self.pos[0:2], self.theta[1])




    def makemove(self):
        if (self.curr_sp_ptr<9 and self.initialized):
            # calculate control input
            self.target_bearing = (self.target-self.pos)
            self.target_dis = np.linalg.norm(self.target_bearing)
            self.theta_dis = self.theta[1] - self.target_theta[1] # in degrees 
	    print (self.target, round(self.target_dis, 2), round(self.theta_dis, 2))
            base_cmd = Twist()

            # turning
            if self.curr_sp_ptr==1:
		if np.abs(self.theta_dis) > 1:
		        if np.abs(self.theta_dis) > 10:
		            turn_val = min(0.5, max(0.5, np.abs(math.radians(self.theta_dis))))
			    base_cmd.angular.z = -np.sign(self.theta_dis)*turn_val
			else:
			    turn_val = min(0.5, max(0.05, np.abs(math.radians(self.theta_dis))))
			    base_cmd.angular.z = -np.sign(self.theta_dis)*turn_val
                else:
                    #print("here with ", np.abs(self.theta_dis))
                    self.curr_sp_ptr = 2

            else:
                if(self.curr_sp_ptr==2 or np.abs(self.target_dis) > 0.1):
                    base_cmd.linear.x = min(0.2, self.target_dis)
		    if np.abs(self.theta_dis) > 10: 
			    turn_val = min(0.5, max(0.5, np.abs(math.radians(self.theta_dis))))
		    else:
			    turn_val = min(0.5, max(0.05, np.abs(math.radians(self.theta_dis))))
                    base_cmd.angular.z = -np.sign(self.theta_dis)*turn_val
                    print(base_cmd.linear.x, base_cmd.angular.z)
                    self.curr_sp_ptr = 3


                else:
                    if len(self.setpoints)==0:
                        self.curr_sp_ptr = 9
                    else:
                        self.target[0], self.target[1] = self.setpoints.pop(0)
                	self.target_bearing = (self.target-self.pos)
                	theta_temp = math.atan2(self.target_bearing[1], self.target_bearing[0])
                	self.target_theta =  [theta_temp, theta_temp * 180 / math.pi]
                        rospy.loginfo("Loaded next set-point (%s, %s) on angle %s", str(self.target[0]), str(self.target[1]), str(self.target_theta[1]))
                        self.curr_sp_ptr = 1
            self.cmd_pub.publish(base_cmd)
            self.r.sleep()

    def showFrame(self, frame, name):
        cv2.imshow(name, frame)
        cv2.waitKey(20)

if __name__ == '__main__':
        go = TrajectoryFollower()
    

