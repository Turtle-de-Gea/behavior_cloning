#! /usr/bin/env python
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


class TrajectoryFinder:
	def __init__(self):
		self.original, self.depth = None, None
		self.lower = np.array([100, 10, 50], dtype = "uint8") #0,48,80
		self.upper = np.array([200, 70, 100], dtype = "uint8") #20,255,255

		self.bench_test, self.publish_image = True, False
		rospy.init_node('turtle_leader', anonymous=True)

		self.bridge = CvBridge()
		im_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.imageCallBack, queue_size=5)
		depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depthCallBack, queue_size=5)
		tag_pose_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.tagPoseCallback, queue_size=5)
		odom_sub = rospy.Subscriber("/odom", Odometry, self.OdometryCallback, queue_size=5)

		self.target_pub = rospy.Publisher('target_info', String, queue_size=5)
		self.cmd_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=5)

		if self.publish_image:
			self.ProcessedRaw = rospy.Publisher('/follow/out_image', Image, queue_size=5)

		self.pos, self.theta = np.zeros(3), 0.0
		self.saver_file = open("src/behavior_cloning/data/tra.txt", "w+")
		self.curr_time = rospy.Time.now()

		try:
			rospy.spin()
		except KeyboardInterrupt:
			print("Rospy Sping Shut down")
			
		
	# for real-time testing
	def imageCallBack(self, rgb_im):
		try:
			im_array = self.bridge.imgmsg_to_cv2(rgb_im, "bgr8")
		except CvBridgeError as e:
			print(e)
		if im_array is None:
			print ('frame dropped, skipping tracking')
		else:
			self.original = np.array(im_array)



	def tagPoseCallback(self, msg):
		if self.original is not None and self.depth is not None:
			if msg.markers!=[]:
				self.tag_msg = msg.markers
				self.tag_pose, self.tag_orien = self.tag_msg[0].pose.pose.position, self.tag_msg[0].pose.pose.orientation  
				#print ("Found: tag", self.tag_msg[0].id)
				#print (self.tag_pose, self.tag_orien) 
				self.makemove()
			if self.bench_test:	
				self.showFrame(self.original, 'input_image')
				#self.showFrame(self.depth, 'input_depth')
		
		if self.publish_image:
			msg_frame = CvBridge().cv2_to_imgmsg(self.original, encoding="bgr8")
			self.ProcessedRaw.publish(msg_frame)
	
		

	def OdometryCallback(self, odom_data):
		vel_x = odom_data.twist.twist.linear.x
		self.theta = odom_data.twist.twist.angular.z
		c, s = np.cos(self.theta), np.sin(self.theta)
		Rot = np.array(((c,-s, 0), (s, c, 0), (0, 0, 1)))
		t = np.array([vel_x, 0, 0])
		self.pos = self.pos + np.dot(Rot, t)
		tra_info = str(vel_x) + ' ' + str(self.theta) + '\n'
		if ((rospy.Time.now() - self.curr_time).to_sec() > 0.04):
			self.curr_time = rospy.Time.now()
			print ('saving at '+ str(self.curr_time.to_sec()) + ' : ' +  tra_info) 
			#self.saver_file.write(tra_info)



	# for real-time testing
	def depthCallBack(self, d_im):
		try:
			d_array = self.bridge.imgmsg_to_cv2(d_im, "32FC1")
		except CvBridgeError as e:
			print(e)
		if d_array is None:
			print ('frame dropped, skipping tracking')
		else:
			self.depth = np.array(d_array)
			
	

	def localize(self):
		if self.tag_msg != None:
			N_tags = len(self.tag_msg) 
			tag_poses, tag_orients, tag_ids = [], [], []  	
			for i in xrange(N_tags):
				tag_poses.append(self.tag_msg[i].pose.pose.position)
				tag_orients.append(self.tag_msg[i].pose.pose.orientation)
				tag_ids.append(self.tag_msg[i].id)
			print ("Found tags ", tag_ids)
			print ("Tag poses: ", tag_poses)
				

	def makemove(self):
		if self.tag_pose != None:
			base_cmd = Twist()
			base_cmd.linear.x = (self.tag_pose.z - 0.5)
			base_cmd.angular.z = -self.tag_pose.x*4
			tra_info = str(base_cmd.linear.x) + ' ' + str(base_cmd.angular.z) + '\n'
			self.saver_file.write(tra_info)
			self.cmd_pub.publish(base_cmd)



	##########################################################################
	###   For bench testing with dataset images ###############################
	def showFrame(self, frame, name):
		cv2.imshow(name, frame)
		cv2.waitKey(20)

	# stream images from directory Dir_
	def image_streamimg(self, Dir_):
		from eval_utils import filter_dir
		dirFiles = filter_dir(os.listdir(Dir_))
		for filename in dirFiles:
			self.original = cv2.imread(Dir_+filename)
			self.ImageProcessor()
	####################################################################################
