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


"""
This class contain functionalities for following a tag and recording the
corresponding odometry data [x, y] as trajectory
"""

class TrajectoryFinder:
	def __init__(self):
		self.original, self.depth = None, None
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
		self.target = np.array([1.92, 0, 0])
		self.saver_file = open("src/behavior_cloning/data/cmds.txt", "w+")
		self.odom_file = open("src/behavior_cloning/data/odoms.txt", "w+")
		self.curr_time = rospy.Time.now()

		self.rate = rospy.Rate(10)

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
		self.pos[0] = odom_data.pose.pose.position.x
		self.pos[1] = odom_data.pose.pose.position.y
		self.pos[2] = odom_data.pose.pose.position.z
		#print self.pos
		#turtle_y = odom_data.pose.pose.position.y
		odo_info = str(self.pos[0]) + ' ' + str(self.pos[1]) + ' ' + str(self.pos[2])+'\n'
		self.odom_file.write(odo_info)



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
				

	def makemove(self):
		if self.tag_pose != None:
			base_cmd = Twist()
			base_cmd.linear.x = (self.tag_pose.z - 0.5)
			base_cmd.angular.z = -self.tag_pose.x*4			
			
			dt = (rospy.Time.now() - self.curr_time).to_sec()
			self.curr_time = rospy.Time.now()
			#self.cmd_pub.publish(base_cmd)
			tra_info = str(base_cmd.linear.x) + ' ' + str(base_cmd.angular.z) + ' ' + str(dt) + '\n'
			print ('saving '+ tra_info)		
			self.saver_file.write(tra_info)


	###   For bench testing with dataset images ###############################
	def showFrame(self, frame, name):
		cv2.imshow(name, frame)
		cv2.waitKey(20)
