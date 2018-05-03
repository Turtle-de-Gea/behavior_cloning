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
This class contain functionalities for following a tag (tag 0 by default)
It records the odometry information [x, y] while following
This data is used by the robot while behavior cloning
"""
class TrajectoryFinder:
	def __init__(self):
		self.FollowTagID = 0 # which tag to follow
		self.original, self.depth = None, None
		self.bench_test, self.publish_image = True, False
		rospy.init_node('turtle_leader', anonymous=True)

		self.bridge = CvBridge()
		im_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.imageCallBack, queue_size=5)
		depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depthCallBack, queue_size=5)
		tag_pose_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.tagPoseCallback, queue_size=5)
		odom_sub = rospy.Subscriber("/odom", Odometry, self.OdometryCallback, queue_size=5)
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
			
		
	# BGR image callback function
	def imageCallBack(self, rgb_im):
		try:
			im_array = self.bridge.imgmsg_to_cv2(rgb_im, "bgr8")
		except CvBridgeError as e:
			print(e)
		if im_array is None:
			print ('frame dropped, skipping tracking')
		else:
			self.original = np.array(im_array)


	# Tag callback function
	def tagPoseCallback(self, msg):
		if self.original is not None and self.depth is not None:
			if msg.markers!=[]:
				self.tag_msg = msg.markers
				self.followTag_()
				
			if self.bench_test:	
				self.showFrame(self.original, 'input_image')
				#self.showFrame(self.depth, 'input_depth')
		if self.publish_image:
			msg_frame = CvBridge().cv2_to_imgmsg(self.original, encoding="bgr8")
			self.ProcessedRaw.publish(msg_frame)

	# See if the target tag is there, if so follow
	def followTag_(self):
		if self.tag_msg is not None:
			N_tags = len(self.tag_msg)
			tag_ids = []
			found_target_tag = False
			for i in xrange(N_tags):
				cur_tag_id = self.tag_msg[i].id
				# only follow specific tag
				if self.FollowTagID == cur_tag_id:
					self.tag_pose, self.tag_orien = self.tag_msg[0].pose.pose.position, self.tag_msg[0].pose.pose.orientation
					found_target_tag = True
				tag_ids.append(cur_tag_id)
			#print("Found tags ", tag_ids)
			if found_target_tag:
				#print ("Found target tag: at ", self.tag_pose)
				self.makemove()


	# Odometry callback function
	def OdometryCallback(self, odom_data):
		self.pos[0] = odom_data.pose.pose.position.x
		self.pos[1] = odom_data.pose.pose.position.y
		self.pos[2] = odom_data.pose.pose.position.z
		#print self.pos
		# save odometry data
		odo_info = str(self.pos[0]) + ' ' + str(self.pos[1]) +'\n'
		self.odom_file.write(odo_info)


	# Depth callback function
	def depthCallBack(self, d_im):
		try:
			d_array = self.bridge.imgmsg_to_cv2(d_im, "32FC1")
		except CvBridgeError as e:
			print(e)
		if d_array is None:
			print ('frame dropped, skipping tracking')
		else:
			self.depth = np.array(d_array)
				
	# Drive the robot to follow the tag
	def makemove(self):
		if self.tag_pose != None:
			base_cmd = Twist()
			base_cmd.linear.x = (self.tag_pose.z - 0.5)
			base_cmd.angular.z = -self.tag_pose.x*4			
			dt = (rospy.Time.now() - self.curr_time).to_sec()
			self.curr_time = rospy.Time.now()
			self.cmd_pub.publish(base_cmd)
			# save the control data for testing purposes
			tra_info = str(base_cmd.linear.x) + ' ' + str(base_cmd.angular.z) + ' ' + str(dt) + '\n'
			self.saver_file.write(tra_info)


	###   For bench testing with dataset images ###############################
	def showFrame(self, frame, name):
		cv2.imshow(name, frame)
		cv2.waitKey(20)
