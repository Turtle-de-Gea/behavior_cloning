#! /usr/bin/env python
import rospy 
import sys
import argparse
from turtle_follower import TrajectoryFollower


if __name__ == '__main__':
    go = TrajectoryFollower()
