#!/usr/bin/python3

import rospy
import sys, os
import time
import math
import numpy as np
import random
import traceback
import argparse
import re
import gym
import gym_sfm.envs.env
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Byte
from ros_gym_sfm.msg import two_dimensional_position

class RosGymSfm:
    def __init__(self):
        self.HZ =  rospy.get_param("/HZ")
        self.MAP = rospy.get_param("/MAP")
        self.TIME_LIMIT = rospy.get_param("/TIME_LIMIT")

        self.env = gym.make('gym_sfm-v0', md = self.MAP, tl = self.TIME_LIMIT)

        self.lidar = LaserScan()                      # scan data
        self.actor_pose = Float32MultiArray()         # actor pose
        self.actor_name= Int32MultiArray()            # actor name
        self.actor_num = Byte()                       # actor total number
        self.agent_pose = two_dimensional_position()  # agent pose
        self.agent_goal = two_dimensional_position()  # agent goal

        #publisher
        self.lidar_pub = rospy.Publisher("ros_gym_sfm/scan", LaserScan, queue_size=10)                           #DWAで使用する
        self.actor_pose_pub = rospy.Publisher("ros_gym_sfm/actor_pose", Float32MultiArray, queue_size=10)
        self.actor_name_pub = rospy.Publisher("ros_gym_sfm/actor_name", Int32MultiArray, queue_size=10)
        self.actor_num_pub = rospy.Publisher("ros_gym_sfm/actor_num", Byte, queue_size=10)
        self.agent_pose_pub = rospy.Publisher("ros_gym_sfm/agent_pose", two_dimensional_position, queue_size=10)
        self.agent_goal_pub = rospy.Publisher("ros_gym_sfm/agent_goal", two_dimensional_position, queue_size=10)

        #subscriber debug
        # rospy.Subscriber("ros_gym_sfm/scan", LaserScan, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/actor_pose", Float32MultiArray, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/actor_name", Int32MultiArray, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/actor_num", Byte, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/agent_pose", two_dimensional_position, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/agent_goal", two_dimensional_position, self.callback_debug)

    def callback_debug(self, debug):
        rospy.loginfo(debug)

    def process(self):
        rate = rospy.Rate(self.HZ)
        observation = self.env.reset()
        done = False

        while not rospy.is_shutdown():
            action = np.array([0, 0], dtype=np.float64)
            
            observation, people_name, people_pose, total_actor_num, agent, reward, done, _ = self.env.step(action)

            # if len(people_name) == 0:
            #     print("no scan")
            
            #scan data publish
            original_obs = self.env.resize_observation(observation)
            self.lidar.ranges = original_obs
            self.lidar_pub.publish(self.lidar)

            # actor pose publish
            self.actor_pose.data = people_pose
            self.actor_pose_pub.publish(self.actor_pose)

            # actor name publish
            self.actor_name_box = [] * len(people_name)
            for i in range(len(people_name)):
                self.actor_name_box.append(int(re.sub(r"\D", "", people_name[i])))
                if i == len(people_name)-1:
                    self.actor_name.data = self.actor_name_box
            self.actor_name_pub.publish(self.actor_name)

            # actor total number publish
            self.actor_num.data = total_actor_num
            self.actor_num_pub.publish(self.actor_num)

            # agent pose publish
            self.agent_pose.x = agent.pose[0]
            self.agent_pose.y = agent.pose[1]
            self.agent_pose_pub.publish(self.agent_pose)
        
            # agent goal publish
            self.agent_goal.x = agent.target[0]
            self.agent_goal.y = agent.target[1]
            self.agent_goal_pub.publish(self.agent_goal)
            
            self.env.render()

            rate.sleep()

        self.env.close()

if __name__ == '__main__':
    rospy.init_node("ros_gym_sfm", anonymous=True)
    # rospy.loginfo("hello ros")   # check to see if ros is working

    ros_gym_sfm = RosGymSfm()

    try:
        ros_gym_sfm.process()

    except rospy.ROSInterruptException:
        pass
