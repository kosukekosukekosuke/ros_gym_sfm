#!/usr/bin/python3

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
import rospy
import tf_conversions
import tf2_ros
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Byte
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
# from ros_gym_sfm.msg import two_dimensional_position

class RosGymSfm:
    def __init__(self):
        # parameter
        self.HZ = rospy.get_param("/HZ")
        self.MAP = rospy.get_param("/MAP")
        self.TIME_LIMIT = rospy.get_param("/TIME_LIMIT")

        # make environment
        self.env = gym.make('gym_sfm-v0', md = self.MAP, tl = self.TIME_LIMIT)

        # create instance
        self.laser = LaserScan()               # scan data
        self.actor_pose = Float32MultiArray()  # actor pose
        self.actor_name= Int32MultiArray()     # actor name
        self.actor_num = Byte()                # actor total number
        self.agent_pose = PoseStamped()        # agent pose
        self.agent_goal = PoseStamped()        # agent goal
        self.agent_cmd_vel = Odometry()        # agent command velocity

        # publisher
        self.laser_pub = rospy.Publisher("ros_gym_sfm/scan", LaserScan, queue_size=10)  #used fot DWA
        self.actor_pose_pub = rospy.Publisher("ros_gym_sfm/actor_pose", Float32MultiArray, queue_size=10)
        self.actor_name_pub = rospy.Publisher("ros_gym_sfm/actor_name", Int32MultiArray, queue_size=10)
        self.actor_num_pub = rospy.Publisher("ros_gym_sfm/actor_num", Byte, queue_size=10)
        self.agent_pose_pub = rospy.Publisher("ros_gym_sfm/agent_pose", PoseStamped, queue_size=10)
        self.agent_goal_pub = rospy.Publisher("ros_gym_sfm/agent_goal", PoseStamped, queue_size=10)
        self.agent_odom_pub = rospy.Publisher("ros_gym_sfm/odom", Odometry, queue_size=10)

        # subscriber
        # rospy.Subscriber("ros_gym_sfm/scan", LaserScan, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/actor_pose", Float32MultiArray, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/actor_name", Int32MultiArray, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/actor_num", Byte, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/agent_pose", PoseStamped, self.callback_debug)
        # rospy.Subscriber("ros_gym_sfm/agent_goal", PoseStamped, self.callback_debug)
        rospy.Subscriber("/local_path/cmd_vel", Twist, self.agent_cmd_vel_callback)

    # callback
    def agent_cmd_vel_callback(self, cmd_vel):
        self.agent_cmd_vel.twist.twist = cmd_vel

    # broadcast tf(map -> base_link)
    def broadcast_tf(self, agent):
        tf_broadcaster = tf2_ros.TransformBroadcaster()
        agent_state = TransformStamped()

        agent_state.header.stamp = rospy.Time.now()
        agent_state.header.frame_id = "map"
        agent_state.child_frame_id = "base_link"

        agent_state_q = tf_conversions.transformations.quaternion_from_euler(0, 0, agent.yaw)

        agent_state.transform.translation.x = agent.pose[0]
        agent_state.transform.translation.y = agent.pose[1]
        agent_state.transform.translation.z = 0.0
        agent_state.transform.rotation.x = agent_state_q[0]
        agent_state.transform.rotation.y = agent_state_q[1]
        agent_state.transform.rotation.z = agent_state_q[2]
        agent_state.transform.rotation.w = agent_state_q[3]

        tf_broadcaster.sendTransform(agent_state)

    def process(self):
        self.rate = rospy.Rate(self.HZ)
        observation = self.env.reset()
        done = False

        while not rospy.is_shutdown():
            action = np.array([self.agent_cmd_vel.twist.twist.linear.x, self.agent_cmd_vel.twist.twist.angular.z], dtype=np.float64)
            
            observation, people_name, people_pose, total_actor_num, agent, reward, done, _ = self.env.step(action)

            # make agent tf
            try:
                self.broadcast_tf(agent)
            except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.rate.sleep()
                continue

            # frame_id & time_stamp
            self.laser.header.frame_id = "laser"
            self.laser.header.stamp = rospy.Time.now()
            self.agent_pose.header.frame_id = "map"
            self.agent_pose.header.stamp = rospy.Time.now()
            self.agent_goal.header.frame_id = "map"
            self.agent_goal.header.stamp = rospy.Time.now()

            # scan data publish
            original_obs = self.env.resize_observation(observation)
            self.laser.ranges = original_obs
                # delete goal info
            laser_list = list(self.laser.ranges)   
            del laser_list[-2:]                   
            self.laser.ranges = tuple(laser_list)  
                # determine laser info
            self.laser.angle_min = - agent.lidar_rad_range/2.0     # lower limit angle [rad]
            self.laser.angle_max = agent.lidar_rad_range/2.0       # upper limit angle [rad]
            self.laser.angle_increment = agent.lidar_rad_step      # tuning angle [rad]
            self.laser.time_increment = agent.time_increment       # time interval of acquired points [sec]
            self.laser.scan_time = agent.scan_time                 # time taken to acquire all point clouds [sec]
            self.laser.range_max = agent.lidar_linear_range        # maximum distance [m]
            self.laser.range_min = agent.lidar_linear_range/100.0  # minimum distance [m]     
            self.laser_pub.publish(self.laser)

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
            self.agent_pose.pose.position.x = agent.pose[0]
            self.agent_pose.pose.position.y = agent.pose[1]
            agent_pose_q = tf_conversions.transformations.quaternion_from_euler(0, 0, agent.yaw)
            self.agent_pose.pose.orientation.x = agent_pose_q[0]
            self.agent_pose.pose.orientation.y = agent_pose_q[1]
            self.agent_pose.pose.orientation.z = agent_pose_q[2]
            self.agent_pose.pose.orientation.w = agent_pose_q[3]
            self.agent_pose_pub.publish(self.agent_pose)
        
            # agent goal publish
            self.agent_goal.pose.position.x = agent.target[0]
            self.agent_goal.pose.position.y = agent.target[1]
            agent_goal_q = tf_conversions.transformations.quaternion_from_euler(0, 0, math.pi/2)
            self.agent_goal.pose.orientation.x = agent_goal_q[0]
            self.agent_goal.pose.orientation.y = agent_goal_q[1]
            self.agent_goal.pose.orientation.z = agent_goal_q[2]
            self.agent_goal.pose.orientation.w = agent_goal_q[3]
            self.agent_goal_pub.publish(self.agent_goal)
            
            # agent odometry publish
            self.agent_odom_pub.publish(self.agent_cmd_vel)

            self.env.render()

            self.rate.sleep()

        self.env.close()

if __name__ == '__main__':
    rospy.init_node("ros_gym_sfm", anonymous=True)
    # rospy.loginfo("hello ros")   # check to see if ros is working

    ros_gym_sfm = RosGymSfm()

    try:
        ros_gym_sfm.process()

    except rospy.ROSInterruptException:
        pass
