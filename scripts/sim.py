#!/usr/bin/python3

import sys, os
import time
import math
import numpy as np
import random
import traceback
import argparse
import re
import copy
import gym
import gym_sfm.envs.env
import rospy
import tf_conversions
import tf2_ros
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, TransformStamped, Point, Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from ros_gym_sfm.msg import Actor

class RosGymSfm:
    def __init__(self):
        # parameter
        self.HZ = rospy.get_param("/HZ")
        self.MAP = rospy.get_param("/MAP")
        self.TIME_LIMIT = rospy.get_param("/TIME_LIMIT")
        self.radius = rospy.get_param("/actor/radius")

        # make environment
        self.env = gym.make('gym_sfm-v0', md = self.MAP, tl = self.TIME_LIMIT)

        # create instance
        self.agent_cmd_vel = Odometry()  # agent command velocity

        # publisher
        self.laser_pub = rospy.Publisher("ros_gym_sfm/scan", LaserScan, queue_size=1)  # used for debug
        self.agent_pose_pub = rospy.Publisher("ros_gym_sfm/agent_pose", PoseStamped, queue_size=1)
        self.agent_goal_pub = rospy.Publisher("ros_gym_sfm/agent_goal", PoseStamped, queue_size=1)
        self.actor_pub = rospy.Publisher("ros_gym_sfm/actor_info", Actor, queue_size=1)
        self.all_actor_pub = rospy.Publisher("ros_gym_sfm/all_actor", Marker, queue_size=1)  # used for debug
        self.agent_odom_pub = rospy.Publisher("ros_gym_sfm/odom", Odometry, queue_size=1)

        # subscriber
        # rospy.Subscriber("ros_gym_sfm/scan", LaserScan, self.callback_debug)
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
        goal_pub_flag = False

        while not rospy.is_shutdown():
            action = np.array([self.agent_cmd_vel.twist.twist.linear.x, self.agent_cmd_vel.twist.twist.angular.z], dtype=np.float64)
            
            observation, people_name, people_pose, all_people_pose, agent, reward, done, _ = self.env.step(action)

            # make agent tf
            try:
                self.broadcast_tf(agent)
            except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.rate.sleep()
                continue

            # publish scan data 
            laser = LaserScan()     
            laser.header.frame_id = "laser"
            laser.header.stamp = rospy.Time.now()
            original_obs = self.env.resize_observation(observation)
            laser.ranges = original_obs
                # delete goal info
            laser_list = list(laser.ranges)   
            del laser_list[-2:]                   
            laser.ranges = tuple(laser_list)  
                # determine laser info
            laser.angle_min = - agent.lidar_rad_range/2.0     # lower limit angle [rad]
            laser.angle_max = agent.lidar_rad_range/2.0       # upper limit angle [rad]
            laser.angle_increment = agent.lidar_rad_step      # tuning angle [rad]
            laser.time_increment = agent.time_increment       # time interval of acquired points [sec]
            laser.scan_time = agent.scan_time                 # time taken to acquire all point clouds [sec]
            laser.range_max = agent.lidar_linear_range        # maximum distance [m]
            laser.range_min = agent.lidar_linear_range/100.0  # minimum distance [m]     
            self.laser_pub.publish(laser)

            # publish agent pose 
            agent_pose = PoseStamped()
            agent_pose.header.frame_id = "map"
            agent_pose.header.stamp = rospy.Time.now()
            agent_pose.pose.position.x = agent.pose[0]
            agent_pose.pose.position.y = agent.pose[1]
            agent_pose_q = tf_conversions.transformations.quaternion_from_euler(0, 0, agent.yaw)
            agent_pose.pose.orientation.x = agent_pose_q[0]
            agent_pose.pose.orientation.y = agent_pose_q[1]
            agent_pose.pose.orientation.z = agent_pose_q[2]
            agent_pose.pose.orientation.w = agent_pose_q[3]
            self.agent_pose_pub.publish(agent_pose)
        
            # publish agent goal 
            if goal_pub_flag == False:
                agent_goal = PoseStamped() 
                agent_goal.pose.position.x = agent.target[0]
                agent_goal.pose.position.y = agent.target[1]
                agent_goal_q = tf_conversions.transformations.quaternion_from_euler(0, 0, math.pi/2)
                agent_goal.pose.orientation.x = agent_goal_q[0]
                agent_goal.pose.orientation.y = agent_goal_q[1]
                agent_goal.pose.orientation.z = agent_goal_q[2]
                agent_goal.pose.orientation.w = agent_goal_q[3]
                goal_pub_flag = True
            agent_goal.header.frame_id = "map"
            agent_goal.header.stamp = rospy.Time.now()
            self.agent_goal_pub.publish(agent_goal)

            # publish actor pose and name 
            actor = Actor()
            actor.header.frame_id = "map"
            actor.header.stamp = rospy.Time.now()
                # actor pose
            actor_pose_i = Point()
            people_num = int(len(people_pose)/2)
            for i in range(people_num):
                actor_pose_i.x = people_pose[i*2]
                actor_pose_i.y = people_pose[i*2+1]
                actor.pose.points.append(copy.copy(actor_pose_i))
                # actor name
            actor_name_box = [] * len(people_name)
            for i in range(len(people_name)):
                actor_name_box.append(int(re.sub(r"\D", "", people_name[i])))
                if i == len(people_name)-1:
                    actor.name.data = actor_name_box         
            self.actor_pub.publish(actor)

            # publish actor pose for debug for debugging by visualization  
            all_actor = Marker()
            all_actor.header.frame_id = "map"
            all_actor.header.stamp = rospy.Time.now()
            all_actor.color.r = 0
            all_actor.color.g = 1
            all_actor.color.b = 1
            all_actor.color.a = 0.8
            all_actor.ns = "ros_gym_sfm/all_actor"
            all_actor.id = 0
            all_actor.type = Marker.SPHERE_LIST
            all_actor.action = Marker.ADD
            all_actor.lifetime = rospy.Duration()
            all_actor.scale.x = self.radius
            all_actor.scale.y = self.radius
            all_actor.scale.z = self.radius
            pose = PoseStamped()
            pose.pose.orientation.w = 1
            all_actor.pose = pose.pose
            p = Point()
            all_people_num = int(len(all_people_pose) / 2)
            for i in range(all_people_num):
                p.x = all_people_pose[i*2]
                p.y = all_people_pose[i*2+1]
                all_actor.points.append(copy.copy(p))
            if len(all_actor.points) > 0:
                self.all_actor_pub.publish(all_actor)

            # publish agent odometry 
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
