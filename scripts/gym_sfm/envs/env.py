import gym
from gym import error, spaces, utils
from gym.utils import seeding
import Box2D

import numpy as np
import time
import math
import yaml
import sys, os
import random

import pyximport;pyximport.install(setup_args={"include_dirs": np.get_include()})
PARDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PARDIR)

from gym_sfm.envs.node import Node, make_node_net
from gym_sfm.envs.world import World, Collision
from gym_sfm.envs.cython.wall import Wall, make_wall
from gym_sfm.envs.cython.actor import Actor, make_actor_random
from gym_sfm.envs.cython.agent import Agent, make_agent_random
from gym_sfm.envs.cython.zone import Zone, make_zone, select_generate_ac_zone, specified_range_generate_ac_zone, select_generate_ag_zone, the_generate_ag_zone, check_zone_target_existence

def check_name_duplicate(obj):
    name_list = [ o['name'] for o in obj ]
    duplicate = [ name for name in set(name_list) if name_list.count(name) > 1 ]
    if duplicate : raise RuntimeError('---------- The name must be unique.('+str(duplicate)+') ----------')
    else : return True

def get_config(config_file):
    config = []
    if config_file is not '' :
        with open(config_file) as f :
            config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

class GymSFM(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, md, tl, agf='env_default'):
        super(GymSFM, self).__init__()
        self.fps = 25
        self.dt = 1.0/self.fps
        self.suspend_limit = self.fps/2
        self.total_step = 0
        self.step_limit = int(tl/self.dt)

        self.map_dir = md
        self.map = ''

        self.viewer = None
        self.world = World()

        self.ag_file = agf
        self.n = []
        self.agent = None
        self.observation_space = self.reset()
        self.action_space = self.agent.action_space

        self.goal_flag = False
        self.collision_flag = False

    def _destroy(self):
        self.world._destroy()
        self.walls = []
        self.zones = []
        self.nodes = []
        self.actors = []
        self.agent = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self):
        map_file = self.select_mapfile_randam()
        actor_file = 'example/default.yml'
        agent_file = 'default.yml'
        self._destroy()
        self.total_step = 0
        self.actor_num = 0
        self.max_actor_num = 0
        self.max_agent_num = 1
        self.total_actor_num = 0
        self.total_agent_num = 0
        self.convergence_radius = 1.0
        self.actor_conf = {}
        self.agent_conf = {}
        self.map_view_conf = {}
        self.agent_view_conf = {}
        self.actor_view_conf = {}
        self.max_human_obs_range = 10.0 #random.uniform(5.5, 6.0)
        self.min_human_obs_range = 0.5  #random.uniform(0.1, 0.3)
        self.human_obs_resolution = 5

        map = get_config(map_file)
        if 'size' in map :
            self.map_scale = map['size']['scale'] if 'scale' in map['size'] else 1
            self.map_width = map['size']['width']/self.map_scale
            self.map_height = map['size']['height']/self.map_scale
            self.map_ratio = self.map_height/self.map_width
            self.screen_width = 700 if self.map_width < 50 else 1000
            self.screen_height = int(self.screen_width*self.map_ratio)
            self.viewer_scale = self.screen_width/self.map_width
        else : raise RuntimeError('---------- Size element is required ----------')
        if 'walls' in map :
            if check_name_duplicate(map['walls']) :
                for wall in map['walls']:
                    wall_list = make_wall(wall, self.map_scale)
                    self.walls.extend(wall_list)
        if 'nodes' in map :
            if check_name_duplicate(map['nodes']) :
                self.nodes = [ Node(node['name'], np.array(node['pose'], dtype=np.float64), [], self.map_scale) for node in map['nodes'] ]
        if 'zones' in map :
            if check_name_duplicate(map['zones']) :
                if len(map['zones']) < 2 :
                    raise RuntimeError('---------- There must be at least two Zone. ----------')
                self.zones = [ make_zone(zone, [self.map_width, self.map_height, self.map_scale], self.suspend_limit+random.randint(0, 200)) for zone in map['zones'] ]
                check_zone_target_existence(self.zones)
        if 'viewer' in map :
            self.map_view_conf = map['viewer']
        if 'actor' in map :
            if 'config_rel_path' in map['actor'] : actor_file = map['actor']['config_rel_path']
            if 'max_actor_num' in map['actor'] : self.max_actor_num = map['actor']['max_actor_num']
            if 'convergence_radius' in map['actor'] : self.convergence_radius = map['actor']['convergence_radius']
        if 'agent' in map :
            if self.ag_file != 'env_default' : agent_file = self.ag_file
            elif 'config_rel_path' in map['agent'] : agent_file = map['agent']['config_rel_path']
            if 'max_agent_num' in map['agent'] : self.max_agent_num = map['agent']['max_agent_num']

        self.world.make_walls(self.walls)

        # Init agent
        obs = None
        self.make_agent(agent_file)
        if self.agent is not None :
            obs, _ = self.agent.observation(self.world, [self.agent.dis, self.agent.angle_to_goal])

        # Init actor
        self.actor_conf = get_config(PARDIR+'/config/actor/'+actor_file)
        if 'actor' in self.actor_conf :
            stop_generate_actor = False
            while self.actor_num < self.max_actor_num and not stop_generate_actor :
                can_generate_actor = specified_range_generate_ac_zone(self.zones, self.total_step, self.agent.pose) # [start, target, target_zone]
                if can_generate_actor :
                    actor = make_actor_random(
                        self.actor_conf['actor'], can_generate_actor,
                        [self.dt, self.map_scale, 'actor'+str(self.total_actor_num), self.total_step],
                        self.convergence_radius, self.nodes, self.walls, self.world)
                    self.actors.append(actor)
                    self.actor_num += 1
                    self.total_actor_num += 1
                else : stop_generate_actor = True
        else : raise RuntimeError('---------- Actor element is required ----------')
        if 'viewer' in self.actor_conf : self.actor_view_conf = self.actor_conf['viewer']

        can_people_name, can_people_pose, can_people_yaw = self.calc_can_obs_human(self.agent, self.actors, obs)

        self.agent_base_pose = self.agent.pose #if agent is single.

        return obs, can_people_name, can_people_pose
    
    def calc_can_obs_human(self, agent, actors, obs):
        obs_ex = obs * agent.lidar_linear_range
        human_obs_ranges = []
        human_obs_range = np.zeros(0)
        for idx, v in enumerate(obs_ex): # vは今回入れるobs_exの要素
            if idx % self.human_obs_resolution == 0:
                human_obs_range = np.append(human_obs_range, v)
        human_obs_ranges.append(human_obs_range)

        # each_ag_obs_people = []
        for human_obs_range in human_obs_ranges:
            one_detect_people_name = [] * self.max_actor_num
            one_detect_people_pose = [] * self.max_actor_num * 2
            one_detect_people_yaw = [] * self.max_actor_num
            # print(len(one_detect_people))
            # one_ag_obs_people = np.zeros(self.max_actor_num*5)
            # i = 0
            for actor in actors:
                ax = actor.pose[0] - agent.pose[0]
                ay = actor.pose[1] - agent.pose[1]
                ayaw = agent.yaw
                relative_yaw = actor.yaw - agent.yaw
                trans_pose = np.array([
                    ax*np.cos(ayaw) + ay*np.sin(ayaw),
                    -ax*np.sin(ayaw) + ay*np.cos(ayaw),
                    np.arctan2(np.sin(relative_yaw), np.cos(relative_yaw))
                ])
                relative_yaw = math.degrees(relative_yaw)
                if actor.v[0] < 0.001 and actor.v[0] > -0.001:
                    actor.v[0] = 0.001
                w = np.arctan(actor.v[1]/actor.v[0])
                v = math.dist([0,0],actor.v)
                dis = math.dist(agent.pose, actor.pose)

                # if i >= self.max_actor_num*5: #5 is human states
                #     break

                search_human = True
                for idx, d in enumerate(human_obs_range):
                    yaw = idx * agent.lidar_rad_step * self.human_obs_resolution - 0.5*agent.lidar_rad_range
                    # print(yaw*180/math.pi)

                    # yaw = math.radians(idx * 8 - 90)
                    # if math.fabs(d - dis) < 5.0 and math.fabs(relative_yaw - ((idx*4 + 45.0)*math.pi/180)) < (15*math.pi/180) and dis <= d:
                    reso = 0.1
                    if d <= agent.lidar_linear_range: 
                    # if d <= 8.0: 
                        while search_human:
                            xy = np.array([d * np.cos(yaw), d * np.sin(yaw)])

                            # if math.fabs(x - ax) < 0.1 and math.fabs(y - ay) < 0.1:
                            # if np.linalg.norm(trans_pose[:2] - xy) < agent.lidar_linear_range/200:
                            if np.linalg.norm(trans_pose[:2] - xy) < 0.5:
                                # save_pose_vw = np.array([
                                #     trans_pose[0],
                                #     trans_pose[1],
                                #     trans_pose[2],
                                #     v,
                                #     w,
                                # ])

                                one_detect_people_name.append(actor.name)

                                # one_detect_people_pose.append(actor.pose)
                                one_detect_people_pose.extend(actor.pose)

                                one_detect_people_yaw.append(actor.yaw)

                                # one_ag_obs_people[i:i+5] = save_pose_vw
                                                                
                                # if actor.name not in self.n :
                                #     print(actor.name)
                                #     self.n.extend(actor.name)
                                # i +=5
                                search_human = False
                                break

                            d -= reso
                            if d < 0.0 or search_human is not True:
                                break
                        if search_human is not True:
                            break
                    
        return one_detect_people_name, one_detect_people_pose, one_detect_people_yaw 
            # each_ag_obs_people.append(one_detect_people)
        # return each_ag_obs_people

    def resize_observation(self, obs):
        obs_ex = obs * self.agent.lidar_linear_range

        return obs_ex
    
    def step(self, action):
        for a in self.actors:
            # F_target = np.zeros(2)
            F_walls = np.zeros(2)
            F_actors = np.zeros(2)
            a.reset_Force()

            # F_target = a.calc_F_target()
            for w in self.walls:
                consider = w.calc_to_wall_vec(a.pose, self.walls, a.consider_wall_radius) # [w_n, w_dis] or False
                if consider :
                    a.affected_walls.extend(consider[2])
                    F_walls += a.calc_F_wall(consider[0], consider[1])
            for aa in self.actors:
                if aa.name is a.name : continue
                consider = a.can_consider_actor(aa.pose, self.walls) # [aa_n, aa_dis] or False
                if consider :
                    f_a = a.calc_F_avoid_actor(consider[0], aa.v, aa.yaw, aa.to_goal_vec)
                    if consider[1] < a.must_avoid_radius :
                        f_a += a.calc_F_actor(consider[0], consider[1], aa.radius, aa.v)
                    F_actors += f_a
            is_goal = a.update(F_walls, F_actors, self.total_step)

            if is_goal :
                self.world.actors.remove(a.box2d_obj)
                self.world.DestroyBody(a.box2d_obj)
                self.actors.remove(a)
                self.actor_num -= 1
            if self.actor_num < self.max_actor_num :
                can_generate_actor = specified_range_generate_ac_zone(self.zones, self.total_step, self.agent.pose) # [start, target, target_zone]
                if can_generate_actor :
                    actor = make_actor_random(
                        self.actor_conf['actor'], can_generate_actor,
                        [self.dt, self.map_scale, 'actor'+str(self.total_actor_num), self.total_step],
                        self.convergence_radius, self.nodes, self.walls, self.world)
                    self.actors.append(actor)
                    self.actor_num += 1
                    self.total_actor_num += 1
            # print(self.actor_num)
            # print(self.total_actor_num)

        obs = None
        state = 0
        reward = 0
        if self.agent is not None :
            update_result = self.agent.update(action, self.total_step)
            out_of_map = self.check_in_map(self.agent.pose)
            # print(self.agent.pose)
            obs, is_collision = self.agent.observation(self.world, update_result[1:3])
            if update_result[0] : state = 1
            elif out_of_map : state = 2
            elif self.total_step > self.step_limit : state = 3
            elif is_collision : state = 4
            reward = self.get_reward(state, *update_result[1:], *action)
        self.world.Step(1.0/self.fps, 0, 0)
        self.total_step += 1
        # print(self.total_step)
        
        can_people_name, can_people_pose, can_people_yaw = self.calc_can_obs_human(self.agent, self.actors, obs)

        all_actors_pose = [] * self.max_actor_num * 2
        for actor in self.actors:
            all_actors_pose.extend(actor.pose)

        return obs, can_people_name, can_people_pose, all_actors_pose, self.agent, self.goal_flag, self.collision_flag, reward, state, {'total_step':self.total_step}

    def get_reward(self, state, dis, angle, ddis, v, omega):
        reward = 0
        if state == 0 :
            reward = -0.01 if v < 1e-3 else ddis
        elif state == 1 :
            print('--- Goal. ---')
            self.goal_flag = True
            reward += 5
        elif state == 2 :
            print('--- Out of map. ---')
            reward += ddis
        elif state == 3 :
            print('--- Time out. ---')
            reward += ddis
        elif state == 4 :
            print('--- Collition. ---')
            self.collision_flag = True
            reward += -5
        return reward

    def render(self, mode='human', close=False):
        if self.viewer is None:
            from gym_sfm.envs.viewer import Viewer
            screen = [ self.screen_width, self.screen_height, self.viewer_scale ]
            self.viewer = Viewer(screen, self.map_view_conf, self.actor_view_conf, self.agent_view_conf)
        self.viewer.make_map(self.walls, self.zones, self.nodes)
        self.viewer.make_actor(self.actors)
        self.viewer.make_agent(self.agent)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def make_agent(self, agent_file):
        self.agent_conf = get_config(PARDIR+'/config/agent/'+agent_file)
        can_generate_agent = the_generate_ag_zone(self.zones, self.total_step) # [start, target]
        if can_generate_agent and self.total_agent_num < self.max_agent_num :
            self.agent = make_agent_random(self.agent_conf['agent'], can_generate_agent, [self.dt, self.map_scale, 'agent'+str(self.total_agent_num)], self.world)
            self.total_agent_num += 1

    def check_collision(self, agent_pose, agent_radius):
        if sum(self.world.contactListener.position) > 0 :
            collision_position = np.array(self.world.contactListener.position)
            if np.linalg.norm(agent_pose - collision_position) < 2*agent_radius : return True
        return False

    def check_in_map(self, agent_pose):
        x, y = agent_pose
        if 0 < x and x < self.map_width and 0 < y and y < self.map_height :
            return False
        return True

    def select_mapfile_randam(self):
        map_dir = PARDIR+'/config/map/'+self.map_dir+'/'
        map_file = random.choice(os.listdir(map_dir))
        self.map = self.map_dir + '/' + map_file
        print(self.map)
        return map_dir + map_file

    def close(self):
        self._destroy()
