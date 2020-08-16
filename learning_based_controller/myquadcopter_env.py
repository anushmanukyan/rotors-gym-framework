import rospy
from std_srvs.srv import Empty
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry

from sensor_msgs.msg import Imu

import time

from enum import Enum
import numpy as np
import pandas as pd
import csv

from tf.transformations import euler_from_quaternion
import tf

import gym
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register
from gym_gazebo.envs import gazebo_env

#register the training environment in the gym as an available one
reg = register(
    id='Quadcopter-v0',
    entry_point='myquadcopter_env:QuadCopterEnv',
    #timestep_limit=100000,
    )

class QuadCopterEnv(gazebo_env.GazeboEnv):

    desired_x = 0
    desired_y = 0
    desired_z = 1.08 


    def __init__(self):

        self.motor_pub = rospy.Publisher('/firefly/command/motor_speed', Actuators, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        rospy.Subscriber('/firefly/ground_truth/pose', Pose, self.callback_pose)
        rospy.Subscriber('/firefly/ground_truth/odometry', Odometry, self.get_rotation)
        rospy.Subscriber('/firefly/imu', Imu, self.get_imu)

        high = np.array([np.inf]*12)
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1, -1, -1]), np.array([+1, +1, +1, +1, +1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high)
        self.reward_range = (-np.inf, np.inf)
        self.vel = Actuators()
        self.vel.header.frame_id = 'firefly/base_link'
        self.reset_sensor_data()
        self._seed()
        self.saved_data_path = 'saved_models/Quadcopter-v0/3006/all_data/'
        self.setup_csv()

        print('[DroneControler] Initialized.')
        
    def reset_sensor_data(self):
        self.x = 0
        self.y = 0
        self.z = 0

        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        # linear acceleration for hovering
        self.linear_x = 0
        self.linear_y = 0
        self.linear_z = 0

        self.rotor_vel_0 = 0
        self.rotor_vel_1 = 0
        self.rotor_vel_2 = 0
        self.rotor_vel_3 = 0
        self.rotor_vel_4 = 0
        self.rotor_vel_5 = 0

        rospy.loginfo('Reset data is called')

    def callback_pose(self, data):
        pose = data.position
        self.x = round(pose.x, 4)
        self.y = round(pose.y, 4)
        self.z = round(pose.z, 4)

    def get_rotation(self, data):
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        self.roll = round(roll, 4)
        self.pitch = round(pitch, 4)
        self.yaw = round(yaw, 4)

    def get_imu(self, msg):
        self.linear_x = msg.linear_acceleration.x
        self.linear_y = msg.linear_acceleration.y
        self.linear_z = msg.linear_acceleration.z

    def stop_uav(self):
        self.vel.angular_velocities = [0, 0, 0, 0, 0, 0]
        
        return(self.motor_pub.publish(self.vel))

    def publish_rotor_speed(self, rotor_speed):
        self.vel.angular_velocities = rotor_speed

        self.motor_pub.publish(self.vel)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        
        rospy.loginfo('Starting reset()...')
        rospy.wait_for_service('/gazebo/reset_world')
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')

        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:    
            print("/gazebo/reset_world service call failed")
        
        self.stop_uav()
        time.sleep(0.2)
        
        self.reset_sensor_data()
        self.good_step_counter = 0
        self.prev_shaping = None

        try:
            #rospy.loginfo('Unpause the simulation from reset()')
            self.unpause()
        except (rospy.ServiceException) as e:    
            print("/gazebo/unpause service call failed")

        state = [self.x, self.y, self.z, self.pitch, self.yaw, self.roll, self.rotor_vel_0, self.rotor_vel_1, self.rotor_vel_2, self.rotor_vel_3, self.rotor_vel_4, self.rotor_vel_5]
            
        try:
            self.pause()
        except (rospy.ServiceException) as e:    
            print("/gazebo/pause service call failed")


        return self.step(np.array([0,0,0,0,0,0]))[0]

    def scale_action(self, action, min_vel, max_vel):
        return np.interp(action, [-1, 1], [min_vel, max_vel])

    def _step(self, action):

        rospy.loginfo('Starting step()...')
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')
        
        self.unpause()
        
        #continuous action space
        min_vel = 545 - 15
        max_vel = 545 + 15
        self.rotor_vel_0 = self.scale_action(action[0], min_vel, max_vel)
        self.rotor_vel_1 = self.scale_action(action[1], min_vel, max_vel)
        self.rotor_vel_2 = self.scale_action(action[2], min_vel, max_vel)
        self.rotor_vel_3 = self.scale_action(action[3], min_vel, max_vel)
        self.rotor_vel_4 = self.scale_action(action[4], min_vel, max_vel)
        self.rotor_vel_5 = self.scale_action(action[5], min_vel, max_vel)

        new_velocities = [
            int(self.rotor_vel_0),
            int(self.rotor_vel_1),
            int(self.rotor_vel_2),
            int(self.rotor_vel_3),
            int(self.rotor_vel_4),
            int(self.rotor_vel_5)
        ]

        self.publish_rotor_speed(new_velocities)
        print('Rotors New Speed: {}'.format(new_velocities))
        
        time.sleep(0.2)
        self.pause()

        state = [self.x, self.y, self.z, self.pitch, self.yaw, self.roll, self.rotor_vel_0, self.rotor_vel_1, self.rotor_vel_2, self.rotor_vel_3, self.rotor_vel_4, self.rotor_vel_5]
        reward, done = self.reward()

        self.save_data( self.rotor_vel_0,
                        self.rotor_vel_1,
                        self.rotor_vel_2,
                        self.rotor_vel_3,
                        self.rotor_vel_4,
                        self.rotor_vel_5,
                        self.pitch,
                        self.roll,
                        self.yaw,
                        self.x,
                        self.y,
                        self.z )

        return np.array(state), reward, done, {}

    def reward(self):
        reward = 0
        done = False

        MAX_DISTANCE = 0.17

        distance_discounted = 0
        pitch_discounted = 0

        current_distance = self.get_distance()
        
        distance_discounted = 1-((current_distance/MAX_DISTANCE)**0.4)
        pitch_discounted = (1 - max(abs(self.pitch),0.0001))**(1-max(current_distance,0.1))
        roll_discounted = (1 - max(abs(self.roll),0.0001))**(1-max(current_distance,0.1))
        #yaw_discounted
        
        if current_distance < 0.03:
            self.good_step_counter += 1
            if abs(self.linear_z) < 9.9 and abs(self.linear_z) > 9.75:
                reward += (self.good_step_counter**2)*4
                print(' ')
                print('[ZONE] HOVERING')
                print(' ')
            else:
                reward = self.good_step_counter**2.5
                print(' ')
                print('[ZONE] GOOD ZONE')
                print(' ')

            print('Good_step_counter: {}'.format(self.good_step_counter))
        elif abs(self.x) >= 0.2 or abs(self.y) >= 0.2 or current_distance >= MAX_DISTANCE:
            reward = -20
            done = True
        else: 
            self.good_step_counter = 0
            print(' ')
            print('[ZONE] -------------------->!!! ON THE WAY !!!')
            print(' ')
            reward += distance_discounted * pitch_discounted * roll_discounted * 0.000001
        
        print('Linear Z: {}'.format(self.linear_z))
        print('Current distance: {} | Reward: {} | Done: {}'.format(current_distance, reward, done))
        print(' ')

        return reward, done

    def calculate_dist_between_two_Points(self, desired_x, desired_y, desired_z, current_x, current_y, current_z):
        a = np.array((current_x, current_y, current_z))
        b = np.array((desired_x, desired_y, desired_z))
        dist = np.linalg.norm(a-b)
        return dist

    def get_distance(self):
        distance = self.calculate_dist_between_two_Points(self.desired_x, self.desired_y, self.desired_z, self.x, self.y, self.z)
        return distance

    ################ Save Data #####################

    def get_csv_filename(self):
        return self.saved_data_path + 'all_values.csv'
    

    def setup_csv(self):
        fields = [
            'rotor_0',
            'rotor_1',
            'rotor_2',
            'rotor_3',
            'rotor_4',
            'rotor_5',
            'pitch',
            'roll',
            'yaw',
            'x',
            'y',
            'z',
            ]
        with open(self.get_csv_filename(), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            
            
    def save_data(self, r0, r1, r2, r3, r4, r5, pitch, roll, yaw, x, y, z):
        with open(self.get_csv_filename(), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([r0, r1, r2, r3, r4, r5, pitch, roll, yaw, x, y, z])

