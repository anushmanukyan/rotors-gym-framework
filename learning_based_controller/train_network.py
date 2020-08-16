from arguments import get_args
import os
#from baselines import logger
#from baselines.common.cmd_util import make_mujoco_env
from trpo_agent import trpo_agent

import gym
import myquadcopter_env
import rospy



if __name__ == '__main__':
    rospy.init_node('drone_controller', anonymous=True)

    args = get_args()

    # make environemnts
    env = gym.make('Quadcopter-v0')
    rospy.loginfo( "[Gym Environment: DONE]")


    trpo_trainer = trpo_agent(env, args)
    trpo_trainer.learn()
