import numpy as np
import torch
import gym
from arguments import get_args
from models import Network

import myquadcopter_env
import rospy

# denormalize
def denormalize(x, mean, std, clip=10):
    x = x - mean
    x = x / (std + 1e-8)
    x = np.clip(x, -clip, clip)
    return x

def get_tensors(x):
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

if __name__ == '__main__':
    rospy.init_node('drone_controller', anonymous=True)

    args = get_args()
    
    # create the environment
    env = gym.make('Quadcopter-v0')
    rospy.loginfo( "[Gym Environment: DONE]")
    
    # build up the network
    net = Network(env.observation_space.shape[0], env.action_space.shape[0])
    
    # load the saved model
    # change the path to the model you want to test
    model_path = args.save_dir + args.env_name + '/3006/demo.pt'
    network_model, filters = torch.load(model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(network_model)
    net.eval()
    for _ in range(50):
        obs = denormalize(env.reset(), filters.rs.mean, filters.rs.std)
        reward_total = 0
        for _ in range(10000):
            #env.render()
            obs_tensor = get_tensors(obs)
            with torch.no_grad():
                _, (mean, _) = net(obs_tensor)
                action = mean.numpy().squeeze()
            obs, reward, done, _ = env.step(action)
            reward_total += reward
            obs = denormalize(obs, filters.rs.mean, filters.rs.std)
            if done:
                break
        print('the reward of this episode is: {}'.format(reward_total))
    #env.close()
