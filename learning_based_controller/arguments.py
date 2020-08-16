import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', 
        type=float, 
        default=0.99, 
        help='the discount factor of the RL')
    parse.add_argument('--env-name', 
        type=str, 
        default='Quadcopter-v0', 
        help='the training environment')
    parse.add_argument('--seed', 
        type=int, 
        default=123, 
        help='the random seed')
    parse.add_argument('--save-dir', 
        type=str, 
        default='saved_models/', 
        help='the folder to save models')
    parse.add_argument('--total-timesteps', 
        type=int, 
        default=int(1e6), 
        help='the total frames')
    parse.add_argument('--nsteps', 
        type=int, 
        default=256, 
        help='the steps to collect samples') #1024
    parse.add_argument('--lr', 
        type=float, 
        default=3e-4)
    parse.add_argument('--batch-size', 
        type=int, 
        default=64, 
        help='the mini batch size ot update the value function')
    parse.add_argument('--vf-itrs', 
        type=int, 
        default=5, 
        help='the times to update the value network')
    parse.add_argument('--tau', 
        type=float, 
        default=0.95, 
        help='the param to calculate the gae')
    parse.add_argument('--damping', 
        type=float, 
        default=0.1, 
        help='the damping coeffificent')
    parse.add_argument('--max-kl', 
        type=float, 
        default=0.01, 
        help='the max kl divergence')
    parse.add_argument('--display-interval', 
        type=int, 
        default=20, 
        help='the interval display the training information')
    
    args = parse.parse_args()

    return args
