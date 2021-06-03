import gym
import sys
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines import TRPO
from rl_policy_env import RLPolicyEnv
import torch
import numpy as np
from stable_baselines.common.callbacks import BaseCallback
#from stable_baselines.results_plotter import load_results
from stable_baselines.bench.monitor import Monitor
import os
from models.tacto_sensor_fusion import SensorFusionSelfSupervised

WEIGHTS_LOC = '/home/mason/multimodal_tacto_rep/multimodal/logging/202105312206_/models/weights_itr_48.ckpt'
NO_TACTO_WEIGHTS_LOC = '/home/mason/multimodal_tacto_rep/multimodal/no_tacto_logging/models/weights_itr_49.ckpt'

def load_model(self, model, path):
    self.logger.print("Loading model from {}...".format(path))
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)
    model.eval()


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


device = torch.device("cuda")

#env = gym.make('CartPole-v1')
log_dir = "/home/mason/perls2/projects/rl_policy_env/policy_log/"
env = RLPolicyEnv('projects/rl_policy_env/rl_policy.yaml', False, "TemplateEnv")
env = Monitor(env, log_dir)

timestep_count = 2000 * 101
#policy = FeedForwardPolicy(net_arch=[128, 128])
model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=timestep_count)
#model.save("trpo_cartpole")

#del model # remove to demonstrate saving and loading

#model = TRPO.load("trpo_cartpole")

ep_rewards = np.array(env.episode_rewards)
ep_lengths = np.array(env.episode_lengths)
ep_mean_rewards = ep_rewards / ep_lengths

EPISODE_COUNT = 20

save_loc = log_dir

np.save(os.path.join(save_loc, "mean_rewards_arr.npy"), ep_mean_rewards)
np.save(os.path.join(save_loc, "len_array.npy"), ep_lengths)
np.save(os.path.join(save_loc, "ep_total_rewards.npy"), ep_rewards)

vis_env =  RLPolicyEnv('projects/rl_policy_env/rl_policy.yaml', True, "TemplateEnv")
for i in range(EPISODE_COUNT):
    obs = vis_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = vis_env.step(action)

        color, depth = vis_env.digits.render()
        vis_env.digits.updateGUI(color, depth)
        #env.render()

        if done:
            print ("RESETTING ENVIRONMENT")
            break
        

