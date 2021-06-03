"""Project template example running the environment.
"""
from __future__ import division
import time
from rl_policy_env import RLPolicyEnv
import logging
import pybullet as p
from PIL import Image
import os
import numpy as np
from util import DatasetUtils
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)


GIF_SAVE_LOCATION = "/home/mason/before_output.gif"
FOLDER_NAME = "example_"
DATASET_LOC = "/home/mason/peg_insertation_dataset/heuristic_data_2/"
EPISODE_COUNT = 100

observations_arr = []
example_count = 0

def get_action(observation):
    """Run your policy to produce an action.
    """

    action = [0, 0, 0]
    return action

def record_obs(observation):
    observations_arr.append(observation)

env = RLPolicyEnv('projects/rl_policy_env/rl_policy.yaml', use_visualizer=False, name="TemplateEnv")
dt_util = DatasetUtils(dataset_loc=DATASET_LOC, folder_name=FOLDER_NAME)

for ep_num in tqdm(range(EPISODE_COUNT)):
    #logging.debug('episode ' + str(ep_num - 1) + ' complete...pausing...')
    step = 0
    observation = env.reset()
    done = False

    observation = reward = termination = None

    while not done:
        start = time.time()
        if env.should_record():
            action = get_action(observation)
            observation, reward, termination, info = env.step(action, start=start)
            example_count += 1
        else:
            #Do Nothing While environment is being setup
            observation, reward, termination, info = env.step([0.0, 0.0, 0.0], start=start)
        
        #color, depth = env.digits.render()
        #env.digits.updateGUI(color, depth)
        
        #print(f"Color_shape: {color[1].shape} Depth_shape: {depth[1].shape}")

        done = termination


#color_imgs[0].save(GIF_SAVE_LOCATION, save_all=True, append_images=color_imgs[1:], optimize=False, duration=40, loop=0)


# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
