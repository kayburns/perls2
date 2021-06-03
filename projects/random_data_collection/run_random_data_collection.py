"""Project template example running the environment.
"""
from __future__ import division
import time
from random_data_collection_env import RandomDataCollectionEnv
import logging
import pybullet as p
from PIL import Image
from tqdm import tqdm
from util import DatasetUtils
logging.basicConfig(level=logging.DEBUG)


GIF_SAVE_LOCATION = "/home/mason/before_output.gif"
FOLDER_NAME = "example_"
DATASET_LOC = "/home/mason/peg_insertation_dataset/random_data_contact_1/"
EPISODE_COUNT = 1000

observations_arr = []
example_count = 0


def get_action(observation):
    """Run your policy to produce an action.
    """

    action = [0, 0, 0]
    return action

def record_obs(observation):
    observations_arr.append(observation)

env = RandomDataCollectionEnv('projects/random_data_collection/random_data_collection.yaml', True, "TemplateEnv")
dt_util = DatasetUtils(dataset_loc=DATASET_LOC, folder_name=FOLDER_NAME)

for ep_num in tqdm(range(EPISODE_COUNT)):
    #logging.debug('episode ' + str(ep_num - 1) + ' complete...pausing...')
    step = 0
    observation = env.reset()
    done = False

    while not done:
        start = time.time()
        action = get_action(observation)

        
        # Pass the start time to enforce policy frequency.
        observation, reward, termination, info = env.step(action, start=start)

        color, depth = env.digits.render()
        env.digits.updateGUI(color, depth)
        
        if env.should_record():
            record_obs(observation)
        
        if ep_num % 100 == 0:
            dt_util.save_obs(observations_arr)
            observations_arr = []
        
        
        
        #print(f"Color_shape: {color[1].shape} Depth_shape: {depth[1].shape}")

        done = termination
        #time.sleep(0.5)


#color_imgs[0].save(GIF_SAVE_LOCATION, save_all=True, append_images=color_imgs[1:], optimize=False, duration=40, loop=0)


# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
