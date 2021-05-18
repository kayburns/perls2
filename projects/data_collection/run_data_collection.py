"""Project template example running the environment.
"""
from __future__ import division
import time
from data_collection_env import DataCollectionEnv
import logging
import pybullet as p
from PIL import Image
import os
import numpy as np
logging.basicConfig(level=logging.DEBUG)


GIF_SAVE_LOCATION = "/home/mason/before_output.gif"
FOLDER_NAME = "example_"
DATASET_LOC = "/home/mason/peg_insertation_dataset/heuristic_data/"

observations_arr = []
obs_count = 0

def get_action(observation):
    """Run your policy to produce an action.
    """

    action = [0, 0, 0]
    return action

def _save_component(obs_component, save_location, idx = 0):
    cam_clr_img = Image.fromarray(obs_component["cam_color"])
    cam_clr_img.save(os.path.join(save_location, f"cam_color_{idx}.png"))
    cam_dpth_img = Image.fromarray(obs_component["cam_depth"]).convert("L")
    cam_dpth_img.save(os.path.join(save_location, f"cam_depth_{idx}.png"))

    tacto_clr_0 = Image.fromarray(obs_component["digits_color"][0])
    tacto_clr_0.save(os.path.join(save_location, f"digits_color_0_{idx}.png"))
    tacto_clr_1 = Image.fromarray(obs_component["digits_color"][1])
    tacto_clr_1.save(os.path.join(save_location, f"digits_color_1_{idx}.png"))

    #tacto_dpth_0 = Image.fromarray(obs_component["digits_depth"][0]).convert("L")
    #tacto_dpth_0.save(os.path.join(save_location, f"digits_depth_0_{idx}.png"))
    #tacto_dpth_1 = Image.fromarray(obs_component["digits_depth"][1]).convert("L")
    #tacto_dpth_1.save(os.path.join(save_location, f"digits_depth_1_{idx}.png"))

    np.save(os.path.join(save_location, f"digits_depth_0_{idx}.npy"), obs_component["digits_depth"][0])
    np.save(os.path.join(save_location, f"digits_depth_1_{idx}.npy"), obs_component["digits_depth"][1])
    robot_proprio = obs_component["proprio"]
    np.save(os.path.join(save_location, f"proprio_{idx}.png"), robot_proprio)

def save_obs(observation_array):
    #Create Example Folder
    count = 0
    for observation in observation_array:
        save_location = os.path.join(DATASET_LOC, FOLDER_NAME + str(count)) + "/"
        if not os.path.exists(save_location):
            os.mkdir(save_location)

        first_comp = observation[0]
        sec_comp = observation[1]
        action = observation[2]

        _save_component(first_comp, save_location, idx=0)
        _save_component(sec_comp, save_location, idx=1)

        np.save(os.path.join(save_location, f"action_vec.png"), action)
        count += 1

def record_obs(observation):
    observations_arr.append(observation)

env = DataCollectionEnv('projects/data_collection/data_collection.yaml', True, "TemplateEnv")

for ep_num in range(1):
    logging.debug('episode ' + str(ep_num - 1) + ' complete...pausing...')
    step = 0
    observation = env.reset()
    done = False

    while not done:
        start = time.time()
        action = get_action(observation)

        
        # Pass the start time to enforce policy frequency.
        observation, reward, termination, info = env.step(action, start=start)

        if env.should_record():
            record_obs(observation)
            obs_count += 1
                
        color, depth = env.digits.render()
        env.digits.updateGUI(color, depth)
        
        print(f"Color_shape: {color[1].shape} Depth_shape: {depth[1].shape}")

        done = termination
        time.sleep(0.1)


#color_imgs[0].save(GIF_SAVE_LOCATION, save_all=True, append_images=color_imgs[1:], optimize=False, duration=40, loop=0)
save_obs(observations_arr)

# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
