"""Project template example running the environment.
"""
from __future__ import division
import time
from data_collection_env import DataCollectionEnv
import logging
import pybullet as p
from PIL import Image
logging.basicConfig(level=logging.DEBUG)


GIF_SAVE_LOCATION = ""
color_imgs = []


def get_action(observation):
    """Run your policy to produce an action.
    """
    color_imgs.append(Image.fromarray(observation["cam_color"]))

    action = [0, 0, 0]
    return action


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

        
        color, depth = env.digits.render()
        env.digits.updateGUI(color, depth)
        
        print(f"Color_shape: {color[1].shape} Depth_shape: {depth[1].shape}")

        done = termination
        time.sleep(0.5)

print (f"num_imgs {len(color_imgs)}")
print (color_imgs[0])
color_imgs[0].save(GIF_SAVE_LOCATION, save_all=True, append_images=color_imgs[1:], optimize=False, duration=40, loop=0)
print ("GIF saved")

# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
