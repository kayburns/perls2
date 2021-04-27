"""Project template example running the environment.
"""
from __future__ import division
import time
from test_project_env import TestProjectEnv
import logging
logging.basicConfig(level=logging.DEBUG)


def get_action(observation):
    """Run your policy to produce an action.
    """
    action = [0, 0, 0, 0, 0, 0]
    action[:3] = observation
    return action


env = TestProjectEnv('projects/test_project/test_project.yaml', True, "TemplateEnv")

for ep_num in range(10):
    if ep_num > 0:
        logging.debug('episode ' + str(ep_num - 1) + ' complete...pausing...')
        time.sleep(2)
    
    step = 0
    observation = env.reset()
    done = False

    while not done:
        start = time.time()
        action = get_action(observation)

        # Pass the start time to enforce policy frequency.
        observation, reward, termination, info = env.step(action, start=start)
        time.sleep(0.2)
        done = termination

# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
