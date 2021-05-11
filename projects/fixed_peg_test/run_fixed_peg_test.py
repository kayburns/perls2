"""Project template example running the environment.
"""
from __future__ import division
import time
from fixed_peg_env import FixedPegEnv
import pybullet as p
import logging
logging.basicConfig(level=logging.DEBUG)


def get_action(observation):
    """Run your policy to produce an action.
    """
    action = [0, 0, 0]
    return action


env = FixedPegEnv('projects/fixed_peg_test/fixed_peg_test.yaml', True, "FixedPegEnv")

for ep_num in range(10):
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
            
        done = termination
        time.sleep(0.5)

# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
