"""Project template example running the environment.
"""
from __future__ import division
import time
from perls_tacto_env import PerlsTactoEnv
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)


def get_action(observation):
    """Run your policy to produce an action.
    """
    delta = observation
    action = delta / np.linalg.norm(delta)
    return [0.0, 0.0, 0.0]


env = PerlsTactoEnv('projects/perls_tacto_mix/perls_tacto_conf.yaml', True, "PerlsTactoEnv")

for ep_num in range(10):
    logging.debug('episode ' + str(ep_num - 1) + ' complete...pausing...')
    step = 0
    observation = env.reset()
    done = False

    while not done:
        start = time.time()
        action = get_action(observation[0])

        # Pass the start time to enforce policy frequency.
        observation, reward, termination, info = env.step(action, start=start)

        color, depth = env.digits.render()
        env.digits.updateGUI(color, depth)

        step += 1
        done = termination

# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
