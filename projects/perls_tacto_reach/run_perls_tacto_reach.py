"""BulletSawyerExample
"""
from __future__ import division

from perls_tacto_reach_env import PerlsTactoReachEnv
import numpy as np
import gym

import perls2
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
import time


def get_action(observation):
    """Dummy policy to get action based on robot state

    Given a delta xyz position from the end effector to the goal, return a
    vector in that direction with fixed magnitude

    Args:
        observation (3f): a vector of 3 floats corresponding to
        goal_position - current_ee_position.
    Returns:
        action (3f): vector of 3 floats corresponding to a unit vector direction

    """
    # Get components from observations
    delta = observation
    action = delta / np.linalg.norm(delta)
    return action


env = PerlsTactoReachEnv('projects/perls_tacto_reach/perls_tacto_reach.yaml', True, None)

# Lists for saving demonstrations
training_list = []
image_list = []
pose_list = []
action_list = []

#env.world.add_object('data/scene_objects/beveled_bin/', name="beveled_bin", pose=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

for ep_num in range(5):
    if (ep_num > 0):
        print("Episode {} complete..pausing".format(ep_num - 1))
        time.sleep(3)
    step = 0
    observation = env.reset()
    done = False

    
    while not done:
        action = get_action(observation[0])
        observation, reward, termination, info = env.step(action)
        step_record = (action, observation, reward, termination)

        # Add observations-actions to demonstration lists.
        pose_list.append(observation[1])
        image_list.append(observation[2])
        action_list.append(action)

        color, depth = env.digits.render()
        env.digits.updateGUI(color, depth)

        step += 1
        done = termination

env.robot_interface.reset()
