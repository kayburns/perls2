from demo_control_env import DemoControlEnv
import numpy as np
import time 
import logging
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

from perls2.controllers.utils.control_utils import *
import perls2.controllers.utils.transform_utils as T

import datetime

class Demo():
    """Class definition for demonstration. 
        Demonstrations are a series of actions that follow a specified pattern and are 
        of appropriate dimension for the controller. 
    
        Attributes:
        env (DemoControlEnv): environment to step with action generated by demo. 
        ctrl_type (str): string identifying type of controller:
            EEPosture, EEImpedance, JointImpedance, JointTorque
        demo_type (str): type of demo to perform (zero, line, sequential, square.)

    """
    def __init__(self, ctrl_type, demo_type, use_abs, test_fn, **kwargs):
        self.env = DemoControlEnv('dev/validation/demo_control_cfg.yaml', 
            use_visualizer=True, 
            use_abs=use_abs, 
            test_fn=test_fn, 
            name='Demo Control Env', 
            )
        print("initializing")
        self.ctrl_type = ctrl_type
        self.env.robot_interface.change_controller(self.ctrl_type)
        self.demo_type = demo_type
        self.use_abs = use_abs
        self.test_fn = test_fn
        self.plot_pos = kwargs['plot_pos']
        self.plot_error = kwargs['plot_error']


        # Initialize lists for storing data.
        self.errors = []
        self.actions = []
        self.states = []
        self.world_type = self.env.config['world']['type']
        self.initial_pose = self.env.robot_interface.ee_pose

    def get_action_list(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def save_data(self, name=None):
        if name is None:
            name = "dev/demo_control/data_npz/{}_{}_{}.npz".format(str(time.time()), self.ctrl_type, self.demo_type)

        np.savez(name, states=self.states, errors=self.errors, actions=self.actions, goals=self.goal_states, allow_pickle=True)
        data = np.load(name)

    def get_goal_state(self, delta):
        raise NotImplementedError

    def make_demo(**kwargs):
        """Factory method for making the write demo per params. 
        """
        if kwargs['ctrl_type'] in ["EEImpedance", "EEPosture"]:
            return OpSpaceDemo(**kwargs)

class OpSpaceDemo(Demo):
    """ Demos for testing operational space control. These include
    End-Effector Imedance, and End-Effector with nullspace posture control.
    """
    def __init__(self, ctrl_type, demo_type, use_abs, delta_val=0.02, num_steps=50, test_fn='set_ee_pose', **kwargs):
        """
        ctrl_type (str): string identifying type of controller:
            EEPosture, EEImpedance, JointImpedance, JointTorque
        demo_type (str): type of demo to perform (zero, line, sequential, square.)
        """
        super().__init__(ctrl_type, demo_type, use_abs, test_fn, **kwargs)
        self.delta_val = delta_val
        self.num_steps = num_steps
        self.action_list = self.get_action_list()
        self.goal_states = self.get_goal_states()
        #self.print_name = "Demo_{}_{}_{}_{}".format()

    def run(self):
        """Run the demo. Execute actions in sequence and calculate error. 
        """
        print("Running {} demo \n with control type {}.\n Test function {}".format(
            self.ctrl_type, self.demo_type, self.test_fn))

        print(self.get_state())
        for i, action in enumerate(self.action_list):
            print("Action:\t{}\n".format(action))
            self.env.step(action, time.time())
            self.actions.append(action)
            new_state = self.get_state()
            print(new_state)
            self.states.append(new_state)
            self.errors.append(self.compute_error(self.goal_states[i], new_state))
        
        if self.plot_error:
            self.plot_errors()
        if self.plot_pos:
            self.plot_positions()

    def get_action_list(self):
        """Get the set of actions based on the type of the demo, 
        """

        if self.demo_type == "Zero":
            initial_pose = self.env.robot_interface.ee_pose
            if self.use_abs:
                # Holding an absolute pose. [x y z qx qy qz]
                action_list = [np.array(initial_pose)]*self.num_steps
            else:
                # Just sending all 0s as actions (open loop).
                action_list = [np.zeros(6)]*self.num_steps
        elif self.demo_type in ["Line", "Square"]:
            initial_pose = self.env.robot_interface.ee_pose_euler

            if self.demo_type == "Square":
                self.path = Square(start_pos=initial_pose, 
                          side_num_pts=int(self.num_steps/4),
                          delta_val=self.delta_val)
            elif self.demo_type == "Line":
                self.path = Line(start_pos=initial_pose, 
                    num_pts=self.num_steps, 
                    delta_val=self.delta_val)

            if self.use_abs:
                action_list = self.path.path
            else:
                action_list = self.path.deltas
        else:
            raise ValueError("Invalid Demo type")
        return action_list

    def get_goal_states(self):
        """ Get goal states based on the demo type and action list. 

        For Zero goal states with use_abs, goal states are just
        copied initial end-effector pose.
        """ 
        if self.demo_type == "Zero":
            # Save goal states as eulers because they are easier to compute difference.
            initial_pose_euler = self.env.robot_interface.ee_pose_euler
            goal_states = [np.array(initial_pose_euler)]*self.num_steps
        elif self.demo_type in ["Line", "Square"]:
            if self.path is not None:
                goal_states = self.path.path
            else:
                raise ValueError("Get actions list before goal states")
        else:
            raise ValueError("Invalid demo type.")
        return goal_states 

    def get_state(self):
        """ Proprio state for robot we care about for this demo: ee_pose. 
        Used to compute error. 
        """
        return self.env.robot_interface.ee_pose_euler

    def compute_error(self, goal_state, new_state):
        """ Compute the error between current state and goal state.
        For OpSpace Demos, this is position and orientation error.
        """
        goal_pos = goal_state[:3]
        new_pos = new_state[:3]
        # Check or convert to correct orientation 
        goal_ori = T.convert_euler_quat_2mat(goal_state[3:])
        new_ori = T.convert_euler_quat_2mat(new_state[3:])

        pos_error = np.subtract(goal_pos, new_pos)
        ori_error = orientation_error(goal_ori, new_ori)

        return np.hstack((pos_error, ori_error))

    def plot_positions(self):
        """ Plot 3 plots showing xy, xz, and yz position. 
        Helps for visualizing decoupling.
        """
        goal_x = [goal[0] for goal in self.goal_states]
        goal_y = [goal[1] for goal in self.goal_states]
        goal_z = [goal[2] for goal in self.goal_states]

        state_x = [state[0] for state in self.states]
        state_y = [state[1] for state in self.states]
        state_z = [state[2] for state in self.states]

        fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1,3)

        ax_xy.plot(goal_x, goal_y, 'or')
        ax_xy.plot(state_x, state_y, '*b')
        ax_xy.set_xlabel("x position (m)")
        ax_xy.set_ylabel("y position (m)")
        
        ax_xz.plot(goal_x, goal_z, 'or')
        ax_xz.plot(state_x, state_z, '*b')
        ax_xz.set_xlabel("x position (m)")
        ax_xz.set_ylabel("z position (m)")

        ax_yz.plot(goal_y, goal_z, 'or')
        ax_yz.plot(state_y, state_z, '*b')
        ax_yz.set_xlabel("y position (m)")
        ax_yz.set_ylabel("z position(m)")

        plt.show()

    def plot_errors(self):
        """ Plot 6 plots showing errors for each dimension. 
        x, y, z and qx, qy, qz euler angles (from orientation_error)
        """
        errors_x = [error[0] for error in self.errors]
        errors_y = [error[1] for error in self.errors]        
        errors_z = [error[2] for error in self.errors]

        errors_qx = [error[3] for error in self.errors]
        errors_qy = [error[4] for error in self.errors]
        errors_qz = [error[5] for error in self.errors]

        fig, ((e_x, e_y, e_z), (e_qx, e_qy, e_qz)) = plt.subplots(2, 3)
        
        e_x.plot(errors_x)
        e_x.set_title("X error per step.")
        e_x.set_ylabel("error (m)")
        e_x.set_xlabel("step num")

        e_y.plot(errors_y)
        e_y.set_ylabel("error (m)")
        e_y.set_xlabel("step num")
        e_y.set_title("y error per step")

        e_z.plot(errors_z)
        e_z.set_title("z error per step.")
        e_z.set_ylabel("error (m)")
        e_z.set_xlabel("step num")

        e_qx.plot(errors_qz)
        e_qx.set_title("qx error per step.")
        e_qx.set_ylabel("error (rad)")
        e_qx.set_xlabel("step num")

        e_qy.plot(errors_qy)
        e_qy.set_title("qy error per step.")
        e_qy.set_ylabel("error (rad)")
        e_qy.set_xlabel("step num")

        e_qz.plot(errors_qz)
        e_qz.set_title("qz error per step.")
        e_qz.set_ylabel("error (rad)")
        e_qz.set_xlabel("step num")        

        plt.show()
           
class Path():
    """Class definition for path definition (specific to ee trajectories)
    
    A Path is a series
 
    """

    def __init__(self, shape, num_pts): 
        self.shape = shape
        self.num_pts = num_pts
        self.path = []

    def make_path(self):
        self.path = [self.start_pos]
        for delta in self.deltas:
            self.path.append(np.add(self.path[-1], delta))
class Line(Path):
    """Class definition for straight line in given direction. 
    """
    def __init__(self, start_pos, num_pts, delta_val, dim=0):
        self.start_pos = start_pos
        self.num_pts = num_pts
        self.delta_val = delta_val
        self.dim = dim
        self.deltas = []
        self.get_deltas()
        self.path = []
        self.make_path()

    def get_deltas(self):
        delta = np.zeros(6)
        delta[self.dim] = self.delta_val
        self.deltas = [delta]*self.num_pts
        self.deltas[0] = np.zeros(6)

class Square(Path):
    """Class def for square path. 

    Square path defined by side length and start point. 
    At step 4 * sidelength -1, ee is not at initial point. 
    Last step returns to initial point. 

    Square path is ordered in clockwise from origin (Bottom, Left, Top, Right)

    Attributes: 
        start_pos (3f): xyz start position to begin square from. 
        side_num_pts (int): number of steps to take on each side. (not counting start.)
        delta_val (float): step size in m to take for each step. 
        deltas (list): list of delta xyz from a position to reach next position on path.
        path (list): list of actions to take to perform square path. Actions are either delta
            xyz from current position (if use_abs is False) or they are absolute positions
            taken by adding the deltas to the start pos.

    """ 
    def __init__(self, start_pos, side_num_pts, delta_val):

        self.start_pos = start_pos
        self.side_num_pts = side_num_pts
        self.delta_val = delta_val
        self.deltas = []
        self.get_deltas()
        self.path = []
        self.make_path()

    def get_deltas(self):
        """ Get a series of steps from current position that produce 
        a square shape. Travel starts with bottom side in positive direction, 
        then proceeds counter-clockwise (left, top, right.)

        """
        self.deltas = [[0, 0, 0, 0, 0, 0]]
        # Bottom side.
        for pt in range(self.side_num_pts):
           self.deltas.append([self.delta_val, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Left Side
        for pt in range(self.side_num_pts):
             self.deltas.append([0.0, self.delta_val, 0.0, 0.0, 0.0, 0.0])
        # Top side
        for pt in range(self.side_num_pts):
             self.deltas.append([-self.delta_val, 0, 0.0, 0.0, 0.0, 0.0])
        # Right side
        for pt in range(self.side_num_pts):
            self.deltas.append([0.0, -self.delta_val, 0.0, 0.0, 0.0, 0.0])



