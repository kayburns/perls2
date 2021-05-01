"""Template Environment for Projects.
"""
from perls2.envs.env import Env
import numpy as np
import tacto
import logging

class PerlsTactoThirdEnv(Env):
    """The class for Pybullet Sawyer Robot environments performing a reach task.
    """

    def __init__(self,
                 cfg_path='template_project.yaml',
                 use_visualizer=False,
                 name="TemplateEnv"):
        """Initialize the environment.

        Set up any variables that are necessary for the environment and your task.
        """
        super().__init__(cfg_path, use_visualizer, name)
        self.digits = tacto.Sensor(**self.config["tacto"])

        self.goal_position = self.robot_interface.ee_position
        self.object_interface = self.world.object_interfaces['013_apple']
        self.update_goal_position()

        self.robot_interface.reset()
        self.reset_position = self.robot_interface.ee_position
        self._initial_ee_orn = self.robot_interface.ee_orientation

        left_joint = self.robot_interface.get_joint_id_from_name('joint_finger_tip_left')
        right_joint = self.robot_interface.get_joint_id_from_name('joint_finger_tip_right')

        self.digits.add_camera(self.robot_interface.arm_id, [left_joint, right_joint])

        obj_path = 'data/objects/ycb/013_apple/google_16k/textured_relative.urdf' #os.path.join(self.world.arena.data_dir, 'objects/ycb/013_apple/google_16k/textured.urdf')
        obj_id = self.world.arena.object_dict['013_apple']
        self.digits.add_object(obj_path, obj_id, globalScaling=1.0)

        self.CONV_RADIUS = 0.05
        self.in_position = False
        self.grasped = False

    def update_goal_position(self):
        """Take current object position to get new goal position

            Helper function to raise the goal position a bit higher
            than the actual object.
        """
        goal_height_offset = 0.08#0.2
        object_pos = self.object_interface.position
        object_pos[2] += goal_height_offset
        self.goal_position = object_pos

    def get_observation(self):
        """Get observation of current env state

        Returns:
            observation (dict): dictionary with key values corresponding to
                observations of the environment's current state.

        """
        obs = {}
        """
        Examples:
        # Proprio:
        # Robot end-effector pose:
        obs['ee_pose'] = self.robot_interface.ee_pose

        # Robot joint positions
        obs['q'] = self.robot_interface.q

        # RGB frames from sensor:
        obs['rgb'] = self.camera_interface.frames()['rgb']

        # Depth frames from camera:
        obs['depth'] = self.camera_interface.frames()['depth']

        # Object ground truth poses (only for sim):
        obs['object_pose'] = self.world.object_interfaces['object_name'].pose

        """
        return obs

    def _get_dist_to_goal(self):

        current_ee_pos = np.asarray(self.robot_interface.ee_position)
        #weight_vec = np.array([1.2, 1.2, 0.8], dtype=np.float32)
        diff = (self.goal_position - current_ee_pos)
        abs_dist = np.linalg.norm(diff)

        return abs_dist
    
    def _exec_action(self, action):
        """Applies the given action to the environment.

        Args:
            action (list): usually a list of floats bounded by action_space.

        Examples:

            # move ee by some delta in position while maintaining orientation
            desired_ori = [0, 0, 0, 1] # save this as initial reset orientation.
            self.robot_interface.move_ee_delta(delta=action, set_ori=desired_ori)

            # Set ee_pose (absolute)
            self.robot_interface.set_ee_pose(set_pos=action[:3], set_ori=action[3:])

            # Open Gripper:
            self.robot_interface.open_gripper()

            # Close gripper:
            self.robot_interface.close_gripper()

        """

        if self.world.is_sim:
            """ Special cases for sim
            """
            pass
        else:
            """ Special cases for real world.
            """
            pass

        #Always move to goal position
        if self.in_position == False:
            self.robot_interface.set_ee_pose_position_control(self.goal_position, self._initial_ee_orn)
            if self._get_dist_to_goal() <= self.CONV_RADIUS:
                self.in_position = True
        else:
            lower_goal = self.goal_position
            lower_goal[2] -= 0.001
            #self.robot_interface.set_ee_pose_position_control(lower_goal, self._initial_ee_orn)
            self.robot_interface.set_gripper_to_value(0.8)
            self.grasped = True
        
    def reset(self):
        """Reset the environment.

        This reset function is different from the parent Env function.
        The object placement and camera intrinsics/extrinsics are
        are randomized if we are in simulation.

        Returns:
            The observation (dict):
        """
        self.episode_num += 1
        self.num_steps = 0
        self.world.reset()
        self.robot_interface.reset()
        if (self.world.is_sim):
            """
            Insert special code for resetting in simulation:
            Examples:
                Randomizing object placement
                Randomizing camera parameters.
            """
            pass
        else:
            """
            Insert code for reseting in real world.
            """
            pass

        self.robot_interface.set_gripper_to_value(0.0)
        # Step simulation until object has reached stable position.
        #self.world.wait_until_stable()

        observation = self.get_observation()

        self.in_position = False
        self.grasped = False

        ee_pose = self.robot_interface.ee_pose_euler
        #logging.debug(f"--------------------RESET ee_pose: {ee_pose}")
        return observation

    def step(self, action, start=None):
        """Take a step.

        Args:
            action: The action to take.
        Returns:
            -observation: based on user-defined functions
            -reward: from user-defined reward function
            -done: whether the task was completed or max steps reached
            -info: info about the episode including success

        Takes a step forward similar to openAI.gym's implementation.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._exec_action(action)
        self.world.step(start)
        self.num_steps = self.num_steps + 1

        termination = self._check_termination()

        # if terminated reset step count
        if termination:
            self.num_steps = 0

        reward = self.rewardFunction()

        observation = self.get_observation()

        info = self.info()

        return observation, reward, termination, info

    def _check_termination(self):
        """ Query state of environment to check termination condition

        Check if end effector position is within some absolute distance
        radius of the goal position or if maximum steps in episode have
        been reached.

            Args: None
            Returns: bool if episode has terminated or not.
        """
        if (self.num_steps > self.MAX_STEPS):
            logging.debug("Maximum steps reached")
            return True
        else:
            if self.grasped == True:
                return True
            return False

    def visualize(self, observation, action):
        """Visualize the action - that is,
        add visual markers to the world (in case of sim)
        or execute some movements (in case of real) to
        indicate the action about to be performed.

        Args:
            observation: The observation of the current step.
            action: The selected action.
        """
        pass

    def handle_exception(self, e):
        """Handle an exception.
        """
        pass

    def info(self):
        return {}

    def rewardFunction(self):
        """Implement reward function here.
        """
        return -1
