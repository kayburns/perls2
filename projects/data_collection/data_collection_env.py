"""Template Environment for Projects.
"""
from perls2.envs.env import Env
import numpy as np
import tacto
import logging
import os
import pybullet as p


class DataCollectionEnv(Env):
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
        #Movement State Machine Setup
        self.state_list = ["PEG_SETUP", "PEG_GRAB", "PEG_MOVING", "PEG_COMPLETE"]
        self.state_dict = {
                            "PEG_SETUP": {"method": self._peg_setup_exec, "next": "PEG_GRAB"},
                            "PEG_GRAB": {"method": self._peg_grab_exec, "next": "PEG_MOVING"},
                            "PEG_MOVING": {"method": self._peg_move_exec, "next": "PEG_COMPLETE"},
                            "PEG_COMPLETE": {"method": self._peg_complete_exec, "next": None},
                          }
        
        self.peg_interface = self.world.object_interfaces['peg']
        self.hole_interface = self.world.object_interfaces['hole_box']

        self.term_state = None #"PEG_MOVING" #None #"PEG_COMPLETE"

        #Object Information
        self.BOX_W = 1.0
        self.BOX_H = 0.2
        self.HOLE_W = 0.2

        self.PEG_H = 0.3
        self.PEG_W = 0.2

        self.scale_dict = {}

        #Tacto Setup
        self.digits = tacto.Sensor(**self.config["tacto"])

        self.left_joint_idx = self.robot_interface.get_joint_id_from_name('joint_finger_tip_left')
        self.right_joint_idx = self.robot_interface.get_joint_id_from_name('joint_finger_tip_right')

        self.left_finger_link = self.robot_interface.get_link_id_from_name('finger_left_tip')
        self.right_finger_link = self.robot_interface.get_link_id_from_name('finger_right_tip')

        self.gripper_base_link = self.robot_interface.get_link_id_from_name('gripper_base_link')
        self.digits.add_camera(self.robot_interface.arm_id, [self.left_joint_idx, self.right_joint_idx])

        self.focus_point_link = self.robot_interface.get_link_id_from_name('focus_point')
        self.ee_point = self.focus_point_link

        self.CONV_RADIUS = 0.05
        self.tacto_add_objects()

        #Robot Setup
        self.reset_position = self.robot_interface.ee_position
        self._initial_ee_orn = self.robot_interface.ee_orientation

        #Peg and hole initial_positions
        self.peg_initial_pos = [0, 0, 0]
        self.hole_initial_pos = [0, 0, 0]

    def tacto_add_objects(self):
        """
            Adds the objects in the configuration file to the Tacto sim
        """
        if not isinstance(self.config["object"], dict):
            return
        if ("object_dict" in self.config["object"].keys()) == False:
            return

        data_dir = self.config["data_dir"]
        for obj_idx, obj_key in enumerate(self.config["object"]["object_dict"]):
            object_name = self.config["object"]["object_dict"][obj_key]["name"]
            object_path = self.config["object"]["object_dict"][obj_key]["path"]
            scale = self.config["object"]["object_dict"][obj_key]["scale"]
            #print (f"Object Path: {object_path}")
            
            object_path = os.path.join(data_dir, object_path)
            pb_obj_id = self.world.arena.object_dict[object_name]

            self.scale_dict[object_name] = scale
            self.digits.add_object(object_path, pb_obj_id, globalScaling=scale)
            #print (f"DIGIT ADDED: {object_name}")

    def should_record(self):
        """Returns whether or not the observations should be recorded for data collection"""
        return (self.curr_state == "PEG_MOVING" or self.curr_state == "PEG_COMPLETE")

    def _hole_position(self):
        hole_scale = self.scale_dict["hole_box"]
        box_pos = self.hole_interface.position

        box_side_w = 0.5 * (self.BOX_W - self.HOLE_W)
        offset = hole_scale * (0.5 * box_side_w + 0.5 * self.HOLE_W)

        box_pos[1] += offset
        return box_pos
    
    def _grab_height_offset(self):
        peg_scale = self.scale_dict["peg"]
        return 0.5 * peg_scale * self.PEG_H #-0.02

    def _get_dist_to_goal(self, link_id, goal_position):
        current_ee_pos = np.asarray(self.robot_interface.link_position(link_id))
        diff = (goal_position - current_ee_pos)
        abs_dist = np.linalg.norm(diff)

        return abs_dist
    
    def _get_delta_to_goal(self, link_id, goal_position):
        current_ee_pos = np.asarray(self.robot_interface.link_position(link_id))
        diff = (goal_position - current_ee_pos)
        return diff
    
    def _peg_touching_box(self):
        contact_list = p.getContactPoints(self.peg_interface.obj_id, self.hole_interface.obj_id)
        return len(contact_list) > 0

    def get_observation(self):
        """Get observation of current env state

        Returns:
            observation (dict): dictionary with key values corresponding to
                observations of the environment's current state.

            Dict:
            - cam_color: color output from the camera (np_array)
            - cam_depth: depth output from the camera (np_array)
            - digits_depth: depth output from the figertip sensors [np_array, np_array] (one for each finger)
            - digits_color: color output from the fingertip sensors [np_array, np_array] (one for each finger)
            - proprio: 7f list of floats of joint positions in radians
        """
        #proprio = self.robot_interface.q
        proprio = self.robot_interface.link_position(self.focus_point_link, computeVelocity=True)
        #print (f"proprio: {proprio}")

        cam_frames = self.world.camera_interface.frames()
        digits_color, digits_depth = self.digits.render()
        
        obs = {"cam_color": cam_frames["rgb"], 
                "cam_depth": cam_frames["depth"], 
                "digits_depth": digits_depth, 
                "digits_color": digits_color,
                "proprio": proprio}

        
        return obs

    def _peg_setup_exec(self):
        #print ("PEG_SETUP")
        object_pos = self.peg_interface.position
        object_pos[2] += 0.3 #self._grab_height_offset() + 0.1
        goal_position = object_pos

        #print (f"PEG_POS: {self.peg_interface.position}")
        #print (f"GOAL_POS: {goal_position}")

        self.robot_interface.set_link_pose_position_control(self.ee_point, goal_position, self._initial_ee_orn)
        if self._get_dist_to_goal(self.ee_point, goal_position) <= self.CONV_RADIUS:
            self.curr_state = self.state_dict[self.curr_state]["next"]

        return goal_position

    def _peg_grab_exec(self):
        #print ("PEG_GRAB")
        object_pos = self.peg_interface.position
        object_pos[2] += self._grab_height_offset() #+ 0.1
        goal_position = object_pos
        self.robot_interface.set_link_pose_position_control(self.ee_point, goal_position, self._initial_ee_orn)
        
        if self._get_dist_to_goal(self.ee_point, goal_position) <= self.CONV_RADIUS:
            self.curr_state = self.state_dict[self.curr_state]["next"]
        
            self.robot_interface.set_gripper_to_value(0.4)
            self.grasped = True
        
        return goal_position
        

    def _peg_move_exec(self):
        #print ("PEG_MOVE")
        hole_scale = self.scale_dict["hole_box"]
        peg_scale = self.scale_dict["peg"]

        goal_position = self._hole_position()
        peg_z = peg_scale * self.PEG_H * 0.5 + hole_scale * self.BOX_H * 0.5

        #goal_position[2] = self.peg_interface.position[2] + self._grab_height_offset()
        goal_position[2] = peg_z + self._grab_height_offset() + 0.02

        focus_pos = self.robot_interface.link_position(self.ee_point)
        #goal_position[2] = focus_pos[2]

        goal_delta = self._get_delta_to_goal(self.ee_point, goal_position)
        goal_delta = goal_delta / np.linalg.norm(goal_delta)

        move_rate = 0.02
        goal_delta = move_rate * goal_delta

        interim_goal = self.robot_interface.link_position(self.ee_point) + goal_delta
        #interim_goal[2] = self.peg_interface.position[2] + self._grab_height_offset()

        #self.robot_interface.move_ee_delta(goal_list, set_ori=self._initial_ee_orn)
        self.robot_interface.set_link_pose_position_control(self.ee_point, interim_goal, self._initial_ee_orn)
        #print (f"GOAL_DIST: {self._get_dist_to_goal(self.ee_point, goal_position)}")

        focus_pos = self.robot_interface.link_position(self.ee_point)
        #print (f"Self h: {focus_pos[2]} Goal h: {interim_goal[2]}")

        if self._get_dist_to_goal(self.ee_point, goal_position) <= 0.01: #0.005:
            self.curr_state = self.state_dict[self.curr_state]["next"]
        
            #self.robot_interface.set_gripper_to_value(0.5)
            self.grasped = True

        return interim_goal
        

    def _peg_complete_exec(self):
        #print ("PEG_DOWN")
        peg_scale = self.scale_dict["peg"]
        goal_position = self._hole_position() 
        goal_position[2] += peg_scale * self.PEG_H + self._grab_height_offset()

        self.robot_interface.set_link_pose_position_control(self.focus_point_link, goal_position, self._initial_ee_orn)
        #print (f"GOAL_DIST: {self._get_dist_to_goal(self.focus_point_link, goal_position)}")
        if self._get_dist_to_goal(self.focus_point_link, goal_position) <= 0.005:
            self.curr_state = self.state_dict[self.curr_state]["next"]
        
            self.robot_interface.set_gripper_to_value(0.0)
            self.grasped = True
        
        return goal_position


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

        #Defer to state machine for actions
        pre_action_ee_pos = np.asarray(self.robot_interface.link_position(self.focus_point_link))
        exec_method = self.state_dict[self.curr_state]["method"]
        action_pos = exec_method()

        post_action_ee_pos = np.asarray(self.robot_interface.link_position(self.focus_point_link))
        pos_delta = post_action_ee_pos - pre_action_ee_pos

        return pos_delta #action_pos

    def _get_table_pos(self):
        table_id = self.world.arena.scene_objects_dict["table"]
        table_pos, _ = p.getBasePositionAndOrientation(
            table_id, self.world._physics_id)
        return table_pos


    def _random_reset_objects(self):
        table_pos = self._get_table_pos()

        box_side_w = 0.5 * (self.BOX_W - self.HOLE_W)

        #Reset the position of the hole
        hole_scale = self.scale_dict["hole_box"]
        hole_height = hole_scale * self.BOX_H
        hole_pos = self.hole_interface.position

        hole_pos = np.random.uniform(low=[0.6, -0.2, 0.0], high=[0.9, 0.2, 0.0])
        #hole_pos = np.random.multivariate_normal([0.8, 0.0, 0.0], hole_scale * 0.05 * np.eye(3))

        hole_pos[2] = (hole_height) + table_pos[2]

        self.hole_initial_pos = hole_pos
        self.hole_interface.set_position(hole_pos)

        #Reset the position of the peg
        peg_scale = self.scale_dict["peg"]
        peg_pos = hole_pos
        x_range = hole_scale * 0.5 * self.BOX_W - 0.5 * peg_scale * self.PEG_W
        peg_pos[0] -= np.random.uniform(-1.0 * x_range, x_range, size=None)

        y_range = hole_scale * 0.5 * box_side_w - peg_scale * 0.5 * self.PEG_W

        peg_pos[1] += np.random.randint(1, 2, size=None) * hole_scale * (box_side_w + self.HOLE_W) + np.random.uniform(-1.0 * y_range, y_range)

        peg_pos[2] += peg_scale * self.PEG_H * 0.5 + hole_scale * self.BOX_H * 0.5

        self.peg_initial_pos = peg_pos
        self.peg_interface.set_position(peg_pos)
        self.peg_interface.set_orientation(p.getQuaternionFromEuler([0, 0, 0]))

    def _reset_objects(self):
        table_pos = self._get_table_pos()

        #Reset the position of the hole
        hole_scale = self.scale_dict["hole_box"]
        hole_height = hole_scale * self.BOX_H
        #hole_pos = self.hole_interface.position
        hole_pos = np.array([0.9, -0.2, 0.0])

        hole_pos[2] = (hole_height) + table_pos[2]

        self.hole_initial_pos = hole_pos
        self.hole_interface.set_position(hole_pos)

        #Reset the position of the peg
        peg_scale = self.scale_dict["peg"]
        peg_pos = hole_pos
        peg_pos[0] += (hole_scale * self.BOX_W * 0.5) - (peg_scale * self.PEG_W * 0.5)
        
        peg_pos[2] += peg_scale * self.PEG_H * 0.5 + hole_scale * self.BOX_H * 0.5

        self.peg_initial_pos = peg_pos
        self.peg_interface.set_position(peg_pos)
        self.peg_interface.set_orientation(p.getQuaternionFromEuler([0, 0, 0]))
        
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

        observation = self.get_observation()

        #self._reset_objects()
        self._random_reset_objects()

        self.robot_interface.set_gripper_to_value(0.0)
        self.curr_state = self.state_list[0]

        return [0, 0, 0]

    def step(self, action, start=None):
        """Take a step.

        Args:
            action: The action to take.
        Returns:
            -observation: based on user-defined functions
            -reward: from user-defined reward function
            -done: whether the task was completed or max steps reached
            -info: info about the episode including success

        Observation: [(frames before action was taken), (frames after action was taken), action]
        Takes a step forward similar to openAI.gym's implementation.
        """
        pre_action_obs = self.get_observation()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_pos = self._exec_action(action)
        self.world.step(start)
        self.num_steps = self.num_steps + 1

        termination = self._check_termination()

        # if terminated reset step count
        if termination:
            self.num_steps = 0

        reward = self.rewardFunction()

        #print (f"PEG_CONTACT: {self._peg_touching_box()}")
        
        peg_contact_bool = self._peg_touching_box()

        post_action_observation = self.get_observation()

        info = self.info()

        obs_arr = [pre_action_obs, post_action_observation, action_pos, peg_contact_bool]
        return obs_arr, reward, termination, info

    def _check_termination(self):
        """ Query state of environment to check termination condition

        Check if end effector position is within some absolute distance
        radius of the goal position or if maximum steps in episode have
        been reached.

            Args: None
            Returns: bool if episode has terminated or not.
        """
        if (self.num_steps > self.MAX_STEPS):
            return True
        else:
            if self.curr_state == self.term_state:
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
