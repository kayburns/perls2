"""Template Environment for Projects.
"""
from perls2.envs.env import Env
import numpy as np
import tacto
import logging
import os
import pybullet as p
import gym.spaces as spaces
import torch
import sys

#sys.path.append('/home/mason/multimodal_tacto_repo/multimodal/')
from models.tacto_sensor_fusion import SensorFusionSelfSupervised

FEATURE_LEN = 128

class RLPolicyEnv(Env):
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
                            "PEG_GRAB": {"method": self._peg_grab_exec, "next": None},
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
        self.PLANAR_RADIUS = self.CONV_RADIUS
        self.GOAL_RADIUS = self.PLANAR_RADIUS * 0.1

        self.FAIL_RADIUS = 0.2
        self.PEG_GRIP_MAX_DIST = 0.15

        self.tacto_add_objects()

        #Robot Setup
        self.reset_position = self.robot_interface.ee_position
        self._initial_ee_orn = self.robot_interface.ee_orientation

        #Peg and hole initial_positions
        self.peg_initial_pos = [0, 0, 0]
        self.hole_initial_pos = [0, 0, 0]

        self.observation_space = spaces.Box(np.full((FEATURE_LEN), -np.inf), np.full((FEATURE_LEN), np.inf), dtype=np.float32)


        #Torch Initialization
        self.device = torch.device("cuda")
        self.z_dim = 128
        self.action_dim = 3
        self.model = SensorFusionSelfSupervised(
            device=self.device,
            encoder=False,
            deterministic=False,
            z_dim=self.z_dim,
            action_dim=self.action_dim,
        ).to(self.device)


        self.model_path = '/home/mason/multimodal_tacto_rep/multimodal/no_tacto_logging/models/weights_itr_49.ckpt' #'/home/mason/multimodal_tacto_rep/multimodal/logging/202105312206_/models/weights_itr_48.ckpt'

        self.load_model(self.model_path)

    def load_model(self, path):
        print("Loading model from {}...".format(path))
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt)
        self.model.eval()

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
            print (f"Object Path: {object_path}")
            
            object_path = os.path.join(data_dir, object_path)
            pb_obj_id = self.world.arena.object_dict[object_name]

            self.scale_dict[object_name] = scale
            self.digits.add_object(object_path, pb_obj_id, globalScaling=scale)
            print (f"DIGIT ADDED: {object_name}")

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
    
    def model_encode(self, img, depth, tacto_clr, tacto_depth, proprio):
        mdl_img = torch.from_numpy(np.expand_dims(img, axis=0).astype(np.float32)).to(self.device)
        #print (f"depth_shape: {torch.from_numpy(np.expand_dims(depth, axis=0)).shape}")

        mdl_depth = np.expand_dims(depth, axis=2)
        mdl_depth = np.expand_dims(mdl_depth, axis=0).astype(np.float32)

        mdl_depth = torch.from_numpy(mdl_depth).transpose(1, 3).transpose(2, 3).to(self.device)
        mdl_tacto_clr = torch.from_numpy(np.expand_dims(tacto_clr, axis=0).astype(np.float32)).to(self.device)

        mdl_tacto_depth = np.expand_dims(tacto_depth, axis=3)
        mdl_tacto_depth = np.expand_dims(mdl_tacto_depth, axis=0).astype(np.float32)

        mdl_tacto_depth = torch.from_numpy(mdl_tacto_depth).to(self.device)
        mdl_proprio = torch.from_numpy(np.expand_dims(proprio, axis=0).astype(np.float32)).to(self.device)

        op = self.model.forward_encoder(mdl_img, mdl_tacto_clr, mdl_tacto_depth, mdl_proprio, mdl_depth, None)

        #print (f"output: {op}")
        return op
        
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
        proprio = self.robot_interface.link_position(self.focus_point_link, computeVelocity=True)[:8]
        cam_frames = self.world.camera_interface.frames()
        digits_color, digits_depth = self.digits.render()
        
        
        obs = {"cam_color": cam_frames["rgb"], 
                "cam_depth": cam_frames["depth"], 
                "digits_depth": 0 * digits_depth, 
                "digits_color": 0 * digits_color,
                "proprio": proprio}

        #print(f"===================HERE===========")
        _, _, _, _, _, z, = self.model_encode(cam_frames["rgb"], cam_frames["depth"], digits_color, digits_depth, proprio)
        #print (f"output: {type(z)}")

        out_vec = z.cpu().detach().numpy()[0]
        return out_vec

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

    def _peg_hole_distance(self):
        peg_scale = self.scale_dict["peg"]
        goal_position = self._hole_position() 
        goal_position[2] += peg_scale * self.PEG_H + self._grab_height_offset()

        return self._get_dist_to_goal(self.focus_point_link, goal_position)

    def _peg_grip_distance(self):
        peg_scale = self.scale_dict["peg"]
        grip_position = self.robot_interface.link_position(self.focus_point_link)
        peg_position = self.peg_interface.position

        dist = np.linalg.norm((grip_position - peg_position))
        return dist
    
    def _peg_in_grasp(self):
        peg_grip_dist = self._peg_grip_distance()
        if self.curr_state != self.term_state:
            return True
        return peg_grip_dist <= self.PEG_GRIP_MAX_DIST

        
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

        #Defer to state machine for Peg grabbing Actions
        if self.curr_state != self.term_state:
            exec_method = self.state_dict[self.curr_state]["method"]
            exec_method()
        else:
            move_delta = action
            curr_position = self.robot_interface.link_position(self.focus_point_link)[:3]
            new_position = curr_position + action
            self.robot_interface.set_link_pose_position_control(self.focus_point_link, new_position, self._initial_ee_orn)

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

        return observation #np.zeros((FEATURE_LEN), np.float32)

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
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._exec_action(action)
        
        self.world.step(start)
        self.num_steps = self.num_steps + 1

        termination = self._check_termination()

        # if terminated reset step count
        if termination:
            self.num_steps = 0

        reward = self.rewardFunction()

        info = self.info()

        post_action_obs = self.get_observation()

        #This is a placeholder for the learned multimodal representation
        obs_palceholder = np.zeros((FEATURE_LEN), np.float32)

        return post_action_obs, reward, termination, info

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
        if (self.curr_state != self.term_state):
            return False
        else:
            peg_hole_dist = self._peg_hole_distance()
            if peg_hole_dist <= self.GOAL_RADIUS or peg_hole_dist > self.FAIL_RADIUS:
                #print ("================out of bounds trip=================")
                return True
            if self._peg_in_grasp() == False:
                #print ("=============out of grasp trip==============")
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
        peg_scale = self.scale_dict["peg"]
        goal_position = self._hole_position() 
        goal_position[2] += peg_scale * self.PEG_H + self._grab_height_offset()

        goal_delta = self._get_delta_to_goal(self.focus_point_link, goal_position)

        rel_delta = np.abs(goal_delta)

        xy_dist = np.linalg.norm(rel_delta[:2])
        full_dist = np.linalg.norm(rel_delta)

        lam = 0.1
        cr = 5
        reach_reward = cr - (0.5 * cr) * (np.tanh(lam *full_dist) + np.tanh(lam * xy_dist))
        z_thresh = 0.05
        #x_re = (1 - np.tanh(rel_delta[0]))
        #y_re = (1 - np.tanh(rel_delta[1]))
        #z_re = 0.0
        al_reward = 0.0
        in_reward = 0.0
        insertion_happened = 0.0
        #planar_dist = np.linalg.norm(rel_delta[:2])
        if xy_dist <= self.PLANAR_RADIUS:
            ca = 0.2
            al_reward = 2 - ca * xy_dist

            if rel_delta[2] < z_thresh:
                in_reward = 4 - 2 * (rel_delta[2] / (self._grab_height_offset()))
            if np.linalg.norm(goal_delta) <= self.GOAL_RADIUS:
                #Insertion Happened
                insertion_happened += 10

        insertion_reward = reach_reward + al_reward + in_reward + insertion_happened
        
        #print (f"Reward: {insertion_reward}")

        peg_grasp_float = float(self._peg_in_grasp())
        return peg_grasp_float * insertion_reward
