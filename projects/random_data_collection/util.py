from PIL import Image
import numpy as np
import os
from numpy.lib.npyio import save
from tqdm import tqdm
import cv2

class DatasetUtils:
    def __init__(self, dataset_loc="/home/mason/peg_insertation_dataset/heuristic_data/", folder_name="example_"):
        self.dataset_loc = dataset_loc
        self.folder_name = folder_name
        self.example_count = 0
        pass

    def _save_component(self, obs_component, save_location, idx = 0):
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
        np.save(os.path.join(save_location, f"proprio_{idx}.npy"), robot_proprio)

    def _save_example(self, first_comp, sec_comp, save_location):
        cam_clr_img = Image.fromarray(first_comp["cam_color"])
        cam_clr_img.save(os.path.join(save_location, f"cam_color.png"))
        cam_dpth_img = Image.fromarray(first_comp["cam_depth"]).convert("L")
        cam_dpth_img.save(os.path.join(save_location, f"cam_depth.png"))

        tacto_clr_0 = Image.fromarray(first_comp["digits_color"][0])
        tacto_clr_0.save(os.path.join(save_location, f"digits_color_0.png"))
        tacto_clr_1 = Image.fromarray(first_comp["digits_color"][1])
        tacto_clr_1.save(os.path.join(save_location, f"digits_color_1.png"))

        optical_flow, _ = self._get_optical_flow(first_comp, sec_comp)
        np.save(os.path.join(save_location, "flow.npy"), optical_flow)
        #optical_flow_img = Image.fromarray(optical_flow)
        #optical_flow_img.save(os.path.join(save_location, "optical_flow.png"))

        np.save(os.path.join(save_location, f"digits_depth_0.npy"), first_comp["digits_depth"][0])
        np.save(os.path.join(save_location, f"digits_depth_1.npy"), first_comp["digits_depth"][1])
        ee_yaw_next = sec_comp["proprio"][:3]

        np.save(os.path.join(save_location, "ee_yaw_next.npy"), ee_yaw_next)

        robot_proprio = first_comp["proprio"]
        np.save(os.path.join(save_location, f"proprio.npy"), robot_proprio)

    def save_obs(self, observation_array):
        for observation in tqdm(observation_array):
            example_idx = self.example_count
            save_location = os.path.join(self.dataset_loc, self.folder_name + str(example_idx)) + "/"
            if not os.path.exists(save_location):
                os.mkdir(save_location)
            
            first_comp = observation[0]
            sec_comp = observation[1]
            action = observation[2]
            peg_contact = observation[3]

            #NOTE: I have appended the peg contct bool as a float to the end of the action vector
            action = np.append(action, [float(peg_contact)], axis=0)
            self._save_example(first_comp, sec_comp, save_location)

            np.save(os.path.join(save_location, f"action_vec.npy"), action)
            self.example_count += 1
            


    """def save_obs(self, observation_array):
        #Create Example Folder
        for observation in observation_array:
            example_idx = self.example_count
            save_location = os.path.join(self.dataset_loc, self.folder_name + str(example_idx)) + "/"
            if not os.path.exists(save_location):
                os.mkdir(save_location)

            first_comp = observation[0]
            sec_comp = observation[1]
            action = observation[2]

            self._save_component(first_comp, save_location, idx=0)
            self._save_component(sec_comp, save_location, idx=1)

            np.save(os.path.join(save_location, f"action_vec.npy"), action)
            self.example_count += 1"""

    def _recall_component(self, save_location, idx=0):
        op_component = {}
        cam_clr_img = Image.open(os.path.join(save_location, f"cam_color_{idx}.png"))
        op_component["cam_color"] = np.asarray(cam_clr_img)

        cam_dpth_img = Image.open(os.path.join(save_location, f"cam_depth_{idx}.png"))
        op_component["depth_color"] = np.asarray(cam_dpth_img)

        tacto_clr_0 = Image.open(os.path.join(save_location, f"digits_color_0_{idx}.png"))
        tacto_clr_1 = Image.open(os.path.join(save_location, f"digits_color_1_{idx}.png"))
        op_component["digits_color"] = [np.asarray(tacto_clr_0), np.asarray(tacto_clr_1)]

        op_component["digits_depth"] = [np.load(os.path.join(save_location, f"digits_depth_0_{idx}.npy")),
                                    np.load(os.path.join(save_location, f"digits_depth_1_{idx}.npy"))]
        op_component["proprio"] = np.load(os.path.join(save_location, f"proprio_{idx}.npy"))

        return op_component

    def _recall_example(self, save_location):
        op_example = {}
        cam_clr_img = Image.open(os.path.join(save_location, f"cam_color.png"))
        op_example["cam_color"] = np.asarray(cam_clr_img)
        cam_clr_img = Image.open(os.path.join(save_location, f"cam_depth.png"))
        op_example["cam_depth"] = np.asarray(cam_clr_img)

        tacto_clr_0 = Image.open(os.path.join(save_location, f"digits_color_0.png"))
        tacto_clr_1 = Image.open(os.path.join(save_location, f"digits_color_1.png"))
        op_example["digits_color"] = [np.asarray(tacto_clr_0), np.asarray(tacto_clr_1)]

        op_example["digits_depth"] = [np.load(os.path.join(save_location, f"digits_depth_0.npy")),
                                    np.load(os.path.join(save_location, f"digits_depth_1.npy"))]
        op_example["proprio"] = np.load(os.path.join(save_location, f"proprio.npy"))

        #optical_flow_img = Image.open(os.path.join(save_location, f"optical_flow.png"))
        op_example["optical_flow"] = np.load(os.path.join(save_location, "flow.npy")) #np.asarray(optical_flow_img)

        action_arr = np.load(os.path.join(save_location, f"action_vec.npy"))
        contact_bool = (action_arr[-1] == 1.0)

        op_example["contact"] = contact_bool
        op_example["action"] = action_arr[:-1]
        
        yaw_next = np.load(os.path.join(save_location, f"ee_yaw_next.npy"))
        op_example["yaw_next"] = yaw_next
        #print (f"Action Shape: {op_example['action'].shape}")
        return op_example

    def _get_optical_flow(self, first_comp, sec_comp):
        first_image = first_comp["cam_color"]
        sec_image = sec_comp["cam_color"]

        #mask = np.zeros_like(first_image)

        first_gray = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY)
        sec_gray = cv2.cvtColor(sec_image, cv2.COLOR_RGB2GRAY)

        #print (f"{first_gray.shape}")
        flow = cv2.calcOpticalFlowFarneback(first_gray, sec_gray, 
                                            None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        #print (f"========================flow shape: {flow.shape}==============================")
        #print (f"flow shape: {flow.shape}")
        #mask[..., 1] = 255

        #magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #print (f"angle shape: {angle.shape}")

        #mask[..., 0] = angle * 180 / np.pi / 2
      
        # Sets image value according to the optical flow
        # magnitude (normalized)
        #mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
      
        # Converts HSV to RGB (BGR) color representation
        #rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)

        flow_mask = np.expand_dims(
                        np.where(
                            flow.sum(axis=2) == 0, 
                            np.zeros_like(flow.sum(axis=2)), 
                            np.ones_like(flow.sum(axis=2)),
                        ),
                        2,
                )
        return flow, flow_mask

    def recall_obs(self):
        op_list = []
        folders_list = [f.path for f in os.scandir(self.dataset_loc) if f.is_dir()]

        for example_dir in tqdm(folders_list[:1000]):
            example = self._recall_example(example_dir)
            op_list.append(example)
            pass

        return op_list

    """def recall_obs(self):
        
            Returns the dataset at a given save location.
            Arguments:
                - dataset_loc (str): location of the dataset folder
            Returns:
                - list [list[dict, dict, np_array]] - dataset as a list of training examples
        
        op_list = []
        folders_list = [f.path for f in os.scandir(self.dataset_loc) if f.is_dir()]
        for example_dir in tqdm(folders_list[:1000]): #tqdm(folders_list):
            first_comp = self._recall_component(example_dir, idx=0)
            second_comp = self._recall_component(example_dir, idx=1)

            action_arr = np.load(os.path.join(example_dir, f"action_vec.npy"))

            element_dict = {
                "pre_action": first_comp,
                "post_action": second_comp,
                "action": action_arr
            }

            optical_flow_img, flow_mask = self._get_optical_flow(element_dict)
            element_dict["flow"] = optical_flow_img
            element_dict["flow_mask"] = flow_mask
            
            #op_list.append([first_comp, second_comp, action_arr])
            op_list.append(element_dict)
        return op_list"""