from PIL import Image
import numpy as np
import os


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

    def save_obs(self, observation_array):
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
            self.example_count += 1

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
        op_component["proprio"] = np.load(os.path.join(save_location, f"proprio_{idx}.png"))

    
    def recall_obs(self):
        """
            Returns the dataset at a given save location.
            Arguments:
                - dataset_loc (str): location of the dataset folder
            Returns:
                - list [list[dict, dict, np_array]] - dataset as a list of training examples
        """
        op_list = []
        folders_list = [f.path for f in os.scandir(self.dataset_loc) if f.is_dir()]
        for example_dir in folders_list:
            first_comp = self._recall_component(example_dir, idx=0)
            second_comp = self._recall_component(example_dir, idx=1)

            action_arr = np.load(os.path.join(example_dir, f"action_vec.npy"))
            op_list.append([first_comp, second_comp, action_arr])
        return op_list