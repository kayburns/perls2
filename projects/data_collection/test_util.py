from util import DatasetUtils

d_util = DatasetUtils(dataset_loc="/home/mason/peg_insertation_dataset/heuristic_data_contact/", folder_name="example_")

dataset_arr = d_util.recall_obs()

print (dataset_arr[0]["contact"])