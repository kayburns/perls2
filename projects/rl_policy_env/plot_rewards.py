import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

monitor_loc = "projects/rl_policy_env/policy_log/monitor.csv"
monitor_info = np.genfromtxt(monitor_loc, delimiter=",", skip_header=2)[:2000]

#mean_rewards = np.load("projects/rl_policy_env/policy_log/mean_rewards_arr.npy")

#ep_count = mean_rewards.shape[0] 
print (f"{monitor_info.shape}")
ep_count = monitor_info.shape[0]
x_var = np.linspace(1, ep_count, num=ep_count)

ep_rewards = monitor_info[:, 0]
ep_lengths = monitor_info[:, 1]

mean_rewards = ep_rewards / ep_lengths
plt.xlabel("Number of episodes")
plt.ylabel("Average Episode Reward")

plt.plot(x_var, mean_rewards)
plt.show()

