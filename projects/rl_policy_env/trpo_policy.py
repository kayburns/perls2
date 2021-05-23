import gym

from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines import TRPO
from rl_policy_env import RLPolicyEnv

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)

#env = gym.make('CartPole-v1')
env = RLPolicyEnv('projects/rl_policy_env/rl_policy.yaml', False, "TemplateEnv")
#policy = FeedForwardPolicy(net_arch=[128, 128])
model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
#model.save("trpo_cartpole")

#del model # remove to demonstrate saving and loading

#model = TRPO.load("trpo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    #env.render()

    if done:
        print ("RESETTING ENVIRONMENT")
        obs = env.reset()

