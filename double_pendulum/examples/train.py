from stable_baselines3 import PPO
import gym

import double_cartpole_custom_gym_env

env = gym.make('double-cartpole-custom-v0', render_sim=False, n_steps=1000)
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000000)
model.save('double_pendulum')
