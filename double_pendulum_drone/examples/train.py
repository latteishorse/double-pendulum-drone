from stable_baselines3 import PPO
import gym

import double_pendulum_drone_env

env = gym.make('double_pendulum_drone', render_sim=False, n_steps=500, render_path=True, render_shade=True,
            shade_distance=70,  n_fall_steps=10, change_target=True, initial_throw=True)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000000)
model.save('double_pendulum_drone_model')
