from stable_baselines3 import PPO
import gym
import time
import sys

import double_pendulum_drone_env

continuous_mode = True 
random_action = False 
render_sim = True 

env = gym.make('double_pendulum_drone', render_sim=render_sim, n_steps=500, render_path=True, render_shade=True,
            shade_distance=70,  n_fall_steps=100, change_target=True, initial_throw=True)

model = PPO.load("double_pendulum_drone_model.zip")
model.set_env(env)
random_seed = int(time.time())
model.set_random_seed(random_seed)

obs = env.reset()

try:
    while True:
        if render_sim:
            env.render()

        if random_action:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        if done is True:
            if continuous_mode is True:
                state = env.reset()
            else:
                break

finally:
    env.close()
