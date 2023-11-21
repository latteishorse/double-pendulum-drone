from double_pendulum_drone_env.double_pendulum_drone_env import *
from gym.envs.registration import register

register(
    id='double_pendulum_drone',
    entry_point='double_pendulum_drone_env:DPDEnv',
    kwargs={'render_sim': False,'n_steps': 500,  'render_path': True, 'render_shade': True,
            'shade_distance': 75, 'n_fall_steps': 10, 'change_target': False,
            'initial_throw': True})
