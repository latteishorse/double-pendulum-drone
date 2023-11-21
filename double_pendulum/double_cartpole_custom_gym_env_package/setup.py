from setuptools import setup, find_packages
import os

setup(
    name='double_cartpole_custom_gym_env',
    version='1.0.0',
    url='https://github.com/latteishorse/double-pendulum-drone',
    packages=find_packages(),
    include_package_data = True,
    install_requires=['gym', 'pygame', 'pymunk', 'numpy', 'stable-baselines3'],
    keywords=['double pendulum', 'reinforcement learning']
)
