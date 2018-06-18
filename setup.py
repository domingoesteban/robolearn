# setup.py
from setuptools import setup

setup(
    name='robolearn',
    version='0.2.0',
    description='A Robot-Learning package: Robot reinforcement learning.',
    maintainer="Domingo Esteban",
    maintainer_email="domingo.esteban@iit.it",
    packages=['robolearn'],
    install_requires=[
        'gym',
        'numpy',
        'torch',
        'robolearn_gym_envs'
    ],
)
