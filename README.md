# RoboLearn Python package
"A Python package for Robot Learning"

<p align="center">
<img src="robolearn_logo2.png" alt="robolearn_logo" width="100" height="100" class="center" />
</p>

**Robolearn** is a python package, mainly focused on learning control, that defines common interfaces
between robot learning algorithms and real/simulated robots.

**This package is gradually becoming public**, so the public version is still in 
development. Sorry for any inconvenience.

![robolearn diagram](robolearn_diagram.png)


## Algorithms
- Reinforcement Learning
    - Model-Based RL
        - Guided Policy Search
            - [ ] MDGPS
            - [ ] DMDGPS
    - Model-Free RL
        - Value-based
            - [ ] Deep Q-Learning (DQL)
            - [ ] Soft Q-Learning (SQL)
        - Policy-based
            - [ ] REINFORCE
            - [ ] PPO
        - Actor-Critic
            - [ ] Deep Deterministic Policy Gradients (DDPG)
            - [ ] Soft Actor Critic (SAC)

- Inverse Reinforcement Learning

- Imitation Learning

- Trajectory Optimization
    - Indirect Methods
        - iLQR
    - [ ] Direct Methods


## Robot Interfaces
- [ ] Simulation
    - [ ] PyBullet
    - [ ] GAZEBO
- [ ] Real

# Installation

```bash
git clone https://github.com/domingoesteban/robolearn
cd robolearn
pip install -e .
```

# Citation
If you use this code or it was useful for something else,
we would appreciate that you can cite:

    @misc{robolearn,
      author = {Esteban, Domingo},
      title = {RoboLearn},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/domingoesteban/robolearn}},
    }

<!--
# Acknowledgements
- Vitchyr Pong for rlkit repository ([rlkit repository](https://github.com/vitchyr/rlkit)). Some algorithms are based (or almost the same) the ones in rlkit. Many functionalities of robolearn use code from rlkit.
- Tuomas Haarnoja for softqlearning repository ([softqlearning repository](https://github.com/haarnoja/softqlearning)). SoftQLearning is based in this TensorFlow implementation.
-->
