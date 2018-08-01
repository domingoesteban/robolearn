# RoboLearn Python package

"A Python package for Robot Learning"

**THIS PACKAGE IS STILL IN DEVELOPMENT**

## Algorithms
- [x] Robot Reinforcement Learning
    - [ ] Model-Based RL
        - [x] Guided Policy Search
            - MDGPS
            - DMDGPS
    - [ ] Model-Free RL
        - [x] Value-based
            - Soft Q-Learning (SQL)
        - [x] Policy-based
            - REINFORCE
        - [ ] Actor-Critic
            - Deep Deterministic Policy Gradients (DDPG)
            - Soft Actor Critic (SAC)

- [x] Trajectory Optimization
    - [ ] Indirect Methods
        - iLQR
    - [ ] Direct Methods

- [ ] Robot Inverse Reinforcement Learning

- [ ] Robot Imitation Learning

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

# Acknowledgements
- Vitchyr Pong for rlkit repository ([rlkit repository](https://github.com/vitchyr/rlkit)). Some algorithms are based (or almost the same) the ones in rlkit. Many functionalities of robolearn use code from rlkit.
- Tuomas Haarnoja for softqlearning repository ([softqlearning repository](https://github.com/haarnoja/softqlearning)). SoftQLearning is based in this TensorFlow implementation.

