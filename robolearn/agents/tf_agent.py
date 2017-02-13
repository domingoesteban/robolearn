
from robolearn.agents.base import Agent
import tensorflow as tf
import numpy as np

class TFAgent(Agent):

    def __init__(self, act_dim, obs_dim):
        super(TFAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim)

        self.state = tf.placeholder(tf.float32, shape=[None, self.obs_dim])

        self.sess = tf.Session()

        self.policy = None

    def act(self, obs):
        self.policy(obs)

    def policy(self, state):
        NotImplementedError