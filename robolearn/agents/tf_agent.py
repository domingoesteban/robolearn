
from robolearn.agents.agent import Agent
import tensorflow as tf
import os.path
import numpy as np


class TFAgent(Agent):

    def __init__(self, act_dim, obs_dim):
        super(TFAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim)

        tf.reset_default_graph()  # Clear the Tensorflow graph.

        # TF ops
        self.sess = tf.Session()
        # self.init = tf.initialize_all_variables()  # TODO Check why it is not working in subclasess

        # Required for policy
        self.state_holder = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name='state')

        # Required to save/restore
        self.saver = None #tf.train.Saver(save_dict)

        # Required for training
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None, self.act_dim], dtype=tf.int32)

    def policy(self, state):
        """
        Function that maps state to action
        :param state:
        :return:
        """
        NotImplementedError

    def train(self, history):
        raise NotImplementedError

    def save(self, file_path):
        if self.saver is None:
            assert AttributeError("self.saver has not been implemented!")

        save_path = self.saver.save(self.sess, file_path)
        print("Model saved in file: %s" % save_path)

    def restore(self, file_path):
        if self.saver is None:
            assert AttributeError("self.saver has not been implemented!")

        if os.path.exists(file_path):
            self.saver.restore(self.sess, file_path)
            print("Model restored.")
        else:
            assert ValueError("File %s does not exist!!!")
