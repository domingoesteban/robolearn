from robolearn.agents.tf_agent import TFAgent
import tensorflow as tf
import numpy as np


class LinearTFAgent(TFAgent):
    def __init__(self, act_dim, obs_dim):
        super(LinearTFAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim)

        # Required for linear policy
        self.weights = tf.Variable(tf.truncated_normal([self.obs_dim, self.act_dim]), name="weights")
        self.biases = tf.Variable(tf.zeros([self.act_dim]), name="biases")
        self.action = tf.add(tf.matmul(self.state_holder, self.weights), self.biases)


        # Required for training
        self.indexes = tf.range(0, tf.shape(self.action)[0]) * tf.shape(self.action)[1] + self.action_holder
        self.responsible_actions = tf.gather(tf.reshape(self.action, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_actions) * self.reward_holder)

        self.train_vars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(self.train_vars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        self.gradients = tf.gradients(self.loss, self.train_vars)

        learning_rate = 1e-2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders, self.train_vars))

        # Initialize TF graph
        self.init = tf.initialize_all_variables()
        self.sess.run(self.init)

    def policy(self, state):
        # TODO: Check obs type and shape before
        return self.sess.run(self.action, feed_dict={self.state_holder: state})

    def train(self, history):
        pass


