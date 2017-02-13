from robolearn.agents.tf_agent import TFAgent
import tensorflow as tf

class LinearTFAgent(TFAgent):
    def __init__(self, act_dim, obs_dim):
        super(LinearTFAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim)

        self.weights = tf.Variable(tf.truncated_normal([self.obs_dim, self.act_dim]), name="weights")
        self.biases = tf.Variable(tf.zeros([self.act_dim]), name="biases")

        self.policy = tf.add(tf.matmul(self.state, self.weights), self.biases)

        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)


    def act(self, obs):
        # TODO: Check obs type and shape before
        return self.sess.run(self.policy, feed_dict={self.state: obs})
