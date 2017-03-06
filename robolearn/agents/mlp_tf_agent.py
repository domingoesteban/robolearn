from robolearn.agents.tf_agent import TFAgent
import tensorflow as tf


class MlpTFAgent(TFAgent):
    def __init__(self, act_dim, obs_dim, hidden_units):
        super(MlpTFAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim)

        # Required for mlp policy
        self.hidden_layers = self.multilayer_perceptron(hidden_units)

        # Output layer with linear activation
        self.action = tf.add(tf.matmul(self.hidden_layers['layer_'+str(len(hidden_units)-1)],
                             tf.Variable(tf.random_normal([hidden_units[-1], self.act_dim]), name='h_out')),
                             tf.Variable(tf.random_normal([self.act_dim]), name='b_out'))

        # Required for training
        # TODO

        # Initialize TF graph
        self.init = tf.initialize_all_variables()
        self.sess.run(self.init)

    def policy(self, state):
        # TODO: Check obs type and shape before
        return self.sess.run(self.action, feed_dict={self.state_holder: state})

    def train(self, history):
        pass

    def multilayer_perceptron(self, hidden_units):
        hidden_layers = {}
        for hh, n_units in enumerate(hidden_units):
            key = 'layer_' + str(hh)
            # Hidden layers with RELU activation
            if hh == 0:
                value = tf.nn.relu(tf.add(tf.matmul(self.state_holder,
                                                    tf.Variable(tf.random_normal([self.obs_dim, n_units]), name='h'+str(hh))),
                                          tf.Variable(tf.random_normal([n_units]), name='b'+str(hh))))
            else:
                value = tf.nn.relu(tf.add(tf.matmul(hidden_layers['layer_'+str(hh-1)],
                                                    tf.Variable(tf.random_normal([hidden_units[hh-1], n_units]), name='h'+str(hh))),
                                          tf.Variable(tf.random_normal([n_units]), name='b'+str(hh))))

            hidden_layers[key] = value

        return hidden_layers
