from robolearn.old_agents.tf_agent import TFAgent
import tensorflow as tf
import numpy as np


class SplineTFAgent(TFAgent):
    def __init__(self, act_dim, obs_dim, init_act, n_via_points, total_points, final_act=None):
        super(SplineTFAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim)

        # Required for spline policy
        self.step = tf.Variable(0., name='step', trainable=False, dtype=tf.float32)
        self.increment_step = tf.cond(self.step < total_points,
                            lambda: tf.assign(self.step, self.step + 1.),
                            lambda: tf.assign(self.step, 1))

        via_points = np.random.random([act_dim, n_via_points])
        via_points[:, 0] = init_act
        if final_act is not None:
            via_points[:, -1] = final_act
        with tf.name_scope('via_points'):
            self.via_points = tf.Variable(via_points, dtype=tf.float32)
            self.variable_summaries(self.via_points)

        time_points = np.linspace(0, total_points, n_via_points)
        diff_points = np.insert(np.diff(time_points), 0, [0])
        with tf.name_scope('time_points'):
            self.time_points = tf.Variable(time_points, dtype=tf.float32)
            self.variable_summaries(self.time_points)

        self.coeffs = tf.Variable(self.get_coefficients(via_points, diff_points),
                                  name='coeffs', trainable=False, dtype=tf.float32)

        self.inf_lim = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)
        self.increment_inf_lim = tf.cond(self.step >= total_points,
                                         lambda: tf.assign(self.inf_lim, 0),
                                         lambda: tf.cond(self.step <= self.time_points[self.inf_lim+1],
                                                 lambda: tf.assign(self.inf_lim, self.inf_lim),
                                                 lambda: tf.assign(self.inf_lim, self.inf_lim + 1)))

        self.x_xi_power = tf.constant([0, 1, 2, 3], name='variable_power', dtype=tf.float32)
        self.x_xi = tf.cond(self.step < total_points,
                            lambda: self.step - self.time_points[self.inf_lim],
                            lambda: self.time_points[-1] - self.time_points[-2])
        self.action = tf.matmul(self.coeffs[:, self.inf_lim, :],
                                tf.reshape(tf.pow(self.x_xi, self.x_xi_power), [4, 1]))

        # Required to save/restore
        #self.saver = tf.train.Saver({save_dict})
        self.saver = tf.train.Saver()

        # Required for training
        self.train_step = 0
        #self.indexes = tf.range(0, tf.shape(self.action)[0]) * tf.shape(self.action)[1] + self.action_holder
        #self.responsible_actions = tf.gather(tf.reshape(self.action, [-1]), self.indexes)
        #self.loss = -tf.reduce_mean(tf.log(self.responsible_actions) * self.reward_holder)

        #self.train_vars = tf.trainable_variables()
        #self.gradient_holders = []
        #for idx, var in enumerate(self.train_vars):
        #    placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
        #    self.gradient_holders.append(placeholder)
        #self.gradients = tf.gradients(self.loss, self.train_vars)

        #learning_rate = 1e-2
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders, self.train_vars))

        # Merge all the summaries and write them out
        self.summary_merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('models' + '/train', self.sess.graph)
        #test_writer = tf.summary.FileWriter('models' + '/test')

        # Initialize TF graph
        self.init = tf.global_variables_initializer()
        #self.init = tf.variables_initializer([variable_name], name='init')
        self.sess.run(self.init)


    def policy(self, state):
        # TODO: Check obs type and shape before
        self.sess.run(self.increment_step)
        action = self.sess.run(self.action, feed_dict={self.state_holder: state})
        self.sess.run(self.increment_inf_lim)
        return action

    #def feed_dict(self, train):
    #    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    #    if train or FLAGS.fake_data:
    #        xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
    #        k = FLAGS.dropout
    #    else:
    #        xs, ys = mnist.test.images, mnist.test.labels
    #        k = 1.0
    #    return {x: xs, y_: ys, keep_prob: k}

    def train(self, history):
        self.train_step += 1
        # TODO: Run at the same time summary_merged and train_step
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        #summary, _ = self.sess.run([self.merged, self.train_step],
        #                           feed_dict=feed_dict(True),
        #                           options=run_options,
        #                           run_metadata=run_metadata)
        summary = self.sess.run(self.summary_merged)
        self.train_writer.add_summary(summary, self.train_step)


    def variable_summaries(self, var):
        """Attach summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def get_coefficients(self, via_points, time):
        if via_points.shape[1] != 2:
            assert ValueError("The column of via_points is not 2!")

        A = 0.  # Initial dy
        B = 0.  # Ending dy

        n = via_points.shape[1] - 1
        h = time[-n:]
        point_dim = via_points.shape[0]

        self.x = np.zeros(n+1)
        for i in xrange(n):
            self.x[i+1] = self.x[i] + h[i]

        h_matrix = np.zeros([n+1, n+1])
        h_matrix[0, 0] = 2. * h[0]
        h_matrix[0, 1] = h[0]
        h_matrix[n, n-1] = h[n-1]
        h_matrix[n, n] = 2.*h[n-1]

        for i in xrange(1, n):
            h_matrix[i, i-1] = h[i-1]
            h_matrix[i, i] = 2. * (h[i-1] + h[i])
            h_matrix[i, i+1] = h[i]

        coeffs = np.zeros([point_dim, n, 4])

        for ii in xrange(point_dim):
            y = via_points[ii, :]

            yh_vector = np.zeros(n+1)
            yh_vector[0] = 6. * ((y[1] - y[0]) / h[0] - A)
            yh_vector[n] = 6. * (B - (y[n] - y[n-1]) / h[n-1])
            for i in xrange(1, n):
                yh_vector[i] = 6. * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

            #m = np.linalg.inv(h_matrix).dot(yh_vector)
            m = np.linalg.solve(h_matrix, yh_vector)

            for i in xrange(n):
                coeffs[ii, i, 0] = y[i]  # a(i)
                coeffs[ii, i, 1] = (y[i+1] - y[i])/h[i] - h[i]*m[i]/2 - h[i]*(m[i+1] - m[i])/6.  # b(i)
                coeffs[ii, i, 2] = m[i]/2.  # c(i)
                coeffs[ii, i, 3] = (m[i+1] - m[i]) / (6. * h[i])  # d(i)

        return coeffs
