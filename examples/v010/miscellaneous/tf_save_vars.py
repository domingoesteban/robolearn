import tensorflow as tf

counter = tf.Variable(tf.zeros([1]), name="counter")
variable_to_save = tf.Variable(tf.random_normal([2, 3]), name="variable_to_save")
increment_counter = tf.assign(counter, counter+1)

#global_init = tf.global_variables_initializer()
all_init = tf.global_variables_initializer()
some_init = tf.variables_initializer([counter], name='some_init')

# Add ops to save and restore all the variables.
#saver = tf.train.Saver()  # Save all variables
saver = tf.train.Saver({'variable_to_save': variable_to_save})


with tf.Session() as sess:
    sess.run(all_init)
    #sess.run(some_init)

    print(sess.run(variable_to_save))
    print(sess.run(counter))
    sess.run(increment_counter)

    # Restore variables from disk.
    saver.restore(sess, "models/tf-test-model.ckpt")
    print("Model restored.")

    print(sess.run(variable_to_save))
    print(sess.run(counter))

    save_path = saver.save(sess, "models/tf-test-model.ckpt")
    print("Model saved in file: %s" % save_path)


