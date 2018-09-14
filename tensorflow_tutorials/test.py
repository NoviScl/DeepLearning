import tensorflow as tf 

x = tf.Variable(tf.random_normal([3,2,4]))
x_us = tf.unstack(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
s, s_us = sess.run([x, x_us])
print (s[-1], type(s[-1]))
print (s_us[-1], type(s_us[-1]))