import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np


state=tf.Variable(0, name='counter')
# print (state.name)

one=tf.constant(1)
new_value=tf.add(state, one)
update=tf.assign(state, new_value)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #every time increase by 1
    #without assign, new_value will always be 1
    for _ in range(3):
        sess.run(update)
        print (sess.run(state))