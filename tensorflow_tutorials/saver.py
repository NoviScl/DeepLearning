import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

#Save to file
#remember to define the same dtype and shape when restore

# W = tf.Variable([[1,2,3], [3,4,5]], dtype=tf.float32, name='weights')
# v = tf.Variable([[1,2,3]], dtype=tf.float32, name='bias')
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, '/Users/sichenglei/Desktop/I2R/tensorGraph/my_net/save_net.ckpt')
#     print ('Save to path: ', save_path)



#restore variables
#redefine the same shape and same dtype for the variables
#the variables must have same name, shape and dtype as in the saved model

# W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='weights')
# b= tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='bias')
W = tf.Variable(np.zeros((2,3)), dtype=tf.float32, name='weights')
b = tf.Variable(np.zeros((1,3)), dtype=tf.float32, name='bias')


#do not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, '/Users/sichenglei/Desktop/I2R/tensorGraph/my_net/save_net.ckpt')
    print ('weights: ', sess.run(W))
    print ('bias: ', sess.run(b))
