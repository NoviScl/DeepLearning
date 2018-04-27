import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram('weights', Weights)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram('bias', bias)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs, Weights) + bias
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram('outputs', outputs)
        return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

xs = tf.placeholder(tf.float32, [None, 784])    #28*28
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), axis=1))
tf.summary.scalar('xentropy loss', cross_entropy)

#default lr for Adam is 0.001, this is smaller tham that of GD
train_step = tf.train.AdamOptimizer(0.006).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter('/Users/sichenglei/Desktop/I2R/tensorGraph/logs', sess.graph)
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i%50==0:
        print (compute_accuracy(mnist.test.images, mnist.test.labels))




