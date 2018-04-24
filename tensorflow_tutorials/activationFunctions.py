import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b=tf.matmul(inputs, Weights) + bias
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def relu(a):
    return tf.maximum(a, 0)

def sigmoid(a):
    return 1/(1+tf.exp(-a))

def tanh(a):
    return (tf.exp(a) - tf.exp(-a))/(tf.exp(a) + tf.exp(-a))


X=tf.random_uniform([10,5])
output=add_layer(X, 5, 2, tanh)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
print (sess.run(output))

