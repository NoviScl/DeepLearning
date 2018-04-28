import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#hyperparameters
lr = 0.002
training_iters = 100000
batch_size = 128


n_inputs = 28   #MNIST data input, one row (treat one row as a input)
n_steps = 28    #time steps: 28 rows
n_hidden_units = 128    #neurons in hidden layer
n_classes = 10  #0-9 digits

#tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

#define weights
weights = {
    #(28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    #(128,10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

bias = {
    #(128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    #(10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def RNN(X, weights, bias):
    #structure: fc-lstm-fc

    #hidden layer for input to cell
    # X: (128batch, 28steps, 28inputs)
    # ==> (128*28, 28inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in: (128batch*28steps, 128hidden)
    X_in = tf.matmul(X, weights['in']) + bias['in']
    #X_in : (128, 28, 128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    #cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    #lstm cell is divided into two parts (c_state ct, m_state at)->state_is_tuple
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    #time_major: whether in X_in : (128, 28, 128) time_step is the first parameter
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    #outputs contains all outputs for every time step, states is just the final state
    #outputs: [batch_size, max_time, cell.output_size]

    #hidden layer for output as the final results
    #this method only works if it's many-to-one
    #results = tf.matmul(states[1], weights['out']) + bias['out']

    #OR:
    #unpack to list steps*[(batch, outputs)]
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))    #output states (at) become the last output
    results = tf.matmul(outputs[-1], weights['out']) + bias['out']

    return results

pred = RNN(x, weights, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        batch_xs = sess.run(tf.reshape(batch_xs, [batch_size, n_steps, n_inputs]))
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
        if step%20 == 0:
            test_xs = mnist.test.images[:batch_size]
            test_ys = mnist.test.labels[:batch_size]
            test_xs = test_xs.reshape([batch_size, n_steps, n_inputs])
            print (sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
        step += 1