import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram('weights', Weights)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram('bias', bias)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs, Weights) + bias
        #dropout
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        #tf.summary.histogram(layer_name+'/outputs', outputs)
        return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#same padding: 
#out_height = ceil(float(in_height) / float(strides[1]))
#out_width = ceil(float(in_width) / float(strides[2]))

#valid padding:
#out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
#out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

def conv2d(x, W):
    #stride [1, height_movement, width_movement, 1] 'NHWC'
    #must have strides[0]=strides[3]=1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, height_movement, width_movement, 1] 'NHWC'
    # must have strides[0]=strides[3]=1
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')




keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(xs, [-1,28,28,1])  #NHWC, C=1 because it is greyscale


#add layer
#conv1 layer
#filter shape in conv2d: [filter_height, filter_width, in_channels, out_channels]
W_conv1 = weight_variable([5,5,1,32])       #patch 5*5, in_channel=1, out_channel=32
b_conv1=bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    #output_size: 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                             #output_size: 14*14*32

#conv2 layer
W_conv2 = weight_variable([5,5,32,64])       #patch 5*5, in_channel=32, out_channel=32
b_conv2=bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    #output_size: 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                             #output_size: 7*7*64

#fully_connected_layer1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
#[n_samples, 7, 7, 64] -> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

#fully_connected_layer2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), axis=1))
tf.summary.scalar('xentropy loss', cross_entropy)

#default lr for Adam is 0.001, this is smaller tham that of GD
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter('/Users/sichenglei/Desktop/I2R/tensorGraph/logs/train', sess.graph)
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys:batch_ys, keep_prob: 0.5})
    if i%50==0:
        print (compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))




