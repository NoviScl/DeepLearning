import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


#load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


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

keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

#add layer
layer1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(layer1, 50, 10, 'l2', activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), axis=1))
tf.summary.scalar('xentropy loss', cross_entropy)

#default lr for Adam is 0.001, this is smaller tham that of GD
train_step = tf.train.AdamOptimizer(0.006).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
merge = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/Users/sichenglei/Desktop/I2R/tensorGraph/logs/train', sess.graph)
test_writer = tf.summary.FileWriter('/Users/sichenglei/Desktop/I2R/tensorGraph/logs/test', sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    if i%50==0:
        #record training error and test error, don't add dropout here
        train_result = sess.run(merge, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merge, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})

        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)











