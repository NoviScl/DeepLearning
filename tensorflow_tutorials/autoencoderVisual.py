import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets('/tmp/data', one_hot=False)


#parameters
learning_rate = 0.01
training_epoches = 10
batch_size = 256
display_step = 1
examples_to_show = 10

n_input = 784	#MNIST data input (28*28)

X = tf.placeholder('float', [None, n_input])

#hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64 
n_hidden_3 = 10
n_hidden_4 = 2


weights = {
	'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
	'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
	'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),

	'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3])),
	'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2])),
	'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1])),
	'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input]))
}

bias = {
	'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
	'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
	'encoder_b4': tf.Variable(tf.truncated_normal([n_hidden_4])),

	'decoder_b1': tf.Variable(tf.truncated_normal([n_hidden_3])),
	'decoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
	'decoder_b3': tf.Variable(tf.truncated_normal([n_hidden_1])),
	'decoder_b4': tf.Variable(tf.truncated_normal([n_input]))
}


def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['encoder_h1'])+bias['encoder_b1'])
	layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['encoder_h2']+bias['encoder_b2']))
	layer_3 = tf.nn.sigmoid(tf.matmul(layer_2, weights['encoder_h3']+bias['encoder_b3']))
	layer_4 = tf.nn.sigmoid(tf.matmul(layer_3, weights['encoder_h4']+bias['encoder_b4']))
	return layer_4


def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['decoder_h1'])+bias['decoder_b1'])
	layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['decoder_h2'])+bias['decoder_b2'])
	layer_3 = tf.nn.sigmoid(tf.matmul(layer_2, weights['decoder_h3'])+bias['decoder_b3'])
	layer_4 = tf.nn.sigmoid(tf.matmul(layer_3, weights['decoder_h4'])+bias['decoder_b4'])
	return layer_4

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred, y_true), 1))
optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)
	for epoch in range(training_epoches):
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)	#min(x)=0, max(x)=1
			_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
		if epoch % display_step == 0:
			print ('Epoch: ', '%04d'%(epoch+1),
				'cost=', '{:.9f}'.format(c))
	print ('Optimization finish')

	encode_decode = sess.run(
		y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
	f, a = plt.subplots(2, 10)
	for i in range(examples_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
		a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
	plt.show()

	encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
	plt.scatter(encoder_result[:,0], encoder_result[:,1], c=mnist.test.labels)	#c: map labels to different colors
	plt.colorbar()
	plt.show()





