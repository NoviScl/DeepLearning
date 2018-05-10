import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=False)


#Visualize decoder setting
#parameters
learning_rate = 0.03
training_epochs = 25
batch_size = 256
display_step = 1
examples_to_show = 10

#network parameters
n_input = 784	#MNIST data input: 28*28

#tf Graph input (only pictures)
X = tf.placeholder('float', [None, n_input])

#hidden layer settings
n_hidden_1 = 256	#1st layer num features
n_hidden_2 = 128	#2nd layer num features
weights = {
	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

bias = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}


#build the encoder:
def encoder(x):
	#Encoder hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
		bias['encoder_b1']))
	#Encoder hidder layer with sigmoid activation #2
	layer_2 = tf.nn.softmax(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
		bias['encoder_b2']))
	return layer_2

#build the decoder:
def decoder(x):
	#should use the same activation for encoder and decoder
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
		bias['decoder_b1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
		bias['decoder_b2']))
	return layer_2


#construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

#prediction
y_pred = decoder_op
#Targets (labels) are the input data
y_true = X

#define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred),1))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)
	for epoch in range(training_epochs):
		#loop over all batches
		for i in range(total_batch):
			#range of pixel value(x) in mnist: 0-1
			#so when we use sigmoid at the last layer of decoder, dun need extra normalization
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
		if epoch%display_step == 0:
			print ('Epoch: ', '%04d'%(epoch+1),
				'cost=', '{:.9f}'.format(c))
	print ('Optimization Finished!')

	#Apply encoder and decoder over test set
	encode_decode = sess.run(
		y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
	#compare original images with their reconstructions
	f, a = plt.subplots(2, 10)	#2 rows, 10 columns
	for i in range(examples_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
		a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
	plt.show()


