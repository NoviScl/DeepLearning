def skipgram():
	batch_inputs = tf.placeholder(tf.int32, shape=[batch_size, ])
	batch_labels = tf.placehodler(tf.int32, shape=[batch_size, 1])
	val_dataset = tf.constant(val_data, dtype=tf.int32)

	with tf.variable_scope("word2vec") as scope:
		embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		batch_embeddings = tf.nn.embedding_lookup(embeddings, batch_inputs)

		weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
		biases = tf.Variable(tf.zeros([vocabulary]))

		loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights, biases=biases, labels=batch_labels, 
			inputs=batch_inputs, num_sampled=num_sampled, num_class=vocabulary_size))

		norm = tf.srqt(tf.reduce_mean(tf.square(embeddings), 1, keepdims=True))
		normalized_embeddings = embeddings/norm

		val_embeddings = tf.nn.embedding_lookup(normalized_embeddings, val_dataset)
		similarity = tf.matmul(val_embeddings, normalized_embeddings, transpose_b=True)
		return batch_inputs, batch_labels, normalized_embeddings, loss, similarity

def run():
	batch_inputs, batch_labels, normalized_embeddings, loss, similarity = skipgram()
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
	init = tf.gloabl_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		average_loss = 0.0
		for step, batch_data in enumerate(train_data):
			inputs, labels = batch_data 
			feed_dict = {batch_inputs: inputs, batch_labels: labels}

			_, loss_val = sess.run([optimizer, loss], feed_dict)
			average_loss += loss_val 

			if step%1000 == 0:
				if step > 0:
					average_loss/=1000
				print('loss at iter', step, ':', average_loss)
				average_loss = 0

			if step%5000 == 0:
				sim = similarity.eval()
				for i in xrange(len(val_data)):
					top_k = 8
					nearest = (sim[i:].argsort()[1:top_k+1])
					print_close_words(val_data[i], nearest, reverse_dictionary)
		final_emneddings = normalized_embeddings.eval()

