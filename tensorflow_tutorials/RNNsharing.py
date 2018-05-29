import tensorflow as tf 

class TrainConfig:
	batch_size = 20
	time_steps = 20 
	input_size = 10
	output_size = 2 
	cell_size = 11
	learning_rate = 0.01

class TestConfig(TrainConfig):
	time_steps = 1

class RNN(object):
	def __init__(self, config):
		self._batch_size = config.batch_size 
		self._time_steps = config.time_steps
		self._input_size = config.input_size
		self._output_size = config.output_size  
		self._cell_size = config.cell_size 
		self._lr = config.learning_rate 
		self._build_RNN()

	def _build_RNN(self):
		with tf.variable_scope('inputs'):
			self._xs = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._input_size], name='xs')
			self._ys = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._output_size], name='ys')
		with tf.name_scope('RNN'):
			with tf.variable_scope('input_layer'):
				l_in_x = tf.reshape(self._xs, [-1, self._input_size], name='2_2D')	#(batch*n_step, in_size)
				#Ws: (in_size, cell_size)
				#this will be reused
				Wi = self._weight_variable([self._input_size, self._cell_size])
				print (Wi.name)
				#bs (cell_size,)
				bi = self._bias_variable([self._cell_size])

				#l_in_y: (batch*n_steps, cell_size)
				with tf.name_scope('Wx_plus_b'):
					l_in_y = tf.matmul(l_in_x, Wi) + bi 
				l_in_y = tf.reshape(l_in_y, [-1, self._time_steps, self._cell_size], name='2_3D')

			with tf.variable_scope('cell'):
				cell = tf.contrib.rnn.BasicLSTMCell(self._cell_size)
				with tf.name_scope('initial_state'):
					self._cell_initial_state = cell.zero_state(self._batch_size, dtype=tf.float32)
				
				self.cell_outputs=[]
				cell_state = self._cell_initial_state
				for t in range(self._time_steps):
					if t>0: 
						tf.get_variable_scope().reuse_variables()
					#cell return shape: [batch_size, self.output_size]
					cell_output, cell_state = cell(l_in_y[:, t, :], cell_state)
					self.cell_outputs.append(cell_output)
				self._cell_final_state = cell_state

			with tf.variable_scope('output_layer'):
				#cell_outputs shape: (time_steps, batch_size, cell_size)
				# cell_outputs_reshaped (batch_size*time_step, cell_size)
				cell_outputs_reshaped = tf.reshape(tf.transpose(self.cell_outputs, [1,0,2]), [-1, self._cell_size])
				Wo = self._weight_variable((self._cell_size, self._output_size))
				bo = self._bias_variable((self._output_size))
				product = tf.matmul(cell_outputs_reshaped, Wo) + bo
				#_pred shape: (batch_size*time*step, output_size)
				self._pred = tf.nn.relu(product)

		with tf.name_scope('cost'):
			_pred = tf.reshape(_pred, [self._batch_size, self._time_steps, self._output_size])
			mse = self.ms_error(_pred, self._ys)
			mse_ave_across_batch = tf.reduce_mean(mse, 0)	#(time_steps, output_size)
			mse_sum_acorss_time = tf.reduc_sum(mse_ave_across_batch, 0)	#(output_size)
			self._cost = mse_sum_acorss_time
			self._cost_ave_time = self._cost/self._time_steps 

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self._cost)

	@staticmethod
	def ms_error(y_target, y_pre):
		return tf.square(y_target - y_pre)

	@staticmethod
	def _weight_variable(shape, name='weights'):
		initializer = tf.random_normal_initializer(mean=0, stddev=0.5)
		return tf.get_variable(shape=shape, initializer=initializer, name=name)

	@staticmethod
	def _bias_variable(shape, name='bias'):
		initializer = tf.constant_initializer(0.1)
		return tf.get_variabel(name=name, shape=shape, initializer=initializer)

if __name__=='__main__':
	train_config = TrainConfig()
	test_config = TestConfig()

	with tf.variable_scope('rnn') as scope:
		sess = tf.Session()
		train_rnn2 = RNN(train_config)
		scope.reuse_variables()
		#this allows us to reuse the same parameters
		test_rnn2 = RNN(test_config)
		init = tf.global_variables_initializer()
		sess.run(init)







