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
			self._ys = tf.placehodler(tf.float32, [self._batch_size, self._time_steps, self._output_size], name='ys')
		with tf.name_scope('RNN'):
			l_in_x = tf.reshape(self._xs, [-1, self._input_size], name='2_2D')
			#Ws: (in_size, cell_size)
			Wi = self._weight_variable([self._input_size, self._cell_size])
			print (Wi.name)
			#bs: (cell_size, )
			bi = self._bias_variable([self._cell_size,])
			#l_in_y: (batch_size*time_steps, cell_size)
			with tf.name_scope('Wx_plus_b'):
				l_in_y = tf.matmul(l_in_x, Wi) + bi 
			l_in_y = tf.reshape(l_in_y, [-1, self._time_steps, self._cell_size])

		with tf.variable_scope('cell'):
			cell = tf.contrib.rnn.BasicLSTMCell(self._cell_size)
			with tf.name_scope('initial_state'):
				self._cell_initial_state = cell.zero_state(self._batch_size, dtype=tf.float32)

			self.cell_outputs = []
			cell_state = self._cell_initial_state
			for t in range(self._time_steps):
				if t>0:
					tf.get_variable_scope().reuse_variables()	#reuse Wi and bi
				cell_output, cell_state = cell(l_in_y[:, t, :], cell_state)
				self.cell_outputs.append(cell_output)	#cell_outputsL (time_steps, batch_size, output_size)
			self._cell_final_state = cell_state 


		with tf.variable_scope('output_layer'):
			#cell_outputs_reshaped (batch_size*time_steps, cell_size)
			cell_outputs_reshaped = tf.reshape(tf.transpose(self.cell_outputs, [1,0,2]), [-1, self._cell_size])
















