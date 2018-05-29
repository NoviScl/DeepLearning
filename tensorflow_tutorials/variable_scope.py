import tensorflow as tf 
tf.set_random_seed(1)

# with tf.name_scope("a_name_scope"):
# 	initializer = tf.constant_initializer(value=1)
# 	#get_variable does not have the name scope
# 	#but it has variable_scope
# 	var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
# 	var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
# 	var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
# 	var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)

with tf.variable_scope('a_variable_scope') as scope:
	initializer = tf.constant_initializer(value=3)
	var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
	#if want to reuse the same variable, must specify and use same name
	scope.reuse_variables()
	var3_reuse = tf.get_variable(name='var3')
	#if use tf.Variable with same name, it will still create a new variable
	var4_reuse = tf.get_variable(name='var3')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print (var3.name)
	print (sess.run(var3))
	print (var4_reuse.name)
	print (sess.run(var4_reuse))
	

#one case to use shared variables:
#RNN use same parameters



