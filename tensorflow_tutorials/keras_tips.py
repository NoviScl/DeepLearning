from keras.callbacks import Tensorboard 
model.fit(X_train, y_train, validation_data=(X_val, y_val),
	epochs=10,
	callbacks=[Tensorboard("/tmp/tboard")])

# !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors !!!
from keras import backend as K
def reset_tf_session():
	K.clear_session()
	tf.reset_default_graph()
	s = K.get_session()
	return s


K.clear_session()
model = make_model()
model.summary()




