import numpy as np

# def softmax(x):
# 	orig_shape = x.shape
# 	if len(x.shape)>1:
# 		#Matrix
# 		exp_minmax = lambda x: np.exp(x - np.max(x))
# 		denom = lambda x: 1.0/np.sum(x)
# 		#apply to each row
# 		#for each row, minus away the max them exp, to save computation
# 		x = np.apply_along_axis(exp_minmax, 1, x)
# 		denominator = np.apply_along_axis(denom, 1, x)
# 		#denominator will become a 1D array
# 		if len(denominator.shape) == 1:
# 			denominator = denominator.reshape((denominator.shape[0], 1))
# 		#note that matrix cannot multiply with 1D array
# 		#because of broadcasting, so must convert 1D array to 2D
# 		x = x*denominator
# 	else:
# 		#vector
# 		x_max = np.max(x)
# 		x = x - x_max
# 		numerator = np.exp(x)	#a vector
# 		denominator = 1.0/np.sum(numerator)	#a number
# 		x = numerator.dot(denominator)
# 		#x = numerator*denominator

# 	assert x.shape == orig_shape
# 	return x

def softmax(x):
	orig_shape = x.shape 
	if len(x.shape) > 1:
		maxi = np.max(x, axis=1)
		x = np.exp(x - maxi[:, np.newaxis])	#change to 2D
		x = x/np.sum(x, keepdims=True, axis=1)
	else:
		maxi = np.max(x, axis=0)
		x = np.exp(x - maxi)
		x = x/np.sum(x, axis=0)

	assert x.shape == orig_shape
	return x



def test_softmax_basic():
	print ("running basic tests")
	test1 = softmax(np.array([1,2]))
	print (test1)
	ans1 = np.array([0.26894142, 0.73105858])
	assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

	test2 = softmax(np.array([[1001, 1002], [3,4]]))
	print (test2)
	ans2 = np.array([[0.26894142, 0.73105858],[0.26894142, 0.73105858]])
	assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

	test3 = softmax(np.array([[-1001, -1002]]))
	print (test3)
	ans3 = np.array([[0.73105858, 0.26894142]])
	assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

def test_softmax():
	print ('Running your tests')
	#raise NotImplementedError

if __name__=="__main__":
	test_softmax_basic()
	test_softmax()



