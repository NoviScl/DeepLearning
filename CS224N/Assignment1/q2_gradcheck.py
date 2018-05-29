import numpy as np 
import random

def gradcheck_naive(f, x):
	"""
	f: a function that takes a single argument and
		outputs the cost and its gradients
	x: the point(numpy array) to check the gradient at
	"""
	rndstate = random.getstate()
	random.setstate(rndstate)
	fx, grad = f(x) #evaluate function value at original point
	h = 1e-4 

	#Iterate over all indexes in x
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		#alter one value at a time and calculate gradient
		ix = it.multi_index
		x[ix] += h

		random.setstate(rndstate)
		new_f1 = f(x)[0]

		x[ix] -= 2*h 

		random.setstate(rndstate)
		new_f2 = f(x)[0]

		x[ix] += h 

		numgrad = (new_f1 - new_f2)/(2*h)

		#compare gradients
		reldiff = abs(numgrad - grad[ix])/max(1, abs(numgrad), abs(grad[ix]))
		if reldiff > 1e-5:
			print ("Gradient check failed")
			print ("First gradient error found at index %s"%str(ix))
			print ("Correct gradient: %f \t Numerical gradient: %f"%(grad[ix], numgrad))
			return

		it.iternext()
	print ("Gradient check passed!")


def sanity_check():
	quad = lambda x: (np.sum(x**2), x*2)

	print ("Running sanity check...")
	gradcheck_naive(quad, np.array(123.456))
	gradcheck_naive(quad, np.random.randn(3,))
	gradcheck_naive(quad, np.random.randn(4,5))

def your_sanity_check():
	from q2_sigmoid import sigmoid, sigmoid_grad

	sig = lambda x: (sigmoid(x), sigmoid_grad(sigmoid(x)))
	print ("Running your sanity checks...")

	gradcheck_naive(sig, np.array(123.456))

if __name__=="__main__":
	sanity_check()
	your_sanity_check()





