import numpy as np 
import random 

from q1_softmax import softmax 
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive 


def forward_backward_prop(data, labels, params, dimensions):
	"""
	Forward and backward propagation for a two-layer sigmoidal network

	Compute the forward propagaion and for the cross entropy cost,
	and backward propagation for the gradients for all parameters

	Arguments:
	data -- M x Dx matrix, where each row is a training example
	labels -- M x Dy matrix, where each row is a one-hot vector 
	params -- Model parameters, these are unpacked already 
	dimensions -- A tuple of input dimension, #hidden units and output dimension
	"""

	#unpack network parameters
	ofs = 0 
	Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

	W1 = np.reshape(params[ofs: ofs+Dx*H], (Dx, H))
	ofs += Dx*H 
	b1 = np.reshape(params[ofs: ofs+H], (1, H))
	ofs += H 
	W2 = np.reshape(params[ofs: ofs+H*Dy], (H, Dy))
	ofs += H*Dy 
	b2 = np.reshape(params[ofs: ofs+Dy], (1, Dy))

	#forward
	h = sigmoid(np.dot(data, W1) + b1)	#(N, H)
	yhat = softmax(np.dot(h, W2) + b2)	#(N, Dy)

	#backward
	cost = -np.sum(labels * np.log(yhat))

	d1 = yhat - labels #dz2: (N, Dy)

	gradW2 = np.dot(h.T, d1)	#(H, Dy)
	gradb2 = np.sum(d1, axis=0)	#(1, Dy)

	d2 = np.dot(d1, W2.T)	#dh: (N, H)
	d3 = d2 * sigmoid_grad(h)	#dz1: (N, H)

	gradW1 = np.dot(data.T, d3) #(Dx, H)
	gradb1 = np.sum(d3, axis=0)	#(1, H)

	#stack gradients
	grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
		gradW2.flatten(), gradb2.flatten()))

	return cost, grad 


def sanity_check():
	print ("Running sanity check...")

	N = 20
	dimensions = [10, 5, 10]
	data = np.random.randn(N, dimensions[0])	#each row will be a datum 
	labels = np.zeros((N, dimensions[2]))
	for i in range(N):
		labels[i, random.randint(0, dimensions[2]-1)] = 1

	params = np.random.randn((dimensions[0]+1)*dimensions[1]+(dimensions[1]+1)*dimensions[2])

	gradcheck_naive(lambda params:
		forward_backward_prop(data, labels, params, dimensions), params)

if __name__=="__main__":
	sanity_check()









