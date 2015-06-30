import numpy as np
import csv


class Network():
	def __init__(self, sizes):
		# sizes = neurons per layer, with input and output
		# example:  [5,100,6] has 5 input, 100 hidden, 6 out

		self.num_layers = len(sizes)
		self.sizes = sizes

		#biases for hidden and output neurons
		self.biases = [np.random.randn(n,1) for n in sizes[1:]]

		#weights for hidden and output
		self.weights = [np.random.randn(y,x) 
						for x,y in zip(sizes[:-1], sizes[1:])]



	def sigmoid_prime(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def sigmoid(self, z):
		return 1.0/(1.0 + np.exp(-z))


	def feedforward(self, a):
		'''for input "a", propagate through entire network'''
		sig_vec = np.vectorize(self.sigmoid)
		for b, w in zip(self.biases, self.weights):
			a = sig_vec(np.dot(w, a)+b)
		return a


	'''Train mini batches'''
	def TMB(self, training_data, epochs, mini_batch_size, eta, 
			test_data = None):

		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			np.random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size]
							for k in np.arange(0,n,mini_batch_size)]

			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)

			if test_data:
				print("Epoch {0}: {1} / {2}".format(
						j, self.evaluate(test_data), n_test))
			else:
				print("Epoch {0} complete".format(j))


	def update_mini_batch(self, mini_batch, eta):
		del_b = [np.zeros(b.shape) for b in self.biases]
		del_w = [np.zeros(w.shape) for w in self.weights]

		for x,y in mini_batch:
			training_del_b, training_del_w = self.backprop(x,y)
			#add gradient terms to running sum
			del_b = [db+tdb for db,tdb in zip(del_b, training_del_b)]
			del_w = [dw+tdw for dw, tdw in zip(del_w, training_del_w)]

		#update weights and biases
		self.weights = [w - eta/len(mini_batch)*dw 
						for w, dw in zip(self.weights, del_w)]
		self.biases  = [b - eta/len(mini_batch)*db 
						for b, db in zip(self.biases, del_b)]



	def backprop(self, x, y):
		''' Backpropagation of training case x, label y 
		Returns a layer of gradients for weights and biases
		similar to self.weights	and self.biases

		'''
		training_del_b = [np.zeros(b.shape) for b in self.biases]
		training_del_w = [np.zeros(w.shape) for w in self.weights]
		sig_vec = np.vectorize(self.sigmoid)
		sig_prim_vec = np.vectorize(self.sigmoid_prime)
		activation = x
		#store activations from forward propagation
		activations = [x]
		# store chain:  activ = phi(z), dactive = phi_prime(z)
		z_vals = []

		'''Feed For`ward'''
		for w,b in zip(self.weights, self.biases):
			z = np.dot(w,activation)+b
			# print("z",z.shape)
			# print("act",activation.shape)
			# print("w",w.shape)
			# print("b",b.shape)

			# [30x21] * [21x1] = [30x1] + [30x1
			
			z_vals.append(z)
			activation = sig_vec(z)
			activations.append(activation)

		''' Begin back prop with last layer'''
		chain_error = self.cost_derivative(activations[-1], y) * sig_prim_vec(z_vals[-1])
		training_del_b[-1] = chain_error
		#chain:  act[-2]*act[-1]*[1-act[-1]*cost_deriv)
		# matrix op, not dot
		training_del_w[-1] = np.dot(chain_error, activations[-2].transpose())

		'''Continue back prop, using negative index'''
		for l in range(2, self.num_layers):
			z = z_vals[-l]

			chain_error = np.dot(self.weights[-l+1].transpose(), chain_error) * sig_prim_vec(z)
			training_del_b[-l] = chain_error
			training_del_w[-l] = np.dot(chain_error, activations[-l-1].transpose())

		return (training_del_b, training_del_w)



	def evaluate(self, test_data):
		'''feed forward test data'''
		pairs = [ (np.argmax(self.feedforward(x)), np.argmax(y))
					for x,y, in test_data]
		return sum([int(predict==label) for predict, label in pairs])



	def cost_derivative(self, predict, actual):
		'''Assuming cost = 0.5* ||predict - actual||^2'''
		return predict - actual 


def encode_label(y):
	encoded = np.zeros((10,1))
	encoded[int(y)]=1.0
	return encoded

def process_data(x):
	intv = np.vectorize(int)
	return intv(x).reshape(784,1)/255

if __name__ == "__main__":
	print("hallo")

	network_shape = [784, 20, 10]

	fr = csv.reader(open('mnist/train.csv', 'r'))
	header = next(fr)


	train = [(process_data(row[1:]), encode_label(row[0])) for row in fr]


	subset = train[0:20000]
	test = train[26000:36000]

	net = Network(network_shape)
	net.TMB(subset, 10 , 10, 2, test_data = test)



