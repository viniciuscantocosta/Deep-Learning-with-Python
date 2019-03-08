import math
import numpy as np

# np.set_printoptions(threshold=np.nan)

# X = (hours in the gym, average sleep hours per day, average calories per day), 
# y = muscle gain factor
X = np.array([[3, 8, 2000],
			  [5, 9, 2500],
			  [0, 6, 2500],
			  [1, 7, 1800],
			  [15, 7, 2500]], dtype=float)

y = np.array([[1.1], [1.3], [0.9], [1.0], [0.8]], dtype=float)

xPredicted = np.array([5, 8, 3000], dtype=float)
xPredicted = xPredicted/np.amax(X, axis=0)


# scale units
X = X/np.amax(X, axis=0)  # maximum of X array by column
y = y/np.amax(y)

print(X)
print(y)

class Neural_Network(object):
	def __init__(self):
		# parameters
		self.inputSize = 3
		self.outputSize = 1
		self.hiddenSize1 = 4
		self.hiddenSize2 = 4
		# learning rate
		self.alpha = 1
		# weights
		self.W1 = np.random.randn(self.inputSize, self.hiddenSize1) # (3x4) weight matrix from input to hidden layer 1
		self.W2 = np.random.randn(self.hiddenSize1, self.hiddenSize2) # (4x4) weight matrix from hidden layer 1 to hidden layer 2
		self.W3 = np.random.randn(self.hiddenSize2, self.outputSize) # (4x1) weight matrix from hidden layer 2 to output
		# parameter for leaky ReLU
		self.alphaReLU = 0.05

	def sigmoid(self, s):
		# activation function 0
		return 1/(1+np.exp(-s))

	def sigmoidPrime(self, s):
		return s*(1-s)

	def tanhPrime(self, s):
		# activation function 1, the tanh function used was the build-in from numpy
		return 1 - s**2

	def reLU(self, s):
		# activation function 2
		return np.where(s > 0, s, 0)

	def reLUPrime(self, s):
		return np.where(s > 0, 1, 0)

	def leakyReLU(self, s):
		# activation function 3
		return np.where(s > 0, s, s * self.alphaReLU)

	def leakyReLUPrime(self, s):
		return np.where(s > 0, 1, self.alphaReLU)

	def forward(self, X):
		# forward propagation through our network
		self.z1_0 = np.dot(X, self.W1) # dot product of X (input) and first set of (3x4) weights 
		self.z1_1 = self.leakyReLU(self.z1_0) # activation

		self.z2_0 = np.dot(self.z1_1, self.W2) # 1st hidden layer
		self.z2_1 = self.leakyReLU(self.z2_0) # activation

		self.z3_0 = np.dot(self.z2_1, self.W3) # 2nd hidden layer
		o = np.tanh(self.z3_0) #final activation function

		return o

	def backward(self, X, y, o):
		# backpropagate through the network
		self.o_error = y - o
		self.o_delta = self.o_error*self.tanhPrime(o) # this computes how far out our z3_0 went from the ideal z3_0 

		self.z2_1_error = self.o_delta.dot(self.W3.T) # z2 error: how much our hidden layer weights contributed to output error
		self.z2_1_delta = self.z2_1_error*self.leakyReLUPrime(self.z2_1)

		self.z1_1_error = self.z2_1_delta.dot(self.W2.T)
		self.z1_1_delta = self.z1_1_error*self.leakyReLUPrime(self.z1_1)

		self.W1 += self.alpha*X.T.dot(self.z1_1_delta)
		self.W2 += self.alpha*self.z1_1.T.dot(self.z2_1_delta)
		self.W3 += self.alpha*self.z2_1.T.dot(self.o_delta)

	def train(self, X, y):
		o = self.forward(X)
		self.backward(X, y, o)

	def saveWeights(self):
		np.savetxt("w1.txt", self.W1, fmt="%s")
		np.savetxt("w2.txt", self.W2, fmt="%s")
		np.savetxt("w3.txt", self.W3, fmt="%s")

	def predict(self):
		print("Predicted data based on trained weights: ")
		print("Input (scaled): \n" + str(xPredicted))
		print("Output: \n" + str(self.forward(xPredicted) * 1.3)) # * np.max(y)

NN = Neural_Network()
for i in range(10000): #train the NN 1000 times
	print("# " + str(i) + "\n")
	print("Input (scaled): \n" + str(X))
	print("Actual Output: \n" + str(y))
	print("Predicted Output: \n" + str(NN.forward(X)))
	print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))))
	print("\n")
	NN.train(X,y)

NN.saveWeights()
NN.predict()


# o = NN.forward(X)

# print("Predicted Output: \n" + str(o))
# print("Actual Output: \n" + str(y))