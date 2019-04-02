# %load main.py


#implementation of stochastic gradient descent learning algorithm for a feedforward neural network
#the gradients are calculated using backpropogation

import random
import numpy as np

class Network(object):

    def __init__(self, sizes):

        #list "sizes" contains number of neurons in respective layers of the network
        #the size of the list is the number of layers in the network wich each element specifying the number of neurons in the layer
        #for example the list [2, 3, 1] would be a three layer network with 2, 3, and 1 neurons in the first second and third layers
        self.num_layers = len(sizes)
        self.sizes = sizes
        #here the biases and weights are initialized randomly, using a Gaussian distribution with mean 0 and variance 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        #when a is input, the output of the network is returned
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def StochasticGradientDescent(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        #here the neural network is trained, using mini-batch stochastic gradient descent


        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        #the networks weights and biases are updated by applying gradient descent using backpropogation
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        #a tuple is returned representing the gradient for the cost function C_x.
      
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] #returns activations
        zs = [] # returns z vectors
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        #returns sum total of inputs that produce a correct output
        #the output is assumed to be the index of whichever neuron in the final layer has the highest activation

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        #return vector of partial derivatives
        return (output_activations-y)
def sigmoid(z):
#sigmoid function
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    #derivative of sigmoid function
    return sigmoid(z)*(1-sigmoid(z))
