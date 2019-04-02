# %load data_loader.py

#this library loads the MNIST image data and 


import pickle
import gzip

import numpy as np

def load_data():
    #MNIST data is returned as a tuple which contains training data, validation data and test data
    #training data is returned as a tuple with two entries.
    #First entry containing an ndarray with 50,000 entries of each of the training images
    #each entry is an ndarray with 784 values which represent the 28*28 pixels in each image
    
    #the second entry in training_data is a ndarray containing 50000 entries.
    #which are the digit values 0-9 for the corresponding images contained in the first entry of the tuple
    
    #validation_data and test_data contain only 10,000 images
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    #Return a tuple containing (training_data, validation_data,test_data)
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):

    #returns a 10-dimensional unit vector with 1.0 in the jth position and zeroes elsewhere
    #converts a 0-9 digit into a corresponding desired output from the neural network
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
