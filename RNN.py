import lasagne
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#from IPython import display





#Generate Data




#Start of Network


BATCH_SIZE = 100


X = np.random.randint(0,10,size=BATCH_SIZE)
x_sym = T.matrix('X')


l_in = lasagne.layers.InputLayer(shape = 100)
l_rec1 = lasagne.layers.RecurrentLayer(l_in, num_units=50)
l_rec2 = lasagne.layers.RecurrentLayer(l_rec1, num_units=50)
l_rec3 = lasagne.layers.RecurrentLayer(l_rec2, num_units=50)

l_out = lasagne.layers.DenseLayer(l_rec3, num_units=5)

print lasagne.layers.get_output(l_out, inputs={l_in: X}).shape