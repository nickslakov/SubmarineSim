import lasagne
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#from IPython import display





#Generate Data




#Start of Network


BATCH_SIZE = 1
SEQUENCE_LENGTH = 1
NUM_INPUTS = 12
NUM_OUTPUTS = 2


X = np.random.randint(0,10,size=(BATCH_SIZE,SEQUENCE_LENGTH, NUM_INPUTS)) # Test inputs
Y = np.random.randint(0,10,size=NUM_OUTPUTS) # Test outputs
x_sym = T.tensor3() # Symbolic inputs
y_sym = T.vector() # Symbolic target outputs

l_in = lasagne.layers.InputLayer(shape = (BATCH_SIZE, SEQUENCE_LENGTH, NUM_INPUTS))
l_hid1 = lasagne.layers.RecurrentLayer(l_in, num_units=25)
l_hid2 = lasagne.layers.RecurrentLayer(l_hid1, num_units=50)
l_hid3 = lasagne.layers.RecurrentLayer(l_hid2, num_units=25)
l_out = lasagne.layers.DenseLayer(l_hid3, num_units=NUM_OUTPUTS)

train_model = lasagne.layers.get_output(l_out, x_sym, deterministic = False)
eval_model = lasagne.layers.get_output(l_out, x_sym, deterministic = True)

#Print shape of output
print train_model.eval({x_sym:X}).shape

# Get outputs from network
train_out = train_model.eval({x_sym:X})
eval_out = eval_model.eval({x_sym:X})

# All trainable params in the network
all_params = lasagne.layers.get_all_params(l_out, trainable=True)

# Define the Cost Function
cost_train = lasagne.objectives.squared_error(train_out, y_sym).mean()
cost_eval = lasagne.objectives.squared_error(eval_out, y_sym).mean()

# Use Theano to compute all of the gradients. Included gradient clipping to avoid gradient explosion.
all_grads = [T.clip(g,-3,3) for g in T.grad(cost_train, all_params)]
all_grads = lasagne.updates.total_norm_constraint(all_grads,3) #Could modify the clipping values if needed.

# Compile the update gradient.
updates = lasagne.updates.adam(all_grads, all_params, learning_rate=0.005)

# Compile the Theano functions.
train_func = theano.function(inputs = [x_sym, y_sym], outputs = [cost_train, train_model], updates=updates)
eval_func = theano.function(inputs = [x_sym, y_sym], outputs = [cost_eval, eval_model])




