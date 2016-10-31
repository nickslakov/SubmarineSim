import lasagne
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#from IPython import display


def rnn(data):

    #Start of Network

    NUM_BATCHES = 10
    BATCH_SIZE = 50
    SEQUENCE_LENGTH = 50
    NUM_INPUTS = 6
    NUM_OUTPUTS = 2
    CLIP_VAL = 1000


    X = np.random.randint(0,10,size=(BATCH_SIZE,SEQUENCE_LENGTH, NUM_INPUTS)) # Test inputs
    Y = np.random.randint(0,10,size=NUM_OUTPUTS) # Test outputs
    x_sym = T.tensor3() # Symbolic inputs
    y_sym = T.vector() # Symbolic target outputs

    l_in = lasagne.layers.InputLayer(shape = (BATCH_SIZE, SEQUENCE_LENGTH, NUM_INPUTS))
    l_hid1 = lasagne.layers.GRULayer(l_in, num_units=25)
    l_out = lasagne.layers.DenseLayer(l_hid1, num_units=NUM_OUTPUTS)

    train_out = lasagne.layers.get_output(l_out, {l_in: x_sym}, deterministic = False)
    #eval_model = lasagne.layers.get_output(l_out, {l_in: x_sym}, deterministic = True)

    #Print shape of output
    #print train_model.eval({x_sym:X}).shape
    #print Y.shape

    # Get outputs from network
    #train_out = train_model #.eval({x_sym:X})
    #eval_out = eval_model #.eval({x_sym:X})

    # All trainable params in the network
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    # Define the Cost Function
    cost_train = lasagne.objectives.squared_error(train_out, y_sym).mean()
    #cost_eval = lasagne.objectives.squared_error(eval_out, y_sym).mean()

    #print cost_train.eval({y_sym:Y})

    # Use Theano to compute all of the gradients. Included gradient clipping to avoid gradient explosion.
    all_grads = [T.clip(g,-CLIP_VAL,CLIP_VAL) for g in T.grad(cost_train, all_params)]
    all_grads = lasagne.updates.total_norm_constraint(all_grads,CLIP_VAL) #Could modify the clipping values if needed.

    # Compile the update gradient.
    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=0.05)

    # Compile the Theano functions.
    train_func = theano.function(inputs = [x_sym, y_sym], outputs = [cost_train, train_out], updates=updates)
    #eval_func = theano.function(inputs = [x_sym, y_sym], outputs = [cost_eval, eval_model])




    #Training Loop

    num_epochs = 1000
    current_epoch = 0

    num_updates = []
    train_cost, val_cost = [],[]
    x_current = np.zeros((BATCH_SIZE,SEQUENCE_LENGTH,6))
    y_current = np.zeros((BATCH_SIZE,SEQUENCE_LENGTH,2))

    for e in range(NUM_BATCHES - 1):
        x_current = data[e]
        y_current = data[e + 1][:][:][:1]

        out = train_func(x_current,y_current)
        train_cost += [out[0]]

        current_epoch += 1
     #  out = eval_func(X,Y)
     #  val_cost += [out[0]]

        #if e % 1000 == 0:WARNING (theano.gof.compilelock): Overriding existing lock by dead process '10928' (I am process '9276')
        print train_cost[-1]
        print len(train_cost)
            #num_updates = np.arange(0, e + 1, 1)

            #plt.plot(num_updates, train_cost)
            #plt.xlabel('Samples Processed', fontsize = 15)
            #plt.ylabel('Training Cost', fontsize = 15)
            #plt.title('Training Squared Error in Position Estimate' , fontsize = 20)
            #plt.grid('on')
            #plt.show()




