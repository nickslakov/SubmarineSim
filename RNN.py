import lasagne
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#from IPython import display


def rnn(data):

    #Divide up the data into a training pool and an evaluation pool

    data_train = data[:8000]
    data_eval = data[8000:]

    #Start of Network

    NUM_BATCHES = 1000
    BATCH_SIZE = 500
    SEQUENCE_LENGTH = 34
    NUM_INPUTS = 6
    NUM_OUTPUTS = 2
    CLIP_VAL = 5

    x_sym = T.tensor3() # Symbolic inputs
    y_sym = T.tensor3() # Symbolic target outputs

    l_in = lasagne.layers.InputLayer(shape = (BATCH_SIZE, None, NUM_INPUTS))
    l_hid1 = lasagne.layers.GRULayer(l_in, num_units=30, learn_init=True)
    #l_hid2 = lasagne.layers.GRULayer(l_hid1, num_units=10, learn_init=True)
    #l_hid3 = lasagne.layers.GRULayer(l_hid2,num_units=10,learn_init=True)
    l_reshape = lasagne.layers.ReshapeLayer(l_hid1,(-1,30))
    l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=NUM_OUTPUTS, nonlinearity=None)
    l_out = lasagne.layers.ReshapeLayer(l_dense, (BATCH_SIZE, None, NUM_OUTPUTS))

    # Define the training and evaluation outputs.
    train_out = lasagne.layers.get_output(l_out, {l_in: x_sym}, deterministic = False)
    eval_out = lasagne.layers.get_output(l_out, {l_in: x_sym}, deterministic = True)

    # All trainable params in the network.
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    # Define the Cost Function.
    cost_train = lasagne.objectives.squared_error(train_out, y_sym).mean()
    cost_eval = lasagne.objectives.squared_error(eval_out, y_sym).mean()

    # Use Theano to compute all of the gradients. Included gradient clipping to avoid gradient explosion.
    #all_grads = T.grad(cost_train, all_params)
    all_grads = [T.clip(g,-CLIP_VAL,CLIP_VAL) for g in T.grad(cost_train, all_params)]
    #all_grads = lasagne.updates.total_norm_constraint(all_grads,CLIP_VAL) #Could modify the clipping values if needed.

    # Compile the update gradient.
    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=0.007)

    # Compile the Theano functions.
    train_func = theano.function(inputs = [x_sym, y_sym], outputs = [cost_train, train_out], updates=updates)
    eval_func = theano.function(inputs = [x_sym, y_sym], outputs = [cost_eval, eval_out])

    #Start of the training loop.
    train_cost, val_cost = [],[]

    for e in range(NUM_BATCHES):

        #Start training

        index_train = np.random.randint(8000, size=BATCH_SIZE)
        train_data_current = np.zeros((BATCH_SIZE,SEQUENCE_LENGTH + 1,NUM_INPUTS))
        for i in range(BATCH_SIZE):
            train_data_current[i,:,:] = data_train[index_train[i],:,:]


        x_current = train_data_current[:,:-1,:]
        y_current = train_data_current[:,1:,:2]

        out = train_func(x_current,y_current)
        denom1 = []
        for i in range(BATCH_SIZE):
            for j in range(SEQUENCE_LENGTH):
                denom1.append((x_current[i,j,0]-y_current[i,j,0])**2 + (x_current[i,j,1]-y_current[i,j,1])**2)

        train_cost += [np.sqrt(out[0]/np.mean(denom1))]


        # Start evaluation

        index_eval = np.random.randint(2000,size=BATCH_SIZE)
        eval_data_current = np.zeros((BATCH_SIZE,SEQUENCE_LENGTH + 1,NUM_INPUTS))
        for i in range(BATCH_SIZE):
            eval_data_current[i,:,:] = data_eval[index_eval[i],:,:]

        x_current = eval_data_current[:,:-1,:]
        y_current = eval_data_current[:,1:,:2]

        out = eval_func(x_current,y_current)

        denom2 = []
        for i in range(BATCH_SIZE):
            for j in range(SEQUENCE_LENGTH):
                denom2.append((x_current[i,j,0]-y_current[i,j,0])**2 + (x_current[i,j,1]-y_current[i,j,1])**2)

        val_cost += [np.sqrt(out[0]/np.mean(denom2))]

        # Printouts

        if (e + 1) % 100 == 0:
            print 'Batch number: ' + repr(e)
            print '\tMean of last ten training costs: ' + repr(np.mean(train_cost[-10:-1]))
            print '\tMean of last ten evaluation costs: ' + repr(np.mean(val_cost[-10:-1]))
            print '\tMin training cost since last printout: ' + repr(np.min(train_cost[-99:-1]))
            print '\tMax training cost since last printout: ' + repr(np.max(train_cost[-99:-1]))

        # Plot some examples at termination

        if e == NUM_BATCHES-1 or train_cost[e] < 0.04:
            fig, ax = plt.subplots()
            ax.plot(y_current[0,:,0], y_current[0,:,1], '.r-')
            ax.plot(out[1][0,:,0], out[1][0,:,1], 'xb-')

            ax.plot(y_current[1, :, 0], y_current[1, :, 1], '.g-')
            ax.plot(out[1][1, :, 0], out[1][1, :, 1], 'xk-')

            plt.show()
            break

    fig, ax = plt.subplots()
    ax.plot(range(len(train_cost) - 100), train_cost[100:], '.k-')
    ax.plot(range(len(val_cost) - 100), val_cost[100:], '.b-')
    plt.show()



