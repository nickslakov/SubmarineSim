import lasagne
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import collections
import csv


def rnn(data):

    #Divide up the data into a training pool and an evaluation pool

    data_train = data[:8000]
    data_eval = data[8000:]

    #Start of Network

    NUM_BATCHES = 3000
    BATCH_SIZE = 1000
    SEQUENCE_LENGTH = 34
    NUM_INPUTS = 6
    NUM_OUTPUTS = 2
    CLIP_VAL = 2

    x_sym = T.tensor3() # Symbolic inputs
    y_sym = T.tensor3() # Symbolic target outputs


    init_state = theano.shared(np.zeros((BATCH_SIZE,30)))
    l_in_secondary = lasagne.layers.InputLayer(shape = (None, 30))

    l_in = lasagne.layers.InputLayer(shape = (None, None, NUM_INPUTS))
    l_hid1 = lasagne.layers.GRULayer(l_in, num_units=30, hid_init=l_in_secondary)
    l_reshape = lasagne.layers.ReshapeLayer(l_hid1,(-1,30))
    l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=NUM_OUTPUTS, nonlinearity=None)
    l_out = lasagne.layers.ReshapeLayer(l_dense, (BATCH_SIZE, SEQUENCE_LENGTH, NUM_OUTPUTS))


    l_out_eval = lasagne.layers.ReshapeLayer(l_dense, (BATCH_SIZE,1,NUM_OUTPUTS))

    # Define the training and evaluation outputs.
    train_out = lasagne.layers.get_output(l_out, {l_in: x_sym, l_in_secondary: init_state}, deterministic = False)
    hidden_state, eval_out = lasagne.layers.get_output([l_hid1, l_out_eval], {l_in: x_sym, l_in_secondary: init_state})

    # Updates for evaluation
    hidden_update = collections.OrderedDict()
    hidden_update[init_state] = hidden_state[:,-1,:]

    # Reset updates for after evaluation
    reset_update = collections.OrderedDict()
    reset_update[init_state] = np.zeros((BATCH_SIZE,30))

    # All trainable params in the network.
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    # Define the Cost Function.
    cost_train = lasagne.objectives.squared_error(train_out, y_sym).mean()

    # Use Theano to compute all of the gradients. Included gradient clipping to avoid gradient explosion.
    all_grads = [T.clip(g,-CLIP_VAL,CLIP_VAL) for g in T.grad(cost_train, all_params)]

    # Compile the update gradient.
    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=0.007)

    # Compile the Theano functions.
    train_func = theano.function(inputs = [x_sym, y_sym], outputs = [cost_train, train_out], updates=updates)
    eval_func = theano.function(inputs=[x_sym], outputs = [eval_out], updates=hidden_update)
    reset_func = theano.function(inputs = [x_sym], outputs = [hidden_state], updates=reset_update)

    #Start of the training loop.
    train_cost, eval_cost = [], []

    for e in range(NUM_BATCHES):

        #Start training

        index_train = np.random.randint(8000, size=BATCH_SIZE)
        train_data_current = np.zeros((BATCH_SIZE,SEQUENCE_LENGTH + 1,NUM_INPUTS))
        for i in range(BATCH_SIZE):
            train_data_current[i,:,:] = data_train[index_train[i],:,:]


        x_current = train_data_current[:,:-1,:]
        y_current = train_data_current[:,1:,:2]

        denom = []
        train_out = train_func(x_current,y_current)
        for i in range(BATCH_SIZE):
            for j in range(SEQUENCE_LENGTH):
                denom += [(x_current[i,j,0]-y_current[i,j,0])**2 + (x_current[i,j,1]-y_current[i,j,1])**2]

        train_cost += [np.sqrt(train_out[0]/np.mean(denom))]

        # Start Eval

        index_eval = np.random.randint(2000, size=BATCH_SIZE)
        eval_data_current = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH + 1, NUM_INPUTS))
        for i in range(BATCH_SIZE):
            eval_data_current[i, :, :] = data_eval[index_eval[i], :, :]
        x_current2 = eval_data_current[:, :-1, :]
        y_current2 = eval_data_current[:, 1:, :2]

        x_test = np.zeros((BATCH_SIZE, 1, NUM_INPUTS))
        x_test[:, :, :] = x_current2[:, :1, :]

        eval_out = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, 2))
        for i in range(SEQUENCE_LENGTH):
            x_test[:, :, 2:] = x_current2[:, i:i + 1, 2:]
            eval_out[:, i:i + 1, :] = eval_func(x_test)[0]
            x_test[:, :, :2] = eval_out[:, i:i + 1, :]

        denom2 = []
        for i in range(BATCH_SIZE):
            denom2 += [y_current2[i, -1, 0] ** 2 + y_current2[i, -1, 1] ** 2]

        eval_cost += [np.sqrt(lasagne.objectives.squared_error(eval_out[:,-1,:],y_current2[:,-1,:]).mean()/np.mean(denom2))]

        reset_func(x_current2)

        # Printouts

        if (e + 1) % 100 == 0:
            print 'Batch number: ' + repr(e)
            print '\tMean of last ten training costs: ' + repr(np.mean(train_cost[-10:-1]))
            print '\tMean of last ten evaluation costs: ' + repr(np.mean(eval_cost[-10:-1]))
            print '\tMin training cost since last printout: ' + repr(np.min(train_cost[-99:-1]))
            print '\tMax training cost since last printout: ' + repr(np.max(train_cost[-99:-1]))

        # Plot some examples at termination

        if e == NUM_BATCHES-1 or train_cost[e] < 0.05:
            x_test = np.zeros((BATCH_SIZE,1,6))
            x_test[:,:,:] = x_current[:,:1,:]

            eval_out = np.zeros((BATCH_SIZE,SEQUENCE_LENGTH,2))
            for i in range(SEQUENCE_LENGTH):
                x_test[:, :, 2:] = x_current[:, i:i + 1, 2:]

                eval_out[:,i:i+1,:] = eval_func(x_test)[0]

                x_test[:,:,:2] = eval_out[:,i:i+1,:]

            # Some example plots
            for i in range(10):
                fig, ax = plt.subplots()
                ax.plot(y_current[i, :, 0], y_current[i, :, 1], '.k-')
                ax.plot(eval_out[i, :, 0], eval_out[i, :, 1], 'xb-')
                ax.plot(train_out[1][i, :, 0], train_out[1][i, :, 1], '.r-')
                plt.show()

                fig, ax = plt.subplots()
                ax.plot(y_current[i, :, 0], y_current[i, :, 1], '.k-')
                ax.plot(train_out[1][i, :, 0], train_out[1][i, :, 1], 'xr-')
                plt.show()

                fig, ax = plt.subplots()
                ax.plot(y_current[i,:,0], y_current[i,:,1], '.k-')
                plt.show()

            break

    # Plot cost. Remove early costs to see structure in cost in later epochs
    fig, ax = plt.subplots()
    ax.plot(range(100,len(train_cost)), train_cost[100:], '.k-')
    ax.plot(range(100,len(eval_cost)), eval_cost[100:], 'xb-')
    plt.show()

    with open('cost.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, deliminator=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(train_cost)



