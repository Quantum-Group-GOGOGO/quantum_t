import tensorflow as tf
import numpy as np

def get_next_batch(batch_size, x_train, y_train, index_in_epoch, perm_array):
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)
        start = 0
        index_in_epoch = batch_size
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

def build_rnn_model(x_train,n_steps, n_inputs, n_neurons, n_outputs, n_layers, learning_rate):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    
    X = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_outputs])
    
    index_in_epoch = 0
    perm_array = np.arange(x_train.shape[0])
    
    # RNN layers
    layers = [tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
              for _ in range(n_layers)]
    
    multi_layer_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(layers)
    rnn_outputs, states = tf.compat.v1.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.compat.v1.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    outputs = outputs[:, n_steps - 1, :]
    
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
    return X, y, loss, training_op, outputs
