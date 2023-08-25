import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import data_processing
import model

# Load and preprocess data
dataset = pd.read_csv('data_test.csv', index_col=0)
df_stock = dataset.copy()
df_stock = df_stock.dropna()
df_stock = df_stock[['Open', 'High', 'Low', 'Close']]
df_stock_norm = df_stock.copy()
df_stock_norm = data_processing.normalize_data(df_stock_norm)

# Load data using data_processing.load_data function
valid_set_size_percentage = 10
test_set_size_percentage = 10
seq_len = 20
x_train, y_train, x_valid, y_valid, x_test, y_test = data_processing.load_data(
    df_stock_norm, seq_len, valid_set_size_percentage, test_set_size_percentage)

# Build the RNN model using model.build_rnn_model function
n_steps = seq_len - 1
n_inputs = 4
n_neurons = 200
n_outputs = 4
n_layers = 2
learning_rate = 0.001
X, y, loss, training_op, outputs = model.build_rnn_model(
    x_train, n_steps, n_inputs, n_neurons, n_outputs, n_layers, learning_rate)

# Training the model
batch_size = 50
n_epochs = 100
train_set_size = x_train.shape[0]
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    index_in_epoch = 0
    perm_array = np.arange(x_train.shape[0])
    
    for iteration in range(int(n_epochs * train_set_size / batch_size)):
        x_batch, y_batch = model.get_next_batch(
            batch_size, x_train, y_train, index_in_epoch, perm_array)
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        
        if iteration % int(5 * train_set_size / batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
            print('%.2f epochs: MSE train/valid = %.6f/%.6f' % (
                iteration * batch_size / train_set_size, mse_train, mse_valid))
    
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})

# Visualization
comp = pd.DataFrame({'Column1': y_test[:, 3], 'Column2': y_test_pred[:, 3]})
plt.figure(figsize=(10, 5))
plt.plot(comp['Column1'], color='blue', label='Target')
plt.plot(comp['Column2'], color='black', label='Prediction')
plt.legend()
plt.show(block=False)