__author__ = 'CLH'

import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import matplotlib.pyplot as plt

"""
simple seq2seq model: fitting curve
"""

"""
    hyper-parameter settings
"""
epochs = 100    # number of epochs
batch_size = 50     # batch size
sequence_length = 30    # length of sequence
num_layers = 3  # number of layers
input_dim = 1   # dimension of input
output_dim = 1  # dimension of output
hidden_dim = 12 # dimension of hidden state
learning_rate = 0.01    # learning rate


def do_generate_x_y(batch_size, sequence_length):
    """
    generate x and y
    :param batch_size: the size of batch
    :param sequence_length: the length of sequence
    :return: batch_x and batch_y
    """
    batch_x, batch_y,batch_sequence_length = [], [], []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sin_data = amp_rand * np.sin(np.linspace(sequence_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
                                                 sequence_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand, sequence_length * 2))

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sig_data = amp_rand * np.sin(np.linspace(sequence_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
                                                 sequence_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand, sequence_length * 2)) + sin_data

        # [sequence_length, batch_size, output_dim]
        batch_x.append(np.array([sig_data[:sequence_length]]).T)
        batch_y.append(np.array([sig_data[sequence_length:]]).T)
        batch_sequence_length.append(sequence_length)
    # print(len(batch_x))
    # [batch_size, sequence_length, output_dim]
    # batch_x = np.array(batch_x).transpose((1,0,2))
    # batch_y = np.array(batch_y).transpose((1,0,2))
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    batch_sequence_length = np.array(batch_sequence_length)

    return batch_x, batch_y, batch_sequence_length

def generate_x_y(is_Train, batch_size, sequence_length):
    """
    generate x and y
    :param batch_size: the size of batch
    :param sequence_length: the length of sequence
    :return: batch_x and batch_y
    """
    if is_Train:
        return do_generate_x_y(batch_size, sequence_length)
    else:
        return do_generate_x_y(batch_size, sequence_length * 2)


def get_inputs(input_dim, output_dim):
    encoder_input = tf.placeholder(tf.float32, shape=(None, sequence_length, input_dim), name='encode_inputs')
    decoder_input = tf.placeholder(tf.float32, shape=(None, sequence_length, output_dim), name='decoder_input')
    expected_output = tf.placeholder(tf.float32, shape=(None, sequence_length, output_dim), name='expected_output')
    batch_sequence_length = tf.placeholder(tf.int32, shape=(None,), name='batch_sequence_length')

    return encoder_input, decoder_input, expected_output, batch_sequence_length

def get_encoder_layer(encoder_input,num_layers, hidden_dim, batch_sequence_length):

    def get_lstm_cell(hidden_dim):
        lstm_cell = tf.contrib.rnn.LSTMCell(hidden_dim, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(hidden_dim) for _ in range(num_layers)])

    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_input,dtype=tf.float32, sequence_length = batch_sequence_length)

    return encoder_output, encoder_state

def get_decoder_layer(decoder_input, encoder_state, num_layers, hidden_dim, output_dim, batch_sequence_length):

    def get_lstm_cell(hidden_dim):
        lstm_cell = tf.contrib.rnn.LSTMCell(hidden_dim, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(hidden_dim) for _ in range(num_layers)])

    # output layer
    output_layer = Dense(output_dim,kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1))

    with tf.variable_scope("decode"):
        traning_helper = tf.contrib.seq2seq.TrainingHelper(inputs = decoder_input, sequence_length = batch_sequence_length)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, traning_helper, encoder_state, output_layer)
        training_decoder_output, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(training_decoder)

    return training_decoder_output, final_state, final_sequence_lengths

def seq2seq_model(encoder_input, decoder_input, num_layers, hidden_dim, output_dim, batch_sequence_length):
    encoder_output, encoder_state = get_encoder_layer(encoder_input,num_layers,hidden_dim, batch_sequence_length)
    training_decoder_output, final_state,final_sequence_length = get_decoder_layer(decoder_input, encoder_state, num_layers, hidden_dim,output_dim, batch_sequence_length)
    return training_decoder_output

train_graph = tf.get_default_graph()
with train_graph.as_default():
    encoder_input, decoder_input, expected_output, batch_sequence_length = get_inputs(input_dim, output_dim)
    ending = tf.strided_slice(decoder_input, [0,0], [batch_size,-1],[1,1])
    # print(ending.shape)
    decoder_input = tf.concat([tf.fill([batch_size,1,1],0.0),ending],1) # processing the input of decoder!!!!

    training_decoder_output = seq2seq_model(encoder_input, decoder_input, num_layers,hidden_dim, output_dim, batch_sequence_length)
    mask = tf.constant(1.0, dtype=np.float32, shape=[batch_size,sequence_length])
    print(training_decoder_output.rnn_output.shape)
    print(mask.shape)
    targets = tf.reshape(expected_output,[-1])
    logits = tf.reshape(training_decoder_output.rnn_output,[-1])
    print(logits.shape,targets.shape)
    with tf.name_scope("optimization"):
        cost = tf.reduce_mean(
            tf.pow(tf.subtract(logits, targets), 2.0)
        )
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


sess = tf.InteractiveSession()
def train_batch(batch_size):
    X, Y, L = generate_x_y(True, batch_size, sequence_length)
    # print(X.shape, Y.shape, batch_sequence_length.shape)
    feed_dict = {encoder_input: X,
                 decoder_input: Y,
                 expected_output: Y,
                 batch_sequence_length: L}
    _, loss_t = sess.run([optimizer, cost], feed_dict)
    return loss_t

def test_batch(batch_size):
    X, Y, L = generate_x_y(True, batch_size, sequence_length)
    feed_dict = {encoder_input: X,
                 decoder_input: Y,
                 expected_output: Y,
                 batch_sequence_length: L}
    _, loss_t = sess.run([optimizer, cost], feed_dict)
    return loss_t

# training
train_losses = []
test_losses = []
sess.run(tf.global_variables_initializer())
for t in range(epochs):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)
    if t % 10 == 0:
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        print(train_loss, test_loss)

# Plot loss over time:
plt.figure(figsize=(12, 6))
plt.plot(np.array(range(0, len(test_losses))) /
    float(len(test_losses) - 1) * (len(train_losses) - 1),
    np.log(test_losses),label="Test loss")

plt.plot(np.log(train_losses),label="Train loss")
plt.title("Training errors over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
plt.show()


# # Testing i don't know to define inference helper
# # Test
# nb_predictions = 5
# # sequence_length *= 2
# print("visualize {} predictions data:".format(nb_predictions))
# X, Y, L = generate_x_y(True, batch_size, sequence_length)
# feed_dict = {encoder_input:X, decoder_input:Y, expected_output:Y, batch_sequence_length:L}
# logits = sess.run([logits], feed_dict)
# preout = np.array(logits).reshape(batch_size,sequence_length,1)
#
# preout = preout.transpose((1,0,2))
# X = X.transpose((1,0,2))
# Y = Y.transpose((1,0,2))
# # print(preout.shape, X.shape, Y.shape)
#
# for j in range(nb_predictions):
#     plt.figure(figsize=(12, 3))
#
#     for k in range(output_dim):
#         past = X[:, j, k]
#         expected = Y[:, j, k]
#
#         pred = preout[:, j, k]
#
#         label1 = "past" if k == 0 else "_nolegend_"
#         label2 = "future" if k == 0 else "_nolegend_"
#         label3 = "Pred" if k == 0 else "_nolegend_"
#         plt.plot(range(len(past)), past, "o--b", label=label1)
#         plt.plot(range(len(past), len(expected) + len(past)),
#                  expected, "x--b", label=label2)
#         plt.plot(range(len(past), len(pred) + len(past)),
#                  pred, "o--y", label=label3)
#
#     plt.legend(loc='best')
#     plt.title("Predictions vs. future")
#     plt.show()
