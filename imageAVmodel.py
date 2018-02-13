import numpy as np
import tensorflow as tf

class ImageAVmodel(object):
    def __init__(self, input_length, num_hidden_layers, num_neurons_in_layers):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, input_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
        
        # l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # hidden layers
        inp = self.input_x

        for num_layer in range(1, num_hidden_layers+1):
            with tf.name_scope('layer-%s' %num_layer):

                nn_to = num_neurons_in_layers[num_layer]
                nn_from = num_neurons_in_layers[num_layer-1]
                W = tf.Variable(tf.random_uniform([nn_from, nn_to], -1.0, 1.0), name='W')   
                b = tf.Variable(tf.random_uniform([nn_to], -1.0, 1.0), name='b')    

                layer = tf.nn.xw_plus_b(
                    x = inp,
                    weights = W,
                    biases = b,
                    name = 'layer')
                activated_layer = tf.nn.relu(layer, name='activated_layer')
                inp = activated_layer

        
        # output layer
        with tf.name_scope('output-layer'):
            nn_from = num_neurons_in_layers[num_hidden_layers]
            W = tf.Variable(tf.random_uniform([nn_from, 1], -1.0, 1.0), name='W')
            b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='b')        
            self.prediction = tf.nn.xw_plus_b(
                x = inp,
                weights = W,
                biases = b,
                name = 'prediction')

        # Squared loss
        with tf.name_scope('loss'):
            self.loss = tf.losses.mean_squared_error(labels = self.input_y, predictions = self.prediction)  #  only named arguments accepted

        with tf.name_scope('loss_summary'):
            self.loss_summary = tf.summary.scalar('loss_summary', self.loss)
            


