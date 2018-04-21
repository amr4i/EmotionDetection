########################################################################################
# Author: Amrit Singhal, Year: 2018                                                    #
# Emotion Extraction from Images in the Circumplex model: Arousal and Valence values   #
# Details:                                                                             #
# Python Implementation of the paper 'Building Emotion Machines: Recognizing Image     #
# Emotion through Deep Neural Networks' by Hye-Rin Kim, Yeong-Seok Kim, Seon Joo Kim,  #
# and In-Kwon Lee.                                                                     #
# Model from https://github.com/amr4i/EmotionDetection                                 #
########################################################################################

import numpy as np
import tensorflow as tf

class ImageAVmodel(object):
    def __init__(self, input_length, num_neurons_in_layers):
        
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, input_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
        
        # l2 regularization loss (optional)
        l2_loss = tf.constant(1.0)

        # hidden layers
        inp = self.input_x

        l1 = tf.layers.dense(inp, num_neurons_in_layers[1], activation=tf.nn.relu, name='hidden_1')
        l2 = tf.layers.dense(l1, num_neurons_in_layers[2], activation=tf.nn.relu, name='hidden_2')
        l3 = tf.layers.dense(l2, num_neurons_in_layers[3], activation=tf.nn.relu, name='hidden_3')
        self.prediction = tf.layers.dense(l3, 1, name='prediction')

        # Squared loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.prediction, self.input_y))

        self.loss_summary = tf.summary.scalar('loss_summary', self.loss)
            


