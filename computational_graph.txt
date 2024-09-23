# computational_graph.py

# Name: Jeramee Oliver
# Project: 3
# Course: MSAI 531 A01 
# Title: Neural Networks Deep Learning
# Date: 9/22/24

import tensorflow as tf  # Use TensorFlow 2.x instead of compat.v1

class Node:
    def __init__(self, value=None):
        self.value = value
        self.gradients = []

    def add_gradient(self, gradient):
        self.gradients.append(gradient)

class FeedForwardLayer:
    def __init__(self, input_size, output_size):
        self.W = tf.Variable(tf.random.normal([input_size, output_size]))
        self.b = tf.Variable(tf.zeros([output_size]))

    def forward(self, x):
        return tf.matmul(x, self.W) + self.b

    @property
    def trainable_variables(self):
        # Corrected typo: changed self.w to self.W
        return [self.W, self.b]
