# jo_project3.py

# Name: Jeramee Oliver
# Project: 3
# Course: MSAI 531 A01 
# Title: Neural Networks Deep Learning
# Date: 9/22/24

# Import necessary libraries and files
# from autograph import linear_layer, simple_nn, simple_function
# from batch_datasets import load_imdb_reviews_dataset
# from mirrored_strategy import create_distributed_model
# from GradientTape import train_with_gradient_tape
# from autograph_timing import fn

import tensorflow as tf
import numpy as np
from batch_datasets import get_train_dataset
from tensorflow.keras.layers import TextVectorization
from computational_graph import FeedForwardLayer

# TensorFlow v2 enables eager execution by default
tf.config.run_functions_eagerly(True)

# Define TextVectorization layer
max_tokens = 20000
output_sequence_length = 100

vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=output_sequence_length
)

# Adapt vectorizer to the training dataset
train_raw_dataset = get_train_dataset()
train_text = [t.decode('utf-8', errors='ignore') for text, _ in train_raw_dataset for t in text.numpy()]
vectorize_layer.adapt(train_text)

# Preprocess text function
def preprocess_text(text, label):
    if text.ndim == 2:
        text = tf.squeeze(text, axis=1)
    text = vectorize_layer(text)
    text = tf.cast(text, tf.int32)
    label = tf.squeeze(label)
    label = tf.cast(label, tf.float64)
    return text, label

# FeedForwardNN class definition
class FeedForwardNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = FeedForwardLayer(input_size, hidden_size)
        self.layer2 = FeedForwardLayer(hidden_size, output_size)

    def forward(self, x):
        z1 = self.layer1.forward(x)
        a1 = tf.nn.relu(z1)
        output = self.layer2.forward(a1)
        return output

    def compute_loss(self, y_true, y_pred):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true_one_hot, logits=y_pred))

    @property
    def trainable_variables(self):
        return self.layer1.trainable_variables + self.layer2.trainable_variables

# Training step
def train_step(nn, x, y_true, optimizer):
    x = tf.cast(x, tf.float32)
    y_true = tf.cast(y_true, tf.float64)
    
    with tf.GradientTape() as tape:
        y_pred = nn.forward(x)
        loss = nn.compute_loss(y_true, y_pred)
    
    gradients = tape.gradient(loss, nn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, nn.trainable_variables))
    return loss

# Train the model
def train_model():
    input_size = output_sequence_length
    hidden_size = 4
    output_size = 2  # Number of classes
    learning_rate = 0.01
    num_epochs = 100

    train_dataset = get_train_dataset().map(preprocess_text)

    nn = FeedForwardNN(input_size, hidden_size, output_size)
    optimizer = tf.optimizers.SGD(learning_rate)

    for epoch in range(num_epochs):
        for text, label in train_dataset:
            loss = train_step(nn, text, label, optimizer)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')

if __name__ == "__main__":
    train_model()
