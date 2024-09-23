# batch_datasets.py

# Name: Jeramee Oliver
# Project: 3
# Course: MSAI 531 A01 
# Title: Neural Networks Deep Learning
# Date: 9/22/24

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def get_train_dataset(batch_size=32, shuffle_buffer_size=10000):
    # Load and prepare the IMDB reviews dataset
    datasets, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

    # Extract training data (datasets['train'] is a tf.data.Dataset)
    train_dataset = datasets['train']

    # Shuffle and batch the dataset
    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)

    return train_dataset

# Optional: Define validation or test datasets similarly
def get_test_dataset(batch_size=32):
    datasets, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    test_dataset = datasets['test'].batch(batch_size)
    
    return test_dataset

