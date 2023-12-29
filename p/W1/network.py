import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)

# Build a simple Sequential model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])