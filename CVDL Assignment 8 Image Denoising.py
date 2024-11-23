#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.datasets import mnist

# Load MNIST dataset and normalize
(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add noise and clip
noise_factor = 0.5
x_train_noisy = np.clip(x_train + noise_factor * np.random.normal(size=x_train.shape), 0., 1.)
x_test_noisy = np.clip(x_test + noise_factor * np.random.normal(size=x_test.shape), 0., 1.)

# Autoencoder model
input_img = Input(shape=(28, 28, 1))
x = Flatten()(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(32, activation='relu')(x)
x = Dense(64, activation='relu')(encoded)
decoded = Reshape((28, 28, 1))(Dense(28 * 28, activation='sigmoid')(x))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train_noisy, x_train, epochs=5, batch_size=128, validation_data=(x_test_noisy, x_test))

# Denoise images
denoised = autoencoder.predict(x_test_noisy)

# Display images
def plot_images(original, noisy, denoised, n=10):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        for idx, img in enumerate([original, noisy, denoised]):
            plt.subplot(3, n, i + 1 + idx * n)
            plt.imshow(img[i].reshape(28, 28), cmap='binary')
            plt.axis('off')
    plt.show()

plot_images(x_test, x_test_noisy, denoised)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.datasets import mnist
import os

# Function to load a custom dataset if specified
def load_dataset(dataset_path=None):
    if dataset_path and os.path.exists(dataset_path):
        # Example: Load custom dataset from a NumPy file
        data = np.load(dataset_path)
        return data['x_train'], data['x_test']
    else:
        # Default to MNIST dataset
        (x_train, _), (x_test, _) = mnist.load_data()
        return x_train, x_test

# Load dataset
dataset_path = None  # Update this with the dataset path if needed
x_train, x_test = load_dataset(dataset_path)

# Normalize dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add noise and clip
noise_factor = 0.5
x_train_noisy = np.clip(x_train + noise_factor * np.random.normal(size=x_train.shape), 0., 1.)
x_test_noisy = np.clip(x_test + noise_factor * np.random.normal(size=x_test.shape), 0., 1.)

# Reshape for input to the model
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train_noisy = x_train_noisy.reshape(-1, 28, 28, 1)
x_test_noisy = x_test_noisy.reshape(-1, 28, 28, 1)

# Autoencoder model
input_img = Input(shape=(28, 28, 1))
x = Flatten()(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(32, activation='relu')(x)
x = Dense(64, activation='relu')(encoded)
decoded = Reshape((28, 28, 1))(Dense(28 * 28, activation='sigmoid')(x))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(x_train_noisy, x_train, epochs=5, batch_size=128, validation_data=(x_test_noisy, x_test))

# Denoise images
denoised = autoencoder.predict(x_test_noisy)

# Display images
def plot_images(original, noisy, denoised, n=10):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        for idx, img in enumerate([original, noisy, denoised]):
            plt.subplot(3, n, i + 1 + idx * n)
            plt.imshow(img[i].reshape(28, 28), cmap='binary')
            plt.axis('off')
    plt.show()

plot_images(x_test, x_test_noisy, denoised)


# In[ ]:




