#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

data = np.load('Image classification/mnist_compressed.npz')  
X_train, Y_train = data['train_images'], data['train_labels']
X_test, Y_test = data['test_images'], data['test_labels']

X_train = X_train.reshape(-1, 28, 56, 1) / 255.0
X_test = X_test.reshape(-1, 28, 56, 1) / 255.0

len(X_train)

Y_train = to_categorical(Y_train, num_classes=100)
Y_test = to_categorical(Y_test, num_classes=100)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 56, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(100, activation='softmax') 
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10, batch_size=64,
                    validation_data=(X_test, Y_test))

model.evaluate(X_test, Y_test)


# In[ ]:




