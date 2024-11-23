#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPool2D,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
data = np.load(r"Image classification\mnist_compressed.npz")

X_train= data["train_images"]/255
X_test = data["test_images"]/255

model = Sequential([
    Conv2D(8,(3,3),input_shape=(28,56,1),activation='relu'),
    MaxPool2D((2,2)),
    Conv2D(16,(3,3),activation='relu'),
    MaxPool2D((2,2)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(100,activation = 'softmax')
])

model.compile(optimizer="adam",loss=SparseCategoricalCrossentropy(from_logits=True),metrics=["Accuracy"])
model.fit(X_train,data["train_labels"],epochs=10)
prediction = model.predict(data["test_images"][14].reshape(1, 28, 56, 1))
print(np.argmax(prediction))
print(prediction)
import matplotlib.pyplot as plt
plt.imshow(data["test_images"][14])


# In[ ]:




