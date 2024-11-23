#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(image_dir, label_dir, image_size=(256, 256)):
    images = []
    labels = []
    
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)
    
    # Create a mapping between images and labels
    for image_name in image_files:
        # Extract the base name (before the extension) from the image filename
        base_name = os.path.splitext(image_name)[0]
        
        # Create the corresponding label filename (adjust this as per your pattern)
        label_name = f"annotated_dog.8790.jpg"  # Adjust based on your naming convention
        
        # Check if the label file exists
        if label_name in label_files:
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            
            # Read and resize the images and labels
            image = load_img(image_path, target_size=image_size)
            label = load_img(label_path, target_size=image_size, color_mode='grayscale')  # For single-channel labels
            
            image = img_to_array(image) / 255.0  # Normalize the image
            label = img_to_array(label) / 255.0  # Normalize the label (0-1 for binary segmentation)
            
            images.append(image)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Example usage
image_dir = r"semantic segmentation\semantic segmentation\Images"
label_dir = r"semantic segmentation\semantic segmentation\Labels"
X, y = load_data(image_dir, label_dir)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

from sklearn.model_selection import train_test_split

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the split data
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


from tensorflow.keras import layers, models

def unet_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    # Decoder
    up4 = layers.UpSampling2D((2, 2))(pool3)
    concat4 = layers.concatenate([conv3, up4], axis=-1)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat4)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    
    up5 = layers.UpSampling2D((2, 2))(conv4)
    concat5 = layers.concatenate([conv2, up5], axis=-1)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat5)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = layers.UpSampling2D((2, 2))(conv5)
    concat6 = layers.concatenate([conv1, up6], axis=-1)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    
    # Output layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv6)  # Sigmoid for binary segmentation
    
    model = models.Model(inputs, output)
    return model

# Initialize the model
model = unet_model()
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
predictions = model.predict(X_test)

import matplotlib.pyplot as plt

def visualize_segmentation(image, label, prediction):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(label.squeeze(), cmap='gray')
    plt.title("True Label")
    
    plt.subplot(1, 3, 3)
    plt.imshow(prediction.squeeze(), cmap='gray')
    plt.title("Predicted Mask")
    
    plt.show()

# Example visualization on a few test samples
for i in range(3):  # Display first 3 test samples
    visualize_segmentation(X_test[i], y_test[i], predictions[i])


# In[ ]:




