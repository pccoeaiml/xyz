#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('MainImage.jpg') 
plt.imshow(image)
plt.title('Original Image')
plt.show()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap = 'gray')
plt.title('Gray Image')
plt.show()

img = cv2.imread('MainImage.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap = 'gray')
plt.title('Gray Scale Image')
plt.show()

#Sobel Filter
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
plt.imshow(sobel_combined, cmap = 'gray')
plt.title('Sobel Image')
plt.show()

#Median Filter
Median_image = cv2.medianBlur(img, 5)
plt.imshow(Median_image, cmap = 'gray')
plt.title('Median Image')
plt.show()

#Gaussian Filter
Gaussian_img = cv2.GaussianBlur(img, (5, 5), 0)
plt.imshow(Gaussian_img, cmap = 'gray')
plt.title('Gaussian Image')
plt.show()

#Average Filter
Avg_img = cv2.blur(img, (5, 5))
plt.imshow(Avg_img, cmap = 'gray')
plt.title('Average Image')
plt.show()


# In[ ]:




