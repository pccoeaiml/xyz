#!/usr/bin/env python
# coding: utf-8

# In[9]:


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# To avoid conflicts with certain system configurations
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def display_images(original_img_rgb, annotated_img_rgb):
    """
    Displays the original and annotated images side by side in Jupyter Notebook.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(annotated_img_rgb)
    axes[1].set_title('Detected Objects')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    model = YOLO(r'yolov8n.pt')

    while True:
        print("\nMenu:")
        print("1. Load and Detect Objects in Image")
        print("2. Exit")

        choice = input("Enter your choice (1/2): ")

        if choice == '1':
            img_path = r"BirdImage1.jpg"

            original_img = cv2.imread(img_path)
            if original_img is None:
                print("Error: Image not found or unable to load.")
                continue

            # Convert BGR to RGB for matplotlib
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = model(original_img)

            # Extract the result
            result = results[0] if isinstance(results, list) else results

            # Annotate the image
            annotated_img = result.plot()

            # Convert the annotated image to RGB
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            # Display images using matplotlib
            display_images(original_img_rgb, annotated_img_rgb)

        elif choice == '2':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()


# In[10]:


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def display_images(original_img_rgb, annotated_img_rgb):
    """
    Displays the original and annotated images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(annotated_img_rgb)
    axes[1].set_title('Detected Objects')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    model = YOLO(r'yolov8n (1).pt')

    while True:
        print("\nMenu:")
        print("1. Load and Detect Objects in Image")
        print("2. Exit")

        choice = input("Enter your choice (1/2): ")

        if choice == '1':
            img_path = r"BirdImage1.jpg"

            original_img = cv2.imread(img_path)
            if original_img is None:
                print("Error: Image not found or unable to load.")
                continue

            # Convert BGR to RGB for display
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Display the original image using matplotlib
            plt.imshow(original_img_rgb)
            plt.title("Original Image")
            plt.axis('off')
            plt.show()

            # Perform object detection
            results = model(original_img)

            # Extract the result
            result = results[0] if isinstance(results, list) else results

            # Annotate the image
            annotated_img = result.plot()

            # Convert the annotated image to RGB
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            # Display images using matplotlib
            display_images(original_img_rgb, annotated_img_rgb)

        elif choice == '2':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()


# In[ ]:




