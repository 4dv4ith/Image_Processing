# Apply histogram equalization on an image and display the result

from matplotlib import pyplot as plt
import numpy as np
import cv2

# Step 1: Read the image
image = plt.imread('image/pexels-pixabay-56866.jpg')

# Step 2: Convert to grayscale (manual formula)
image_gray = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114

# Step 3: Normalize grayscale image to [0, 255]
image_gray = (image_gray - image_gray.min()) / (image_gray.max() - image_gray.min()) * 255
image_gray = image_gray.astype(np.uint8)

# Step 4: Apply histogram equalization using OpenCV
equalized_image = cv2.equalizeHist(image_gray)

# Step 5: Display original grayscale and equalized image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image_gray, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(equalized_image, cmap='gray')
axes[1].set_title('Histogram Equalized Image')
axes[1].axis('off')

plt.show()

# Step 6 (optional): Plot histograms before and after equalization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(image_gray.flatten(), bins=256, color='gray')
plt.title('Histogram Before Equalization')

plt.subplot(1, 2, 2)
plt.hist(equalized_image.flatten(), bins=256, color='gray')
plt.title('Histogram After Equalization')

plt.show()
