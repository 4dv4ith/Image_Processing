#Closing

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read image in grayscale
img = cv2.imread('image/pexels-pixabay-56866.jpg', cv2.IMREAD_GRAYSCALE)

# Step 2: Apply threshold to obtain binary image
_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

# Step 3: Define structuring element (kernel)
kernel = np.ones((5, 5), np.uint8)

# Step 4: Apply Closing (Dilation → Erosion)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Step 5: Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Original Grayscale")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Binary Image")
plt.imshow(binary, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Closing (Dilation → Erosion)")
plt.imshow(closing, cmap='gray')
plt.axis('off')

plt.show()
