# Edge-based segmentation

import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image/pexels-pixabay-56866.jpg', cv2.IMREAD_GRAYSCALE)

# Step 1: Blur to remove noise
blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

# Step 2: Apply Canny edge detector
edges = cv2.Canny(blurred, 50, 150)   # low- and high-threshold

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title("Original"), plt.axis('off')
plt.subplot(1, 3, 2), plt.imshow(blurred, cmap='gray'), plt.title("Gaussian Blurred"), plt.axis('off')
plt.subplot(1, 3, 3), plt.imshow(edges, cmap='gray'), plt.title("Canny Edge Segmentation"), plt.axis('off')
plt.show()
