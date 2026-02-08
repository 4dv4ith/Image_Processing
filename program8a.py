# 	Implement any two segmentation algorithms and compare the efficiency with ground truth.

import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image/pexels-pixabay-56866.jpg', cv2.IMREAD_GRAYSCALE)

# Global threshold (Otsu)
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive threshold
adaptive_thresh = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# Display results
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Otsu Global Threshold")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Adaptive Threshold")
plt.imshow(adaptive_thresh, cmap='gray')
plt.axis('off')

plt.show()
