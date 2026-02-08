# Corrected edge-detection script (Canny). Save as e.g. edges.py and run.
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Option A: read using OpenCV (recommended) ---
img_bgr = cv2.imread('image/pexels-pixabay-56866.jpg')   # BGR color
if img_bgr is None:
    raise FileNotFoundError("File not found: image/pexels-pixabay-56866.jpg")

# convert to grayscale
image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# --- Option B: read using matplotlib (alternative) ---
# Uncomment to use plt.imread instead:
# img = plt.imread('image/pexels-pixabay-56866.jpg')
# # If matplotlib returns float image in [0,1], convert to 0-255 uint8:
# if img.dtype == np.float32 or img.dtype == np.float64:
#     img = (img * 255).astype(np.uint8)
# # if RGB -> convert to grayscale
# if img.ndim == 3:
#     image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# else:
#     image = img.copy()

print("image shape (grayscale):", image.shape)

# Apply Gaussian blur to reduce noise (important before Canny)
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# Canny edge detector: tune the low and high thresholds for your image
low_thresh = 50
high_thresh = 150
edges = cv2.Canny(blurred_image, low_thresh, high_thresh)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original (grayscale)')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Blurred')
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f'Canny Edges ({low_thresh},{high_thresh})')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Optional: save edges to file
cv2.imwrite('edges_output.png', edges)
