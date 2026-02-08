import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import wiener

# Step 1: Read the image in grayscale
img = cv2.imread('image/pexels-pixabay-56866.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("⚠ Image not found! Check path.")
else:
    # Step 2: Simulate degradation (blur + noise)
    kernel = np.ones((5, 5)) / 25
    blurred = cv2.filter2D(img, -1, kernel)

    noise = np.random.normal(0, 10, img.shape)
    noisy_blurred = blurred + noise
    noisy_blurred = np.clip(noisy_blurred, 0, 255).astype(np.uint8)

    # Step 3: Wiener Filter restoration
    restored_img = wiener(noisy_blurred, kernel, balance=0.1)

    # Convert complex → real → uint8 → showable
    restored_img = np.abs(restored_img)
    restored_img = (restored_img / restored_img.max() * 255).astype(np.uint8)

    # Step 4: Display
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Noisy + Blurred Image")
    plt.imshow(noisy_blurred, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Restored Image (Wiener Filter)")
    plt.imshow(restored_img, cmap='gray')
    plt.axis('off')

    plt.show()
