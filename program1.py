# Read an image and convert it into gray scale image without using built-in function.

from matplotlib import pyplot as plt
import numpy as np
import time

image = plt.imread('image/pexels-pixabay-56866.jpg')
fig,xy = plt.subplots(1,2)

start_time = time.time()
image_gray = image[:,:,0]*0.299 +image[:,:,1]*0.587 + image[:,:,2]*0.114
end_time = time.time()
elapsed_time  = end_time - start_time

xy[0].imshow(image)
xy[0].set_title('Original Image')
xy[0].axis('off')

xy[1].imshow(image_gray, cmap ='gray')
xy[1].set_title(f"Grayscale Image\nTime taken: {elapsed_time:.6f}seconds")

plt.show()