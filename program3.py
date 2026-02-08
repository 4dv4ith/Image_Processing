#3.	Read an image and convert it into binary image using thresholding.

from matplotlib import pyplot as plt
import numpy as np
import time

image = plt.imread('image/pexels-pixabay-56866.jpg')
fig,xy = plt.subplots(1,4)
threshold = 0.5
image_gray = image[:,:,0]*0.299 +image[:,:,1]*0.587 + image[:,:,2]*0.114

if image_gray.max() > 1:
    threshold = 128
else:
    threshold = 0.5

image_binary = np.where(image_gray >= threshold,1,0)

xy[0].imshow(image)
xy[0].set_title('Original Image')
xy[0].axis('off')


xy[1].imshow(image_gray, cmap='grey')
xy[1].set_title('Greyscale Image')
xy[1].axis('off')

xy[2].imshow(image_binary, cmap='grey')
xy[2].set_title('binary image')
xy[2].axis('off')

plt.show()

