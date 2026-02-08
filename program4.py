#	Display the histogram of the gray scale image.

from matplotlib import pyplot as plt
import numpy as np

image = plt.imread('image/pexels-pixabay-56866.jpg')
fig,xy = plt.subplots(1,2)

image_gray = image[:,:,0]*0.299 +image[:,:,1]*0.587 + image[:,:,2]*0.114

xy[0].imshow(image_gray, cmap = 'gray')
xy[0].set_title('Grayscale Image')
xy[0].axis('off')
xy[1].hist(image_gray.ravel(), bins = 256, color = 'black')

xy[1].set_title('Histogram of Grayscale Image')
xy[1].set_xlabel('Pixel Intensity')
xy[1].set_ylabel('Frequency')
plt.show()


