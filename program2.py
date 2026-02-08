#Read an image and display the RGB channels of the images separately.

from matplotlib import pyplot as plt
import numpy as np
import time

image = plt.imread('image/pexels-pixabay-56866.jpg')

fig,xy = plt.subplots(1,4)
start_time = time.time()
r_channel = image[:,:,0]
g_channel = image[:,:,1]
b_channel = image[:,:,2]
end_time = time.time()
elapsed_time = end_time - start_time

xy[0].imshow(image)
xy[0].set_title('Original Image')
xy[0].axis('off')


xy[1].imshow(r_channel, cmap='Reds')
xy[1].set_title('Red Channel')
xy[1].axis('off')

xy[2].imshow(g_channel, cmap='Greens')
xy[2].set_title('Green Channel')
xy[2].axis('off')

xy[3].imshow(b_channel,cmap = 'Blues')
xy[3].set_title('Blue Channel')
xy[3].axis('off')

plt. show()

print(elapsed_time)