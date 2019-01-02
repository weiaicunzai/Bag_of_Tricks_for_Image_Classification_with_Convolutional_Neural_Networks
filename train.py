


import numpy as np
import cv2


path = '/Users/didi/Downloads/train/2d281959a02178bbcdeea424c8757b1d.jpg'

image = cv2.imread(path)

print(image.dtype)
print(np.max(image))
image = image.astype('float32')

print(image.dtype)
print(np.max(image))



