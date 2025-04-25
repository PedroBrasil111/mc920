import numpy as np
import cv2 as cv

size = 512

line1 = np.random.randint(0, 128, (size), dtype=np.uint8)
line2 = np.random.randint(128, 256, (size), dtype=np.uint8)

image = np.zeros((size, size), dtype=np.uint8)

i = 0
w = 3
while i < size:
    image[i:i+w, :] = line1
    image[i+w:i+2*w, :] = line2
    i += 2*w

#cv.imshow("image", image)
#cv.waitKey(0)

cv.imwrite(f"images/hor_{size}.png", image)
cv.imwrite(f"images/ver_{size}.png", image.T)

img3 = np.zeros((size, size), dtype=np.uint8)
img3[:]