import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

size = 216
k = 8

kernel = np.ones((k, k), dtype=np.float32)

image1 = np.identity(size, dtype=np.uint16) * 1
image2 = np.fliplr(np.identity(size, dtype=np.uint16) * 1)
for i in range(0, size, 8*k):
    image1 += np.roll(image1, i, axis=1)
    image2 += np.roll(image2, i, axis=1)
image1 = cv.filter2D(image1, -1, kernel)
image2 = cv.filter2D(image2, -1, kernel)

image3 = np.zeros((size, size), dtype=np.uint16)
image3[size//2 - 10: size//2 + 10, :] = 1
image4 = np.transpose(image3)

image = image1 + image2 #+ image3 + image4
image[image > 0] = 255
image = image.astype(np.uint8)

#ruido = np.random.randint(0, 50, (size, size), dtype=np.uint16)
#image = np.clip(image.astype(np.uint16) + ruido, 0, 255).astype(np.uint8)

cv.imshow("Image", image)
cv.waitKey(0)
cv.imwrite("t1/images/bordado.png", image.astype(np.uint8))


