import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as F

image_path = F.get_image_path("watch.png")
image = cv.imread(image_path, cv.IMREAD_COLOR)

kernel = np.array(
    [0.289, 0.5870, 0.1140], dtype=np.float32
)

transf = np.dot(image, kernel.T)
transf[transf > 255] = 255
result = transf.astype(np.uint8)

cv.cvtColor(result, cv.COLOR_RGB2BGR, dst=result)

cv.imshow('Result', result)
cv.waitKey(0)
