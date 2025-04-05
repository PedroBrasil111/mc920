import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

image = np.full((256, 256, 3), 255, dtype=np.uint8)

cv.imwrite("t1/images/full.png", image)