import numpy as np
import os
import cv2 as cv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alignment import *
import time

img = cv.imread("images/arial.png", cv.IMREAD_GRAYSCALE)
for angle in list(range(-90, 91, 10)) + [1, 2, 3]:
    rotated = rotate_image(img, angle, remove_border=True)
    histogram = add_histogram(rotated)
    cv.imwrite(f"experiments/3/rotated_{angle}.png", histogram)
