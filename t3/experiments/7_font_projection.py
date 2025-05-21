import numpy as np
import os
import cv2 as cv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alignment import *
import time

img = cv.imread("images/andale_mono_rotated.png", cv.IMREAD_GRAYSCALE)
rot1 = rotate_image(img, 17, remove_border=True)
rot2 = rotate_image(img, -73, remove_border=True)
res1 = add_histogram(rot1)
res2 = add_histogram(rot2)

res0 = rotate_image(img, 0, remove_border=True)
res0 = add_histogram(res0)

cv.imwrite("experiments/andale_mono/andale_mono.png", res0, [cv.IMWRITE_PNG_STRATEGY, cv.IMWRITE_PNG_STRATEGY_DEFAULT])
cv.imwrite("experiments/andale_mono/andale_mono_rotated_pos17.png", res1, [cv.IMWRITE_PNG_STRATEGY, cv.IMWRITE_PNG_STRATEGY_DEFAULT])
cv.imwrite("experiments/andale_mono/andale_mono_rotated_neg73.png", res2, [cv.IMWRITE_PNG_STRATEGY, cv.IMWRITE_PNG_STRATEGY_DEFAULT])