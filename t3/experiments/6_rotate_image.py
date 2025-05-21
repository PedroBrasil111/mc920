import numpy as np
import os
import cv2 as cv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alignment import *
import time

img = cv.imread("images/arial.png", cv.IMREAD_GRAYSCALE)
rot = rotate_image(img, 173, remove_border=False)
cv.imwrite("images/arial_rotated_173.png", rot, [cv.IMWRITE_PNG_STRATEGY, cv.IMWRITE_PNG_STRATEGY_DEFAULT])