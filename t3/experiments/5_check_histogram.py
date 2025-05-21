import numpy as np
import os
import cv2 as cv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alignment import *
import time

img = cv.imread("images/arial.png", cv.IMREAD_GRAYSCALE)
angle = detect_rotation_angle_projection(img)
rotated = rotate_image(img, angle, remove_border=False)
histogram = add_histogram(rotated)
cv.imshow("histogram", histogram)
cv.waitKey(0)
cv.destroyAllWindows()