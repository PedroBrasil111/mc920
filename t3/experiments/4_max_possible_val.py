import numpy as np
import os
import cv2 as cv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alignment import *
import time

for size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 256, 512, 1024, 2048, 4096, 8192]:
    arr = np.zeros((size, size))
    arr[::2] = 255
    arr[1::2] = 0

    projection = _horizontal_projection(arr)
    print(projection)
    maximum = np.sqrt((projection.shape[0] - 1) * ((255*arr.shape[1])**2))
    print(np.sqrt(np.sum(np.square(np.diff(projection)))), maximum)
    #print(projection_objective_function(horizontal_projection(arr)))
