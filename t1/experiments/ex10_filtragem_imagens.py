import cv2 as cv
import os
import numpy as np
import helper_functions as F

image_path = F.get_image_path("waterfall.png")
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

h1 = np.array(
    [[0, 0, -1, 0, 0],
     [0, -1, -2, -1, 0],
     [-1, -2, 16, -2, -1],
     [0, -1, -2, -1, 0],
     [0, 0, -1, 0, 0]], dtype=np.float32
)

h2 = np.array(
    [[1, 4, 6, 4, 1],
     [4, 16, 24, 16, 4],
     [6, 24, 36, 24, 6],
     [4, 16, 24, 16, 4],
     [1, 4, 6, 4, 1]], dtype=np.float32
) / 256

h3 = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], dtype=np.float32
)

h4 = np.array(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]], dtype=np.float32
)

h5 = np.array(
    [[-1, -1, -1],
     [-1, 8, -1],
     [-1, -1, -1]], dtype=np.float32
)

h6 = np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]], dtype=np.float32
) / 9

h7 = np.array(
    [[-1, -1, 2],
     [-1, 2, -1],
     [2, -1, -1]], dtype=np.float32
)

h8 = np.array(
    [[2, -1, -1],
     [-1, 2, -1],
     [-1, -1, 2]], dtype=np.float32
)

h9 = np.identity(9, dtype=np.float32) / 9

h10 = np.array(
    [[-1, -1, -1, -1, -1],
     [-1, 2, 2, 2, -1],
     [-1, 2, 8, 2, -1],
     [-1, 2, 2, 2, -1],
     [-1, -1, -1, -1, -1]], dtype=np.float32
) / 8

h11 = np.array(
    [[-1, -1, 0],
     [-1, 0, 1],
     [0, 1, 1]], dtype=np.float32
)

res = {
    "h1": cv.filter2D(image, -1, h1),
    "h2": cv.filter2D(image, -1, h2),
    "h3": cv.filter2D(image, -1, h3),
    "h4": cv.filter2D(image, -1, h4),
    "h5": cv.filter2D(image, -1, h5),
    "h6": cv.filter2D(image, -1, h6),
    "h7": cv.filter2D(image, -1, h7),
    "h8": cv.filter2D(image, -1, h8),
    "h9": cv.filter2D(image, -1, h9),
    "h10": cv.filter2D(image, -1, h10),
    "h11": cv.filter2D(image, -1, h11),
}

res_combo = np.sqrt(np.power(res["h3"], 2, dtype=np.uint16) + np.power(res["h4"], 2, dtype=np.uint16))
res["combo"] = res_combo.astype(np.uint8)

F.display_image_grid(res, (3, 4))

for key in res.keys():
    res[key] = cv.normalize(res[key], None, 0, 255, cv.NORM_MINMAX)
    res[key] = res[key].astype(np.uint8)    

F.display_image_grid(res, (3, 4))