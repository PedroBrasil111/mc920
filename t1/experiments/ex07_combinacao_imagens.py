import cv2 as cv
import os
import numpy as np
import helper_functions as F

image_path_1 = F.get_image_path("baboon_monocromatica.png")
image_path_2 = F.get_image_path("butterfly.png")

image_1 = cv.imread(image_path_1, cv.IMREAD_GRAYSCALE)
image_2 = cv.imread(image_path_2, cv.IMREAD_GRAYSCALE)

image_2 = cv.resize(image_2, (512, 512))

weights = [
    (0.2, 0.8),
    (0.5, 0.5),
    (0.8, 0.2)
]

for i, (w1, w2) in enumerate(weights):
    #result = (w1*image_1 + w2*image_2).astype(np.uint8)
    result = np.add(np.multiply(w1, image_1), np.multiply(w2, image_2)).astype(np.uint8)    
    cv.imshow(f"Combinacao {i+1}", result)
    cv.waitKey(0)
