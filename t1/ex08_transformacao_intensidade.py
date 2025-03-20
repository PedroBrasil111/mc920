import cv2 as cv
import os
import numpy as np

image_path = os.path.join(os.path.dirname(__file__), 'images', 'city.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

transformations = {
    "negative": 255 - image,
    "intensity": cv.normalize(image, None, 100, 200, cv.NORM_MINMAX),
    "inverted_even": image[1::2] + image[::2][::-1],
    ""
}

for i, (w1, w2) in enumerate(weights):
    result = (w1*image_1 + w2*image_2).astype(np.uint8)
    cv.imwrite(result_path.replace('07', f'07_{i}'), result)
