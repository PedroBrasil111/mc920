import cv2 as cv
import os
import numpy as np

image_path_1 = os.path.join(os.path.dirname(__file__), 'images', 'baboon_monocromatica.png')
image_path_2 = os.path.join(os.path.dirname(__file__), 'images', 'butterfly.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

image_1 = cv.imread(image_path_1, cv.IMREAD_GRAYSCALE)
image_2 = cv.imread(image_path_2, cv.IMREAD_GRAYSCALE)

weights = [
    (0.2, 0.8),
    (0.5, 0.5),
    (0.8, 0.2)
]

for i, (w1, w2) in enumerate(weights):
    result = (w1*image_1 + w2*image_2).astype(np.uint8)
    cv.imwrite(result_path.replace('07', f'07_{i}'), result)
