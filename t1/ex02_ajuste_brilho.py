import cv2 as cv
import os
import numpy as np

image_path  = os.path.join(os.path.dirname(__file__), 'images', 'baboon_monocromatica.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

alpha_list = [1.5, 2.5, 3.5]
image = image / 255 # Normalizacao para [0, 1]
for i, alpha in enumerate(alpha_list):
    result = np.floor(np.power(image, 1/alpha) * 255).astype(np.uint8)
    cv.imwrite(result_path.replace('02', f'02_{i}'), result)
