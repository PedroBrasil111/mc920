import cv2 as cv
import os
import numpy as np

image_path = os.path.join(os.path.dirname(__file__), 'images', 'baboon_monocromatica.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

bit_list = [0, 4, 7]
for i, bit in enumerate(bit_list):
    result = image & (0b1 << bit)
    result = cv.normalize(result, -1, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    cv.imwrite(result_path.replace('06', f'06_plano{bit}'), result)
    # cv.imshow('Result', result)
    # cv.waitKey(0)
