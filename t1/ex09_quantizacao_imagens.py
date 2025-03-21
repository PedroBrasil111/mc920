import cv2 as cv
import os
import numpy as np

image_path = os.path.join(os.path.dirname(__file__), 'images', 'baboon_monocromatica.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

quantization_levels = [255, 63, 31, 15, 7, 3, 1]
for i, quantization_level in enumerate(quantization_levels):
    result = cv.normalize(image, None, 0, quantization_level, cv.NORM_MINMAX).astype(np.uint8)
    result = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    #cv.imwrite(result_path.replace("09", f"09_{quantization_level + 1}niveis"), result)
    cv.imshow(f"{quantization_level + 1}", result)
cv.waitKey(0)
