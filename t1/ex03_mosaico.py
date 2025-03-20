import cv2 as cv
import os
import numpy as np

mapping = {
    1: 6,  2: 11,  3: 13,  4: 3,
    5: 8,  6: 16,  7: 1,   8: 9,
    9: 12, 10: 14, 11: 2,  12: 7,
    13: 4, 14: 15, 15: 10, 16: 5,
}

image_path = os.path.join(os.path.dirname(__file__), 'images', 'baboon_monocromatica.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
nrows, ncols = image.shape

### Blocos devem ter as mesmas dimensoes
if nrows % 4 != 0 or ncols % 4 != 0:
    nrows = nrows // 4 * 4
    ncols = ncols // 4 * 4
    np.reshape(image, (nrows, ncols))

block_height = nrows // 4
block_width = ncols // 4

blocks = np.reshape(np.arange(0, nrows*ncols), ())
mapping_func = np.vectorize(lambda x: mapping[])
mapping_func(blocks)

print(blocks[0, 0])
