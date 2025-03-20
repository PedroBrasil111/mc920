import cv2 as cv
import os
import numpy as np

image_path = os.path.join(os.path.dirname(__file__), 'images', 'baboon_monocromatica.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

mapping = {
    0: 5,  1: 10,  2: 12,  3: 2,
    4: 7,  5: 15,  6: 0,   7: 8,
    8: 11, 9: 13,  10: 1,  11: 6,
    12: 3, 13: 14, 14: 9,  15: 4,
}

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
nrows, ncols = image.shape

### Blocos devem ter as mesmas dimensoes
if nrows % 4 != 0 or ncols % 4 != 0:
    nrows = nrows // 4 * 4
    ncols = ncols // 4 * 4
    np.reshape(image, (nrows, ncols))

block_height = nrows // 4
block_width  = ncols // 4

def block_pixels(block: int, block_height: int, block_width: int) -> tuple[int]:
    origin_row = (block // 4) * block_height
    origin_col = (block % 4) * block_width
    end_row = origin_row + block_height
    end_col = origin_col + block_width
    return (origin_row, origin_col, end_row, end_col)

mosaic = np.zeros((nrows, ncols), dtype=np.uint8)
for original_block, new_block in mapping.items():
    original_origin_row, original_origin_col, original_end_row, original_end_col = block_pixels(original_block, block_height, block_width)
    new_origin_row, new_origin_col, new_end_row, new_end_col = block_pixels(new_block, block_height, block_width)
    mosaic[original_origin_row:original_end_row, original_origin_col:original_end_col] = \
        image[new_origin_row:new_end_row, new_origin_col:new_end_col]

cv.imwrite(result_path, mosaic)
