import cv2 as cv
import os
import numpy as np
import helper_functions as F

image_path = F.get_image_path("watch.png")
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

nrows, ncols = image.shape
### Blocos devem ter as mesmas dimensoes
if nrows % 4 != 0 or ncols % 4 != 0:
    nrows = (nrows // 4) * 4
    ncols = (ncols // 4) * 4
    cv.resize(image, (ncols, nrows))
    #image = image[:nrows, :ncols]
block_height = nrows // 4
block_width  = ncols // 4

mapping = {
    0: 5,  1: 10,  2: 12,  3: 2,
    4: 7,  5: 15,  6: 0,   7: 8,
    8: 11, 9: 13,  10: 1,  11: 6,
    12: 3, 13: 14, 14: 9,  15: 4,
}

mosaic = np.zeros((nrows, ncols), dtype=np.uint8)
for old_block, new_block in mapping.items():
    old_origin_row, old_origin_col = (block_height * (old_block // 4), block_width * (old_block % 4))
    new_origin_row, new_origin_col = (block_height * (new_block // 4), block_width * (new_block % 4))
    mosaic[new_origin_row:new_origin_row+block_height, new_origin_col:new_origin_col+block_width] = \
        image[old_origin_row:old_origin_row+block_height, old_origin_col:old_origin_col+block_width]

cv.imshow("Original", image)
cv.imshow("Mosaico", mosaic)
cv.waitKey(0)

# TESTE


# Reshape into (4, block_height, 4, block_width)
blocks = image.reshape(4, block_height, 4, block_width).transpose(0, 2, 1, 3)

# Flatten blocks to access them easily
flat_blocks = blocks.reshape(16, block_height, block_width)

# Mapping array
mapping = np.array([
    5, 10, 12, 2,
    7, 15, 0, 8,
    11, 13, 1, 6,
    3, 14, 9, 4
])

# Reorder blocks
reordered_blocks = flat_blocks[mapping]

# Reshape back to 4×4 block grid
mosaic = reordered_blocks.reshape(4, 4, block_height, block_width).transpose(0, 2, 1, 3).reshape(nrows, ncols)

cv.imshow("Original", image)
cv.imshow("Mosaico", mosaic)
cv.waitKey(0)