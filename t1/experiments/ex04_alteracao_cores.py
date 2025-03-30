import cv2 as cv
import os
import numpy as np

image_path = os.path.join(os.path.dirname(__file__), 'images', 'watch.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

kernel = np.array(
    [[0.393, 0.769, 0.189],
     [0.349, 0.686, 0.168],
     [0.272, 0.534, 0.131]], dtype=np.float32
)

image = cv.imread(image_path, cv.IMREAD_COLOR)

for r in range(len(image)):
    for c in range(len(image[0])):
        image[r][c] = np.clip(np.dot(kernel, image[r][c].T), 0, 255)

cv.imshow('Result', image)
cv.waitKey(0)
