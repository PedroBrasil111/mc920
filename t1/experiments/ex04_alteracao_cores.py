import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as F

image_path = F.get_image_path("watch.png")
image = cv.imread(image_path, cv.IMREAD_COLOR)

kernel = np.array(
    [[0.393, 0.769, 0.189],
     [0.349, 0.686, 0.168],
     [0.272, 0.534, 0.131]], dtype=np.float32
)

transf = np.dot(image, kernel)

#transf = np.dot(image, kernel.T)
result = cv.normalize(transf, None, 0, 255, cv.NORM_MINMAX)
result = result.astype(np.uint8)
cv.cvtColor(result, cv.COLOR_RGB2BGR, dst=result)

result2 = np.clip(transf, 0, 255)
result2 = result2.astype(np.uint8)
cv.cvtColor(result2, cv.COLOR_RGB2BGR, dst=result2)

F.display_image_grid({
    "Original": image,
    "Transformed": result,
    "Clipped": result2
}, (1, 3))

cv.imwrite(os.path.join("t1/results/transformed_image_1.png"), result)
cv.imwrite(os.path.join("t1/results/transformed_image_2.png"), result2)

cv.waitKey(0)
