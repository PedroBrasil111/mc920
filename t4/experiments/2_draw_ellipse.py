import cv2 as cv
import numpy as np 

ellipse_img = np.full((256, 256, 3), 255, dtype=np.uint8)
a = 100
b = 50

center = (ellipse_img.shape[1] // 2, ellipse_img.shape[0] // 2)

cv.ellipse(ellipse_img, center, (a, b), 0, 0,  360, (255, 0, 0), -1)

cv.imshow("Ellipse", ellipse_img)
cv.imwrite("images/ellipse.png", ellipse_img)
cv.waitKey(0)

