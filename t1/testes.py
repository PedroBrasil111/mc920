import cv2 as cv
import os

img_path = os.path.join(os.path.dirname(__file__), 'images', 'watch.png')

img = cv.imread(img_path, cv.IMREAD_COLOR)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred_img = cv.GaussianBlur(gray_img, (21, 21), 0)
result = cv.divide(gray_img, blurred_img, gray_img, scale=256)

cv.imshow('Result', result)
cv.waitKey(0)