import cv2 as cv
import os

image_path  = os.path.join(os.path.dirname(__file__), 'images', 'watch.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

image = cv.imread(image_path, cv.IMREAD_COLOR)
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred_image = cv.GaussianBlur(gray_image, (21, 21), 0)
result = cv.divide(gray_image, blurred_image, gray_image, scale=256)

cv.imwrite(result_path, result)