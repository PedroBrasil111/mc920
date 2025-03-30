import cv2 as cv
import numpy as np
import helper_functions as F

image_path = F.get_image_path("waterfall.png")
image = cv.imread(image_path, cv.IMREAD_COLOR)

print("Image shape:", image.shape)

gaussian_ksize  = (21, 21)
#gaussian_ksize_list  = [(x, x) for x in range(3, 24, 2)]
gaussian_ksize_list  = [(x, x) for x in range(3, 24, 2)]

gaussian_sigma = 0 # 0.3*((ksize-1)*0.5 - 1) + 0.8
gaussian_sigma_list = np.linspace(0.1, 11.5, 11)

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

image_dict = {}
blur_dict = {}

blur_dict["Original"] = gray_image
image_dict["Original"] = gray_image
for ksize in gaussian_ksize_list:
    title = f"kernel = {ksize[0]}"
    blurred_image = cv.GaussianBlur(gray_image, ksize, gaussian_sigma)
    blur_dict[title] = blurred_image
    result = cv.divide(gray_image, blurred_image, scale=256)
    image_dict[title] = result
    print(f"Kernel size: {ksize[0]} - sigma: {0.3*((ksize[0]-1)*0.5 - 1) + 0.8:.2f}")

F.display_image_grid(blur_dict, (2,6))
F.display_image_grid(image_dict, (2,6))

image_dict = {}
blur_dict = {}

blur_dict["Original"] = gray_image
image_dict["Original"] = gray_image
for sigmaX in gaussian_sigma_list:
    title = f"sigma = {sigmaX:.2f}"
    blurred_image = cv.GaussianBlur(gray_image, gaussian_ksize, sigmaX)
    blur_dict[title] = blurred_image
    result = cv.divide(gray_image, blurred_image, scale=256)
    image_dict[title] = result

F.display_image_grid(blur_dict, (2,6))
F.display_image_grid(image_dict, (2,6))