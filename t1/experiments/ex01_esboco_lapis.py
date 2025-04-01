import cv2 as cv
import numpy as np
import helper_functions as F

image_path = F.get_image_path("waves.png")
image = cv.imread(image_path, cv.IMREAD_COLOR)
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

print("Image shape:", image.shape)

# gaussian_ksize  = (21, 21)
# #gaussian_ksize_list  = [(x, x) for x in range(3, 24, 2)]
# gaussian_ksize_list  = [(x, x) for x in range(3, 100, 2)][:5]

# gaussian_sigma = 0 # 0.3*((ksize-1)*0.5 - 1) + 0.8
# gaussian_sigma_list = np.linspace(0.1, 5, 6)


# image_dict = {}
# blur_dict = {}

# blur_dict["Original"] = gray_image
# image_dict["Original"] = gray_image
# for ksize in gaussian_ksize_list:
#     title = f"kernel = {ksize[0]}"
#     blurred_image = cv.GaussianBlur(gray_image, ksize, gaussian_sigma)
#     blur_dict[title] = blurred_image
#     result = cv.divide(gray_image, blurred_image, scale=256)
#     image_dict[title] = result
#     print(f"Kernel size: {ksize[0]} - sigma: {0.3*((ksize[0]-1)*0.5 - 1) + 0.8:.2f}")

# image_grid_blur = F.display_image_grid(blur_dict, (2,3))
# image_grid_res  = F.display_image_grid(image_dict, (2,3))

# cv.imwrite("./t1/results_experiments/grid_blur_ksize.png", image_grid_blur)
# cv.imwrite("./t1/results_experiments/grid_res_ksize.png", image_grid_res)

# image_dict = {}
# blur_dict = {}

# #blur_dict["Original"] = gray_image
# #image_dict["Original"] = gray_image
# for sigmaX in gaussian_sigma_list:
#     title = f"sigma = {sigmaX:.2f}"
#     blurred_image = cv.GaussianBlur(gray_image, gaussian_ksize, sigmaX)
#     blur_dict[title] = blurred_image
#     result = cv.divide(gray_image, blurred_image, scale=256)
#     image_dict[title] = result

# image_grid_blur = F.display_image_grid(blur_dict, (2,3))
# image_grid_res  = F.display_image_grid(image_dict, (2,3))

# cv.imwrite("./t1/results_experiments/grid_blur_sigma.png", image_grid_blur)
# cv.imwrite("./t1/results_experiments/grid_res_sigma.png", image_grid_res)

# sigmaX = 5
# sigmaY = 10
# ksize = (23,23)
# blurred_image = cv.GaussianBlur(gray_image, ksize, sigmaX=sigmaX, sigmaY=sigmaY)
# result = cv.divide(gray_image, blurred_image, scale=256)
# cv.imshow("Result1", result)

# sigmaX = 5
# ksize = (23,23)
# blurred_image = cv.GaussianBlur(gray_image, ksize, sigmaX=sigmaX)
# result = cv.divide(gray_image, blurred_image, scale=256)
# cv.imshow("Result", result)
# cv.waitKey(0)

# cv.waitKey(0)

step = np.min(gray_image.shape) // 10
print(np.min(gray_image.shape))
step = step if step % 2 == 0 else step + 1
gaussian_ksize_list  = [(43,43), (101,101)]
gaussian_sigma_list = [np.sqrt(ksize[0] - 1)/2 for ksize in gaussian_ksize_list]

blur_dict = {}
image_dict = {}
blur_dict["Original"] = gray_image
image_dict["Original"] = gray_image
for i, (sigma, ksize) in enumerate(zip(gaussian_sigma_list, gaussian_ksize_list)):
    title = f"Kernel = {ksize[0]} | Sigma = {sigma:.2f}"
    blurred_image = cv.GaussianBlur(gray_image, ksize, 0)
    result = cv.divide(gray_image, blurred_image, scale=256)
    blur_dict[title] = blurred_image
    image_dict[title] = result

image_grid_blur = F.display_image_grid(blur_dict, (1,3))
image_grid_res  = F.display_image_grid(image_dict, (1,3))

cv.imwrite("./t1/results_experiments/grid_end.png", image_grid_res)


# ksize = (21,21)
# sigma1 = 0.3*((21-1)*0.5 - 1) + 0.8
# sigma2 = 0
# image1 = cv.GaussianBlur(gray_image, ksize, sigma1)
# image2 = cv.GaussianBlur(gray_image, ksize, sigma2)
# result1 = cv.divide(gray_image, image1, scale=256)
# result2 = cv.divide(gray_image, image2, scale=256)
# print(np.sum(~(result1 == result2)))