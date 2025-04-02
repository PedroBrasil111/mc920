import cv2 as cv
import os
import numpy as np
import helper_functions as F

image_path = F.get_image_path("baboon_monocromatica.png")
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

res_dict = {}

alpha_list = [1.5, 2.5, 3.5]
res_dict["Original"] = image
div_image = image / 255 # Normalizacao para [0, 1]
for i, alpha in enumerate(alpha_list):
    result = np.floor(np.power(div_image, 1/alpha) * 255).astype(np.uint8)
    res_dict[f"gamma = {alpha}"] = result

F.display_image_grid(res_dict, (1, 4))

res_dict = {}
round_functions = {
    "Piso": np.floor,
    "Teto": np.ceil,
    "Arredondamento": np.round
}

for i, (round_name, round_function) in enumerate(round_functions.items()):
    result = round_function(np.power(div_image, 1/3.5) * 255).astype(np.uint8)
    res_dict[f"{round_name}"] = result
    cv.imwrite(f"./t1/results_experiments/ajuste_brilho_{round_name}.png", result)

F.display_image_grid(res_dict, (1, 3))

