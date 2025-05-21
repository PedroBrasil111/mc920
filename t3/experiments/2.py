import numpy as np
import os
import cv2 as cv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alignment_methods import *

filenames = os.listdir("images")

for img_name in filenames:
    img = cv.imread(os.path.join("images", img_name), cv.IMREAD_GRAYSCALE)
    with open(f"experiments/horizontal_obj/{img_name[:-4]}.txt", "w+") as f:
        for angle in np.arange(-90, 91, 1):
            # Rotaciona a imagem e calcula a funcao objetivo
            rotated = rotate_image(img, angle)
            rotated_profile = hor_profile(rotated)
            objective_val = profile_objective_function(rotated_profile)

            # Escreve o angulo e o valor da funcao objetivo no arquivo
            f.write(f"Angle: {angle:.2f} - Objective Value: {objective_val:.2f}\n")
