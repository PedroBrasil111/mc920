import numpy as np
import os
import cv2 as cv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from t3.alignment import *
import time

filenames = os.listdir("images")

times = {}
for img_name in filenames:
    img = cv.imread(os.path.join("images", img_name), cv.IMREAD_GRAYSCALE)
    angles = []
    vals = []
    best_vals = []
    best_angle = None
    best_objective_val = -1

    angle_vals = np.empty(181, dtype=int)
    angle_vals[0] = 0
    angle_vals[2::2] = np.arange(1, 91)
    angle_vals[1::2] = -angle_vals[2::2]

    angle_vals = np.arange(-90, 91, 1)

    t = time.time()
    for angle in angle_vals:
        # Rotaciona a imagem e calcula a funcao objetivo
        rotated = rotate_image(img, angle)
        rotated_profile = hor_profile(rotated)
        objective_val = profile_objective_function(rotated_profile)

        # Atualiza o melhor angulo se necessario
        angles.append(angle)
        vals.append(objective_val)

        if objective_val > best_objective_val:
            best_objective_val = objective_val
            best_vals.append(objective_val)

            # Early stopping se a funcao objetivo nao melhorar
            if best_vals and np.average(best_vals[-10:]) < best_vals[-1]/2:
                break

    times[img_name] = time.time() - t

    print(f"Image: {img_name} - Best angle: {angles[np.argmax(vals)]:.2f} - Objective Value: {max(vals):.2f}")

    with open(f"experiments/horizontal_obj/{img_name[:-4]}.txt", "w+") as f:
        for angle, objective_val in zip(angles, vals):
            f.write(f"Angle: {angle:.2f} - Objective Value: {objective_val:.2f}\n")

with open("experiments/horizontal_obj/times4.txt", "w+") as f:
    for img_name, t in times.items():
        f.write(f"{img_name} - {t:.2f}\n")