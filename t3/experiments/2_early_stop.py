import numpy as np
import os
import cv2 as cv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alignment import *
import time

filenames = os.listdir("images")
filenames = [f for f in filenames if f.startswith("pos_") or f.startswith("neg_") or f.startswith("sample") or f.startswith("partitura")]
filenames = ["arial_rotated_81.png"]

times = {}
best = {}
print(f"TESTING RATIO {RATIO} AND WINDOW {WINDOW}")
for img_name in filenames:
    img = cv.imread(os.path.join("images", img_name), cv.IMREAD_GRAYSCALE)
    angles = []
    test = True
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
    for angle in _interleaved_value_range(90):
        rotated = rotate_image(img, angle, remove_border=True)
        objective_val = _projection_objective_function(rotated)
        vals.append(objective_val)

        # Atualiza o melhor angulo se necessario
        if objective_val > best_objective_val:
            best_angle = angle
            best_objective_val = objective_val

        # Early stopping se a funcao objetivo nao melhorar
        if best_angle * angle >= 0 and len(vals) > 2 * WINDOW:
            recent = vals[-2*WINDOW + 1::2] # Apenas angulos com mesmo sinal
            if np.mean(recent) < RATIO * best_objective_val:
                # Ultimo teste: angulo complementar oposto (rotacao de 90 graus)
                last_test = best_angle - 90 if best_angle > 0 else best_angle + 90
                rotated = rotate_image(img, last_test, remove_border=True)
                objective_val = _projection_objective_function(rotated)
                if objective_val > best_objective_val:
                    best_angle = last_test
                break
    times[img_name] = time.time() - t
    best[img_name] = best_angle

    print(f"Image: {img_name} - Best angle: {best_angle:.2f} - Objective Value: {max(vals):.2f}")
    print()

    with open(f"experiments/horizontal_obj/{img_name[:-4]}.txt", "w+") as f:
        for angle, objective_val in zip(angles, vals):
            f.write(f"Angle: {angle:.2f} - Objective Value: {objective_val:.2f}\n")
    #break

with open("experiments/horizontal_obj/times3.txt", "w+") as f:
    f.write("Intercalada early stopping\n")
    for img_name, t in times.items():
        f.write(f"{img_name} - {t:.2f}s - {best[img_name]} deg\n")