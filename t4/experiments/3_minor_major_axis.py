import importlib
from .1_initial1 import *
import numpy as np
import cv2 as cv

THRESHOLD = 0.5
THRESHOLD_LIST = [
    1,
    0.5,
    0.1,
    0.05,
    0.001,
]

def calculate_eccentricity2(contour, label_mask, label, thresh):
    if len(contour) < 2:
        return 0

    # --- Find major axis ---
    major = (None, None)
    max_dist = 0
    for i in range(len(contour)):
        p1 = contour[i][0]
        for j in range(i + 1, len(contour)):
            p2 = contour[j][0]
            if not line_inside_component(p1, p2, label_mask, label):
                continue
            dist = np.linalg.norm(p1 - p2)
            if dist > max_dist:
                max_dist = dist
                major = (p1, p2)

    if major[0] is None:
        return None, None, float("inf")

    p1, p2 = major
    major_vec = p2 - p1
    major_vec_normalized = major_vec / np.linalg.norm(major_vec)
    major_axis_len = np.linalg.norm(major_vec)

    # --- Find minor axis (perpendicular to major) ---
    minor = (None, None)
    minor_axis_len = 0
    min_dot_product = float("inf")
    acutal_min_dot_product = float("inf")
    for i in range(len(contour)):
        pi = contour[i][0]
        for j in range(i + 1, len(contour)):
            pj = contour[j][0]
            vec = pj - pi
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            vec_normalized = vec / vec_norm
            dot_product = np.abs(np.dot(major_vec_normalized, vec_normalized))

            if dot_product < acutal_min_dot_product:
                acutal_min_dot_product = dot_product

            if dot_product < thresh:
                if not line_inside_component(pi, pj, label_mask, label):
                    continue
                if vec_norm > minor_axis_len:
                    minor_axis_len = vec_norm
                    minor = (pi, pj)
                    min_dot_product = dot_product

    if minor_axis_len == 0:
        return major, None, float("inf")

    return major, minor, major_axis_len / minor_axis_len

def main():
    rgb_img = cv.imread("images/objetos3.png")

    # Mascara binaria onde os objetos tem valor 1 e o fundo 0
    mask = build_mask(rgb_img)
    bin_img = (1 - mask) * 255
    # Bordas
    bin_border = borders(mask)
    border_img = bin_border * 255
    # Encontra componentes (fundo: label 0)
    num_labels, labels = cv.connectedComponents(mask, connectivity=4)

    for thresh in THRESHOLD_LIST:
        rgb_img2 = rgb_img.copy()
        for i in range(1, num_labels):
            # Contorno e fecho convexo do objeto
            contour = find_contour(labels, i)

            # Calcula as propriedades
            major2, minor2, e3 = calculate_eccentricity2(contour, labels, i, thresh)

            # Desenha as informacoes nas imagens
            draw_line(rgb_img2, major2[0], major2[1], (0, 0, 0))
            if minor2:
                draw_line(rgb_img2, minor2[0], minor2[1], (0, 0, 0))
        cv.imwrite(f"experiments/images/major_minor_axis_{thresh}.png", rgb_img2)
    return
    w, h = rgb_img.shape[1], rgb_img.shape[0]
    screen_w, screen_h = 1440, 1080
    offset_x, offset_y = 50, 50
    columns = screen_w // (w + offset_x) - 1
    for i, (title, image) in enumerate(windows.items()):
        cv.imshow(title, image)
        x = (i % columns) * (w + offset_x)
        if i < columns:
            y = 0
        else:
            y = ((i // columns) % (screen_h - (h + offset_y))) * (h + offset_y)
        cv.moveWindow(title, x, y)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()