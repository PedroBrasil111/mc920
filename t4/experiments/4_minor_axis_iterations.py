from .1_initial1 import *
import numpy as np
import cv2 as cv

def find_minor_axis(
        label_mask: np.ndarray,
        major_axis: tuple,
        label: int,
        step: float = 1.0
    ) -> tuple[tuple, float]:
    """
    Dada uma mascara de labels, o eixo maior e um label especifico,
    encontra o eixo menor do objeto correspondente ao label.
    Retorna os pontos do eixo menor e seu comprimento.
    """
    height, width = label_mask.shape

    minor_list = []

    # Vetor de suporte sobre o eixo maior
    p1, p2 = major_axis
    major_length = np.linalg.norm(p2 - p1)
    unit_vec = (p2 - p1) / major_length # Vetor unitario apontando de p1 para p2
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])
    cross_vector = perp_vec * major_length / 2  # Vetor perpendicular ao eixo maior e com metade do tamanho

    # Inicializacao das variaveis
    max_dist = 0
    minor = (None, None)
    prev_base = None

    # Itera sobre os pontos da reta com passo step
    for i in np.arange(0, major_length + step, step):
        # Ponto sobre o eixo maior a partir do qual a reta perpendicular sera checada
        line_origin = (p1 + i * unit_vec).astype(int)
        if np.all(prev_base == line_origin):
            continue # Evita calculo desnecessario se ponto ja tiver sido checado
        prev_base = line_origin

        # Calcula os pontos extremos da reta candidata
        pts = (line_origin - cross_vector, line_origin + cross_vector)
        pts = np.clip(pts, [0, 0], [width - 1, height - 1]).astype(np.uint32)

        # Imagem com 1 nas posicoes da reta candidata
        line_mask = np.zeros_like(label_mask, dtype=np.uint8)
        draw_line(line_mask, pts[0], pts[1], color=1)

        # Apenas os pontos da reta dentro do objeto
        pts_inside = np.nonzero(line_mask & (label_mask == label))
        if len(pts_inside[0]) < 2:
            continue # Em pontos de borda, pode ocorrer de nao ter overlap

        # Primeiro e ultimo pontos dentro do objeto
        idx1, idx2 = np.argmax(pts_inside[0]), np.argmin(pts_inside[0])
        if pts_inside[0][idx1] == pts_inside[0][idx2]: # Se reta for vertical
            idx1, idx2 = np.argmax(pts_inside[1]), np.argmin(pts_inside[1])
        m1 = np.array([pts_inside[1][idx1], pts_inside[0][idx1]])
        m2 = np.array([pts_inside[1][idx2], pts_inside[0][idx2]])

        # Atualiza se for maior
        dist = np.linalg.norm(m1 - m2)
        if dist > max_dist:
            max_dist = dist
            minor = (m1, m2)
            minor_list.append(pts)

    return minor, max_dist, minor_list


img = cv.imread("images/ellipse.png", cv.IMREAD_COLOR)

mask = build_mask(img)
bin_img = (1 - mask) * 255
num_labels, labels = cv.connectedComponents(mask, connectivity=4)
labels -= 1
num_labels -= 1

for label in range(num_labels):
    contour = find_contour(labels, label)
    hull, hull_area = find_convex_hull(contour, img.shape[:2])
    major, minor, e1 = calculate_axis_eccentricity(contour, labels, label)
    _, _, minor_list = find_minor_axis(labels, major, label, step=3)
    
    draw_line(img, major[0], major[1], (0, 0, 0))
    for minor in minor_list:
        draw_line(img, minor[0], minor[1], (0, 0, 255))

cv.imshow("Major and Minor Axes", img)
import os
os.makedirs("experiments/images", exist_ok=True)
cv.imwrite("experiments/images/major_minor_axes.png", img)
cv.waitKey(0)
