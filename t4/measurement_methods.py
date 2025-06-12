import matplotlib
matplotlib.use('Agg') # Fix para erro de backend do matplotlib

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *

def area_histogram(areas: np.ndarray) -> tuple[list[int], np.ndarray]:
    """
    Retorna o histograma e de areas dos objetos e o plot correspondente.
    """
    # Define os bins
    mi, ma = np.min(areas), np.max(areas)
    bin_edges = [mi if mi < 1500 else 0, 1500, 3000, ma if ma > 3000 else 4500]

    # Calcula e desenha o histograma
    fig, ax = plt.subplots(figsize=(5, 4))
    hist = ax.hist(areas, bins=bin_edges, color="blue", edgecolor="black")
    ax.set_xlabel("Área")
    ax.set_ylabel("Número de Objetos")
    ax.set_xlim(0, 4500)
    xticklabels = [str(x) for x in bin_edges]

    # Adiciona "..." se limite do ultimo bin for maior que 4500
    if ma > 4500:
        bin_edges = bin_edges[:-1] + [7500/2, 4500]
        xticklabels = xticklabels[:-1] + ["...", str(ma)]
    ax.set_xticks(bin_edges)
    ax.set_xticklabels(xticklabels)

    # Converte para imagem
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    img = cv.cvtColor(buf, cv.COLOR_RGBA2BGR)
    plt.close(fig)

    return [int(val) for val in hist[0]], img

def build_mask(img: np.ndarray) -> np.ndarray:
    """
    Dada uma imagem RGB com fundo branco,
    Retorna uma mascara binaria onde os objetos sao representados por 1
    """
    return np.all(img == 255, axis=2).astype(np.uint8)

def find_borders(bin_obj: np.ndarray) -> np.ndarray:
    """
    Dada uma imagem binaria onde os objetos sao representados por 1,
    retorna uma mascara binaria onde as bordas dos objetos sao representadas por 1
    """
    # Calculo das bordas usando convolucao
    # Um ponto de borda tem menos de 4 vizinhos em vizinhanca-4
    neighbor_kernel = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=np.uint8)
    neighbor_count = cv.filter2D(bin_obj, -1, neighbor_kernel, borderType=cv.BORDER_REPLICATE)
    bin_border = np.where((bin_obj == 1) & (neighbor_count < 4), 0, 1).astype(np.uint8)
    return bin_border

def find_contour(label_mask: np.ndarray, label: int) -> np.ndarray:
    """
    Dada uma mascara de labels e um label especifico,
    encontra o contorno do objeto correspondente ao label.
    """
    component_mask = (label_mask == label).astype(np.uint8)
    contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return contours[0]

def find_convex_hull(contour: np.ndarray, img_shape: tuple) -> tuple[np.ndarray, float]:
    """
    Dado o contorno de um objeto e a forma da imagem,
    retorna os pontos do fecho convexo e sua area.
    """
    hull = cv.convexHull(contour) # Encontra o fecho convexo

    # Cria uma mascara para calcular area do fecho convexo
    hull_mask = np.zeros(img_shape, dtype=np.uint8)
    cv.drawContours(hull_mask, [hull], -1, 1, thickness=cv.FILLED)
    area = np.sum(hull_mask)

    return hull, area

def calculate_area(label_mask: np.ndarray, label: int) -> float:
    """
    Dada uma mascara de labels e um label especifico,
    calcula a area do objeto correspondente ao label.
    """
    return np.sum(label_mask == label)

def calculate_perimeter(contour):
    """
    Dado um contorno, calcula o perimetro do objeto.
    """
    return cv.arcLength(contour, closed=True)

def _line_inside_component(p1: np.ndarray, p2: np.ndarray, label_mask: np.ndarray, label: int) -> bool:
    """
    Verifica se a reta entre os pontos p1 e p2 esta completamente dentro do objeto
    representado pelo label na mascara label_mask.
    """
    line_img = np.zeros_like(label_mask, dtype=np.uint8)
    # Desenha a reta entre p1 e p2 na imagem, e verifica se todos os pixels estao dentro do objeto
    draw_line(line_img, p1, p2, 255)
    return np.all(label_mask[line_img == 255] == label)

def _find_major_axis(
        contour: np.ndarray,
        label_mask: np.ndarray,
        label: int
    ) -> tuple[tuple, float]:
    """
    Dada uma mascara de labels, o contorno de um objeto e um label especifico,
    encontra o eixo maior do objeto correspondente ao label.
    Retorna os pontos do eixo maior e seu comprimento.
    """
    # Inicializacao das variavies
    major = (None, None)
    max_dist = 0

    # Verifica todos os pares de pontos do contorno
    # para achar o eixo maior (maior segmento interno)
    for i in range(len(contour) - 1):
        p1 = contour[i][0]

        contour_pts = contour[i+1:, 0, :]
        dists = np.linalg.norm(contour_pts - p1, axis=1)

        # Checa apenas os pontos que podem ter distancia maior
        # comecando pela maior distancia
        contour_pts = contour_pts[dists > max_dist]
        dists = dists[dists > max_dist]
        sorted_indices = np.argsort(-dists)
        for idx in sorted_indices:
            p2, dist = contour_pts[idx], dists[idx]
            if _line_inside_component(p1, p2, label_mask, label):
                max_dist = dist
                major = (p1, p2)
                break

    return major, max_dist

def _find_minor_axis(
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

    return minor, max_dist

def calculate_axis_eccentricity(
        contour: np.ndarray,
        label_mask: np.ndarray,
        label: int
    ) -> tuple[tuple, tuple, float]:
    """
    Dada uma mascara de labels, o contorno de um objeto e um label especifico,
    encontra os eixos maior e menor do objeto correspondente ao label.
    Retorna os pontos do eixo maior (a), os pontos do eixo menor (b) e a excentricidade,
    definida como e = sqrt(1 - (b^2 / a^2)),
    """
    major, major_axis_len = _find_major_axis(contour, label_mask, label)
    minor, minor_axis_len = _find_minor_axis(label_mask, major, label)
    return major, minor, np.sqrt(1 - minor_axis_len**2 / major_axis_len**2)

def calculate_ellipse_eccentricity(contour: np.ndarray) -> tuple[tuple, float]:
    """
    Dado um contorno, encontra a elipse ajustada e calcula sua excentricidade.
    Retorna a elipse e a excentricidade.
    A excentricidade e definida como e = sqrt(1 - (b^2 / a^2)),
    O contorno deve ter pelo menos 5 pontos para que a elipse possa ser ajustada.
    """
    if len(contour) < 5:
        return None, np.nan # minimo de 5 pontos

    # Fita a elipse e obtem os eixos menor e maior
    ellipse = cv.fitEllipse(contour)
    (major_axis, minor_axis) = ellipse[1]
    a, b = max(major_axis, minor_axis), min(major_axis, minor_axis)
    eccentricity = np.sqrt(1 - (b ** 2) / (a ** 2)) if a != 0 else np.inf
    return ellipse, eccentricity

def draw_line(img: np.ndarray, p1: np.ndarray, p2: np.ndarray, color: tuple):
    """
    Desenha uma linha entre os pontos p1 e p2 na imagem img com a cor especificada.
    A linha eh desenhada com espessura 1.
    """
    cv.line(img, tuple(p1), tuple(p2), color, thickness=1)

def draw_label_with_color(img: np.ndarray, labels: np.ndarray, label: int, color: tuple):
    """
    Dada uma imagem de labels, um label especifico e uma cor,
    colore a regiao correspondente ao label na imagem e escreve o numero do label no centro do objeto.
    """
    # Colore a regiao
    img[labels == label] = color

    # Adiciona texto no centro de massa do objeto
    y_coords, x_coords = np.where(labels == label)
    y_center, x_center = int(np.mean(y_coords)), int(np.mean(x_coords))
    text = str(label)
    font, font_scale, thickness = cv.FONT_HERSHEY_PLAIN, 1, 1
    (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, thickness)
    x_text = x_center - text_width // 2
    y_text = y_center + text_height // 2
    cv.putText(img, text, (x_text, y_text), font, font_scale, (0, 0, 0), thickness, cv.LINE_8)

def draw_contour(img: np.ndarray, contour: np.ndarray, color: tuple):
    """
    Desenha o contorno de um objeto na imagem img com a cor especificada.
    """
    cv.drawContours(img, [contour], -1, color, 1)