import cv2 as cv
import numpy as np
from helper_functions import *

def hor_profile(img):
    """
    Retorna o perfil horizontal da imagem `img`.
    """
    return np.sum(255 - img, axis=1)

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Retorna a imagem `img` rotacionada em `angle` graus, ajustando o tamanho para conter toda a imagem.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])

    # Dimensoes da imagem rotacionada
    new_w = int(w * cos + h * sin)
    new_h = int(w * sin + h * cos)
    print(f"Rotacionando imagem de {w}x{h} para {new_w}x{new_h}")

    # Shift por conta do novo tamanho
    rot_mat[0, 2] += (new_w - w) / 2
    rot_mat[1, 2] += (new_h - h) / 2

    # Rotaciona a imagem (gera uma borda branca)
    rotated = cv.warpAffine(img, rot_mat, (new_w, new_h), borderValue=255)

    # Calcula os limites da borda e a remove
    coords = np.column_stack(np.where(rotated < 255))
    if coords.size == 0:
        return rotated
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return rotated#[y0:y1, x0:x1]

def objective_function_diff(perfil: np.ndarray) -> float:
    """
    Retorna o valor da funcao objetivo para o perfil `perfil`.
    Calculado como a raiz da soma dos quadrados das diferenças entre os pixels adjacentes.
    """
    return np.sqrt(np.sum(np.square(np.diff(perfil))))

def horizontal_projection_inclination(img: np.ndarray, ang1: float, ang2: float, step: float) -> tuple:
    """
    Retorna o melhor angulo de rotacao da imagem `img` entre `ang1` e `ang2` com passo `step`.
    """
    if ang1 > ang2 and step > 0:
        ang2, ang1 = ang1, ang2

    best_angle = None
    best_objective_val = -1

    for angle in np.arange(ang1, ang2, step):
        # Rotaciona a imagem e calcula a funcao objetivo
        rotated = rotate_image(img, angle)
        rotated_profile = hor_profile(rotated)
        objective_val = objective_function_diff(rotated_profile)

        # Atualiza o melhor angulo se necessario
        if objective_val > best_objective_val:
            best_objective_val = objective_val
            best_angle = angle

    return best_angle, best_objective_val

def add_histogram(img: np.ndarray) -> np.ndarray:
    """
    Retorna uma imagem com o histograma do perfil horizontal da imagem `img` adicionado a direita.
    """
    profile = hor_profile(img)
    h, w = img.shape[:2]
    min_val, max_val = 0, np.max(profile)

    # Histograma tem 20% da largura da imagem, e divisao tem 5 px
    w_hist = np.floor(w * 0.2).astype(int)
    w_div  = np.floor(5).astype(int)

    # Cria o histograma
    hist = np.full((h, w_hist), 255, dtype=np.uint8)
    for i in range(h):
        idx = int(np.floor((profile[i] - min_val) / (max_val - min_val) * w_hist))
        hist[i, :idx] = 0

    # Adiciona o histograma na imagem original
    reshaped = np.zeros((h, w + w_hist + w_div), dtype=np.uint8)
    reshaped[:, :w] = img
    reshaped[:, w:w + w_div] = 255
    reshaped[:, -w_hist:] = hist

    return reshaped

def hough_transform_inclination(img: np.ndarray) -> float:
    """
    Retorna o melhor angulo de rotacao da imagem `img` entre -90 e 90 graus.
    """
    # Aplica o operador de Canny
    edges = cv.Canny(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX), 255/3, 255)

    # Aplica a transformada de Hough
    lines = cv.HoughLinesWithAccumulator(edges, 1, np.pi/180, 1, min_theta=0, max_theta=np.pi)
    

    #cv.imshow("Canny", edges)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    # Media dos angulos
    # angle = np.mean([180 * theta / np.pi for rho, theta in lines[:, 0]])

    # 
    #accum_percentile = np.percentile(lines[:, :, 2], 99.99)
    #angle = np.mean(lines[lines[:, :, 2] >= accum_percentile][:, 1]) * 180 / np.pi

    # Menor angulo com maior acumulador
    angles = np.where(lines[:, :, 2] == np.max(lines[:, :, 2]), lines[:, :, 1], 361)
    angle = np.min(angles) * 180 / np.pi

    return angle - 90# if angle > 90 else angle + 90

def main():
    for img_path in ["neg_28.png", "neg_4.png", "partitura.png", "pos_24.png", "pos_41.png", "sample1.png",  "sample2.png"]:
        img = cv.imread(f"images/{img_path}", cv.IMREAD_GRAYSCALE)

        #ang, obj = horizontal_projection_inclination(img, -90, 90, 0.1)

        ang = hough_transform_inclination(img)
        print(f"\nImagem: {img_path}")
        print(f"Melhor angulo: {ang:.2f}")
        rotated = rotate_image(img, ang)
        #cv.imshow("Imagem original", img)
        #cv.imshow("Imagem rotacionada", rotated)
        cv.imshow("Imagem com histograma", add_histogram(img))
        cv.imshow("Rotacionada com histograma", add_histogram(rotated))
        cv.imwrite("results/" + img_path, add_histogram(rotated))
        cv.waitKey(0)
        cv.destroyAllWindows()

def test():
    img = cv.imread("images/neg_28.png", cv.IMREAD_GRAYSCALE)
    for angle in range(0, 91):
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])

        # Dimensoes da imagem rotacionada
        new_w = int(w * cos + h * sin)
        new_h = int(w * sin + h * cos)
        print(f"Angulo - {angle} - Rotacionando imagem de {w}x{h} para {new_w}x{new_h}")


if __name__ == "__main__":
    test()

    main()