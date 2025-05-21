import cv2 as cv
import numpy as np

### Funcoes para tecnica de projecao horizontal

def _horizontal_projection(img):
    """
    Retorna a projecao horizontal da imagem `img`.
    """
    return np.sum(255 - img, axis=1)

def _projection_objective_function(img: np.ndarray) -> float:
    """
    Retorna o valor da funcao objetivo para a projecao `projection`.
    Calculado como o MSE, onde o erro eh a diferenca de intensidade entre pontos adjacentes.
    """ 
    projection = _horizontal_projection(img)
    return np.sum(np.square(np.diff(projection))) / (img.shape[0] * img.shape[1])

def _interleaved_value_range(max_val: float, step: float=1) -> np.ndarray:
    """
    Retorna um array de valores intercalados entre -max_val e max_val, com passo `step`.
    """
    val_range = np.arange(step, max_val + step, step)
    interleaved = np.empty(2*val_range.shape[0] + 1)
    interleaved[0] = 0
    interleaved[2::2] = val_range
    interleaved[1::2] = -val_range
    return interleaved

WINDOW = 5
RATIO = 0.1

def detect_rotation_angle_projection(img: np.ndarray) -> np.ndarray:
    """
    Retorna o melhor angulo de rotacao da imagem `img`
    """
    # Inicializacao de variaveis
    best_angle = None
    best_objective_val = -1
    vals = []

    # Rotaciona a imagem para cada angulo e calcula a funcao objetivo
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

    return best_angle

### Funcoes para tecnica de Hough

def detect_rotation_angle_hough(img: np.ndarray) -> float:
    """
    Retorna o melhor angulo de rotacao da imagem `img` entre -90 e 90 graus.
    """
    # Aplica o operador de Canny
    edges = cv.Canny(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX), 255//3, 255)

    # Aplica a transformada de Hough com rho de 0 a 180 graus
    lines = cv.HoughLinesWithAccumulator(edges, 1, np.pi/180, 1, min_theta=0, max_theta=np.pi)

    # Angulo com maior acumulador
    max_accum = np.max(lines[:, :, 2])
    angles = lines[lines[:, :, 2] == max_accum][:, 1]
    angle = angles[0] * 180 / np.pi

    return angle - 90

### Funcao principal de deteccao de angulo

_ANGLE_DETECTION_FUNCT = {
    "hough": detect_rotation_angle_hough,
    "projection": detect_rotation_angle_projection
}

def run_angle_detection(image: np.ndarray, mode: str) -> float:
    """
    Executa a deteccao de angulo de rotacao da imagem `image` usando o metodo `mode`.
    O modo pode ser 'hough' ou 'projection'.
    """
    return _ANGLE_DETECTION_FUNCT.get(
        mode, detect_rotation_angle_hough # default
    )(image)

### Funcoes de visualizacao / rotacao

def _remove_white_border(img: np.ndarray) -> np.ndarray:
    """
    Remove a borda branca da imagem `img`.
    """
    coords = np.column_stack(np.where(img < 255))
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img[y0:y1, x0:x1]

def rotate_image(
        img: np.ndarray, angle: float,
        remove_border: bool=False
        ) -> np.ndarray:
    """
    Retorna a imagem `img` rotacionada em `angle` graus, ajustando o tamanho para conter toda a imagem.
    Se `remove_border` for True, remove a borda branca da imagem rotacionada.
    """
    # Obtem a matriz de rotacao
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])

    # Dimensoes da imagem apos rotacao (para nao haver crop)
    new_w = int(w * cos + h * sin)
    new_h = int(w * sin + h * cos)

    # Shift do centro por conta do novo tamanho
    rot_mat[0, 2] += (new_w - w) / 2
    rot_mat[1, 2] += (new_h - h) / 2

    # Rotaciona a imagem (gera uma borda branca)
    rotated = cv.warpAffine(img, rot_mat, (new_w, new_h), borderValue=255)

    # Remove borda se necessario
    if remove_border:
        rotated = _remove_white_border(rotated)        

    return rotated

def add_histogram(img: np.ndarray) -> np.ndarray:
    """
    Retorna uma imagem com o histograma do perfil horizontal da imagem `img` adicionado a direita.
    """
    profile = _horizontal_projection(img)
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
