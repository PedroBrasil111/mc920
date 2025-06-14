import numpy as np

def _translation_matrix(tx: float, ty: float) -> np.ndarray:
    """
    Retorna a matriz de translacao 2D para os deslocamentos tx e ty.
    """
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=np.float32)

def _scaling_matrix(scale: float) -> np.ndarray:
    """
    Retorna a matriz de escala 2D para o fator de escala `scale`.
    """
    return np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, 1]], dtype=np.float32)

def _rotation_matrix(angle: float) -> np.ndarray:
    """
    Retorna a matriz de rotacao 2D, no sentido horario, para o angulo `angle` em graus.
    """
    angle_rad = np.deg2rad(angle)
    return np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                     [np.sin(angle_rad),  np.cos(angle_rad), 0],
                     [0, 0, 1]], dtype=np.float32)

def _calculate_output_dimensions(
        w: int, h: int,
        angle: float, scale: float
    ) -> tuple[int, int]:
    """
    Calcula as novas dimensoes de uma imagem w x h apos aplicacao de rotacao e escala.
    Retorna as novas dimensoes (largura, altura).
    """
    angle_rad = np.deg2rad(angle)
    cos_angle = np.abs(np.cos(angle_rad))
    sin_angle = np.abs(np.sin(angle_rad))
    new_w = int((w * cos_angle + h * sin_angle) * scale)
    new_h = int((w * sin_angle + h * cos_angle) * scale)
    return new_w, new_h

def _coordinate_grid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna um grid de coordenadas homogeneas para uma imagem de dimensoes w x h.
    com formato (3, h*w), onde a primeira linha contem as coordenadas x,
    a segunda linha contem as coordenadas y e a terceira linha contem 1's.
    """
    j_grid, i_grid = np.meshgrid(np.arange(w), np.arange(h)) # X, Y
    ones = np.ones_like(j_grid) # W (1's)
    grid = np.stack([j_grid, i_grid, ones], axis=0).reshape(3, -1)
    return grid

def spatial_transformation(
        img: np.ndarray,
        scale: float = 1., angle: float = 0.,
        max_w: int = None, max_h: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna as novas coordenadas de uma imagem `img` apos aplicar transformacoes de escala e rotacao.
    """
    h, w = img.shape[:2]
    new_w, new_h = _calculate_output_dimensions(w, h, angle, scale)
    # Centro da imagem original e da nova imagem
    cx, cy = w / 2, h / 2
    nx, ny = new_w / 2, new_h / 2
    # Matrizes de transformação
    original_trans_mat = _translation_matrix(-cx, -cy) # Translada para o centro da imagem original
    new_trans_mat = _translation_matrix(nx, ny) # Translada para o centro da nova imagem
    scale_mat = _scaling_matrix(scale) 
    rotation_mat = _rotation_matrix(angle) 
    transform = rotation_mat @ scale_mat # Combina escala e rotacao
    transform_mat = new_trans_mat @ transform @ original_trans_mat # Matriz final de transformação
    inv_transform = np.linalg.inv(transform_mat)
    # Gera o grid de coordenadas homogêneas para a nova imagem
    coords_out = _coordinate_grid(new_w, new_h)
    coords_in = inv_transform @ coords_out
    coords_in = coords_in[:2] / coords_in[2] # Volta para coordenadas cartesianas
    coords_in = coords_in.reshape(2, new_h, new_w)
    # Limita para as dimensioes especificadas e centraliza
    max_w = max_w if max_w is not None else new_w
    max_h = max_h if max_h is not None else new_h
    coords_in = coords_in[:,
                          new_h//2-max_h//2 : new_h//2+max_h//2,
                          new_w//2-max_w//2 : new_w//2+max_w//2
                         ]
    return coords_in

def _nearest_neighbor_interpolation(
        input_img: np.ndarray,
        x_in: int, y_in: int,
        dx: float, dy: float,
        w: int, h: int
    ) -> np.ndarray:
    """
    Retorna o valor interpolado usando o metodo do vizinho mais proximo.
    """
    value = np.zeros(3, dtype=np.float32)
    if 0 <= x_in < w - 1 and 0 <= y_in < h - 1:
        x_in = int(round(x_in + dx))
        y_in = int(round(y_in + dy))
        value = input_img[y_in, x_in]
    return value

def _bilinear_interpolation(
        input_img: np.ndarray,
        x_in: int, y_in: int,
        dx: float, dy: float,
        w: int, h: int
    ) -> np.ndarray:
    """
    Retorna o valor interpolado usando o metodo bilinear.
    """
    value = np.zeros(3, dtype=np.float32)
    if 0 <= x_in < w-1 and 0 <= y_in < h-1:
        value = (1 - dx) * (1 - dy) * input_img[y_in, x_in] + \
                dx * (1 - dy) * input_img[y_in, x_in + 1] + \
                (1 - dx) * dy * input_img[y_in + 1, x_in] + \
                dx * dy * input_img[y_in + 1, x_in + 1]
    return value

def _ramp(x: float):
    """
    Retorna o valor da funcao de rampa.
    """
    return x if x > 0 else 0

def _cubic_bspline(s: float):
    """
    Retorna o valor da funcao cubica B-spline.
    """
    return (_ramp(s + 2)**3 - 4*_ramp(s + 1)**3 + 6*_ramp(s)**3 - 4*_ramp(s - 1)**3) / 6

def _bicubic_interpolation(
        input_img: np.ndarray,
        x_in: int, y_in: int,
        dx: float, dy: float,
        w: int, h: int
    ) -> np.ndarray:
    """
    Retorna o valor interpolado usando o metodo bicubico.
    """
    val_out = np.zeros(3, dtype=np.float32)
    if 1 <= x_in < w - 2 and 1 <= y_in < h - 2:
        for n in range(-1, 3):
            for m in range(-1, 3):
                if 0 <= x_in + m < w and 0 <= y_in + n < h:
                    val_in = input_img[y_in + n, x_in + m]
                    val_out += val_in * _cubic_bspline(m - dx) * _cubic_bspline(dy - n)
    return val_out

def _polynomial(d, vals):
    """
    Retorna o valor do polinomio interpolador de Lagrange de grau 3.
    """
    num1 = (-d * (d-1) * (d-2) * vals[0]) / 6
    num2 = ((d+1) * (d-1) * (d-2) * vals[1]) / 2
    num3 = (-d * (d+1) * (d-2) * vals[2]) / 2
    num4 = (-d * (d+1) * (d-1) * vals[3]) / 6
    return num1 + num2 + num3 + num4

def _L_func(img, dx, n, x, y):
    """
    Retorna o valor do polinomio de Lagrange para o pixel (x, y) com deslocamento dx.
    """
    vals = [img[y + n - 2, x + i] for i in range(-1, 3)]
    return _polynomial(dx, vals)

def _lagrange_polynomial(img, x, y, dx, dy):
    """
    Retorna o valor do polinomio de Lagrange interpolador para o pixel (x, y).
    """
    vals = [_L_func(img, dx, n, x, y) for n in range(1, 5)]
    return _polynomial(dy, vals)

def _lagrange_interpolation(
        input_img: np.ndarray,
        x_in: int, y_in: int,
        dx: float, dy: float,
        w: int, h: int
    ) -> np.ndarray:
    """
    Retorna o valor interpolado usando o metodo de Lagrange.
    """
    value = np.zeros(3, dtype=np.float32)
    if 1 <= x_in < w - 2 and 1 <= y_in < h - 2:
        value = _lagrange_polynomial(input_img, x_in, y_in, dx, dy)
    return value

INTERPOLATION_FUNC = {
    "nn": _nearest_neighbor_interpolation,
    "bilinear": _bilinear_interpolation,
    "bicubic": _bicubic_interpolation,
    "lagrange": _lagrange_interpolation,
}

def intensity_interpolation(
        img: np.ndarray,
        out_coords: np.ndarray,
        method: str
    ) -> np.ndarray:
    """
    Interpola a imagem `img` com coordenadas homogeneas especificadas por `out_coords`,
    usando o metodo de interpolação especificado por `method`.
    Os metodos possiveis sao: "nn", "bilinear", "bicubic", "lagrange".
    """
    padded_img = np.pad(
        img, ((1, 2), (1, 2), (0, 0)), mode='reflect'
    )  # Preenche a imagem com bordas para garantir que todos os pixels sejam mapeados
    h, w = padded_img.shape[:2] # limites da imagem original
    output_img = np.zeros(
        (out_coords.shape[1], out_coords.shape[2], 3),
        dtype=np.float32
    )
    # itera sobre as coordenadas da nova imagem
    for r_out in range(out_coords.shape[1]):
        for c_out in range(out_coords.shape[2]):
            x_out, y_out = out_coords[:, r_out, c_out] # coordenadas homogeneas da nova imagem
            x_in, y_in = int(np.floor(x_out)), int(np.floor(y_out)) # coordenadas da imagem original
            dx, dy = x_out - x_in, y_out - y_in
            output_img[r_out, c_out] = INTERPOLATION_FUNC[method](
                padded_img, x_in, y_in, dx, dy, w, h
            )
    return output_img.clip(0, 255).astype(np.uint8)
