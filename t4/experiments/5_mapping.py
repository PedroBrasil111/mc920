import numpy as np
import os
import cv2 as cv

def scale_matrix(scale: float) -> np.ndarray:
    """
    Retorna a matriz de escala.
    """
    return np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, 1]], dtype=np.float32)

def rotation_matrix(angle: float) -> np.ndarray:
    """
    Retorna a matriz de rotacao.
    """
    angle_rad = np.deg2rad(angle)
    return np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                     [np.sin(angle_rad), np.cos(angle_rad), 0],
                     [0, 0, 1]], dtype=np.float32)

def nearest_neighbor_interpolation(
    img: np.ndarray, transformed_coords: np.ndarray, h_new: int, w_new: int
) -> np.ndarray:
    output_img = np.full((h_new, w_new, img.shape[2]), (0,0,0), dtype=img.dtype)

    h_in, w_in = img.shape[:2]
    for i in range(h_in):
        for j in range(w_in):
            y, x = transformed_coords[:, i, j]
            x, y = int(round(x)), int(round(y))
            if 0 <= x < w_new and 0 <= y < h_new:
                output_img[y, x] = img[i, j]
    return output_img


def bilinear_interpolation(img: np.ndarray, transformed_coords: np.ndarray) -> np.ndarray:
    return

def homogeneous_coordinates(img: np.ndarray, scale: float, angle: float) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    h_scaled, w_scaled = int(round(h * scale)), int(round(w * scale))
    
    coords = np.mgrid[0:h, 0:w].reshape(2, -1)  # Original coords
    coords = np.vstack((coords, np.ones((1, coords.shape[1]))))  # shape (3, h*w)

    scale_mat = scale_matrix(scale)
    rotation_mat = rotation_matrix(angle)
    transform_mat = rotation_mat @ scale_mat

    transformed_coords = transform_mat @ coords
    transformed_coords = transformed_coords[:2] / transformed_coords[2]  # shape (2, h*w)

    transformed_coords = transformed_coords.reshape(2, h, w)  # keep shape aligned with input
    return transformed_coords, h_scaled, w_scaled


img = cv.imread("images/baboon_colorida.png", cv.IMREAD_COLOR)

img = cv.imread("images/baboon_colorida.png", cv.IMREAD_COLOR)

transformed_coords, h_new, w_new = homogeneous_coordinates(img, 2, 0)
output_img_nn = nearest_neighbor_interpolation(img, transformed_coords, h_new, w_new)

cv.imwrite("experiments/images/output_nn.png", output_img_nn)
print("Done")