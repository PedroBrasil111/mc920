import numpy as np
import cv2 as cv

def scale_conversion(image: np.ndarray, min_val, max_val) -> np.ndarray:
    min_img = image.min(axis=(0, 1), keepdims=True)
    max_img = image.max(axis=(0, 1), keepdims=True)

    scaled = (image - min_img) / (max_img - min_img) * (max_val - min_val) - min_val
    return np.floor(scaled).astype(np.uint8)

def apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    b, g, r = cv.split(image)

    b_filtered = cv.filter2D(b, -1, kernel)
    g_filtered = cv.filter2D(g, -1, kernel)
    r_filtered = cv.filter2D(r, -1, kernel)

    filtered_image = cv.merge([b_filtered, g_filtered, r_filtered])

    return filtered_image
