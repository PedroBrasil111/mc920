import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_sinusoidal_pattern(img, freq_x, freq_y, amplitude=50):
    """Add a 2D sinusoidal pattern with given frequencies to the image."""
    rows, cols = img.shape
    x = np.arange(cols)
    y = np.arange(rows)
    xv, yv = np.meshgrid(x, y)
    sinusoid = amplitude * np.sin(2 * np.pi * (freq_x * xv / cols + freq_y * yv / rows))
    noisy_img = img + sinusoid
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img.astype(np.uint8)

# Example usage
img = cv2.imread('../images/baboon_monocromatica.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found or invalid path.")

for freq in range(5, 11):
    noisy_img = add_sinusoidal_pattern(img, freq_x=freq, freq_y=freq, amplitude=50)

cv2.imshow('Original Image', img)
cv2.imshow('Noisy Image', noisy_img)
cv2.waitKey(0)
cv2.imwrite("./images/baboon_monocromatica_sinusoidal.png", noisy_img)