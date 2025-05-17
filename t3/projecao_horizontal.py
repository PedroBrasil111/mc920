import cv2 as cv
import numpy as np
from helper_functions import *

def profile(img):
    return np.sum(255 - img, axis=1)

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR, borderValue=255)    
    return result

def objective_function(perfil):
    return np.sqrt(np.sum(np.square(np.diff(perfil))))

def horizontal_projection_inclination(img, ang1, ang2, step):
    best_angle = 0
    best_objective_val = None

    for angle in range(ang1, ang2, step):
        # Rotaciona a imagem
        rotated = rotate_image(img, angle)
        rotated_profile = profile(rotated)
        objective_val = objective_function(rotated_profile)

        if best_objective_val == None or objective_val > best_objective_val:
            best_objective_val = objective_val
            best_angle = angle

    return best_angle, best_objective_val

def add_histogram(img):
    profile = np.sum(255 - img, axis=1)
    min_val, max_val = np.min(profile), np.max(profile)

    # 20% da largura da imagem
    n = np.floor(img.shape[1] * 0.2).astype(int)

    # Cria o histograma
    profile_img = np.full((img.shape[0], n), 255, dtype=np.uint8)
    for i in range(img.shape[0]):
        idx = int(np.floor((profile[i] - min_val) / (max_val - min_val) * (n - 1)))
        profile_img[i, :idx] = 0

    # Adiciona o histograma na imagem original
    reshaped = np.zeros((img.shape[0], img.shape[1] + n), dtype=np.uint8)
    reshaped[:, :img.shape[1]] = img
    reshaped[:, img.shape[1]:] = profile_img

    return reshaped

def main():
    img = cv.imread("images/pos_24.png", cv.IMREAD_GRAYSCALE)

    print(img.shape)
    ang, obj = horizontal_projection_inclination(img, -90, 90, 1)

    rotacionada = rotate_image(img, ang)
    # Exibe as imgs
    cv.imshow("Imagem Original", img)
    cv.imshow("Imagem Rotacionada", rotacionada)
    cv.imshow("Perfil Original", add_histogram(img))
    cv.imshow("Perfil", add_histogram(rotacionada))
    print(f"Melhor ângulo: {ang}°")
    print(f"Melhor função objetivo: {obj}")
    cv.waitKey(0)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()
