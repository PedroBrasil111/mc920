import cv2 as cv
import numpy as np
from helper_functions import *

def funcao_objetivo(perfil):
    func_obj = np.sum(np.square(np.diff(perfil)))
    return func_obj

def calcular_inclinacao(img, ang1, ang2, passo):
    melhor_angulo = 0
    melhor_func_obj = None
    # Calcula a inclinação da imagem
    for angulo in range(ang1, ang2, passo):
        # Rotaciona a imagem
        img_rotacionada = cv.rotate(img, angulo)

        # Calcula a projeção horizontal
        perfil = np.sum(img_rotacionada, axis=0)

        # Calcula a função objetivo
        func_obj = funcao_objetivo(perfil)

        if melhor_func_obj == None or func_obj > melhor_func_obj:
            melhor_func_obj = func_obj
            melhor_angulo = angulo

    return melhor_angulo, melhor_func_obj

def main():
    img = imread_auto("images/neg_4.png")
    ang = calcular_inclinacao(img, 0, 360, 1)
    # Exibe as imgs
    cv.imshow("Imagem Original", img)
    cv.imshow("Imagem Rotacionada", cv.rotate(img, ang[0]))
    cv.waitKey(0)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()
