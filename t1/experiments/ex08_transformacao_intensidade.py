import cv2 as cv
import os
import numpy as np

image_path = os.path.join(os.path.dirname(__file__), 'images', 'city.png')
result_filename = os.path.basename(__file__).replace(".py", ".png") 
result_path = os.path.join(os.path.dirname(__file__), 'results', result_filename)

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
nrows, ncols = image.shape

#b
negativo = np.subtract(255, image)
#c
img_min, img_max = np.min(image), np.max(image)
transformada = cv.normalize(image, None, 100, 200, cv.NORM_MINMAX)
#d
linhas_pares_invertidas = image.copy()
linhas_pares_invertidas[::2] = np.flip(linhas_pares_invertidas[::2], axis=1)
#e
reflexao_linhas = image.copy()
if nrows & 1 == 0: # numero par de linhas
    reflexao_linhas[nrows//2:] = np.flipud(reflexao_linhas[:nrows//2])
else: # numero impar (parte flipada comeca uma linha na frente)
    reflexao_linhas[nrows//2 + 1:] = np.flipud(reflexao_linhas[:nrows//2])
#f
reflexao_vertical = np.flipud(image)

transformations = {
    "negativo": negativo,
    "transformada": transformada,
    "linhas_pares_invertidas": linhas_pares_invertidas,
    "reflexao_linhas": reflexao_linhas,
    "espelhamento_vertical": reflexao_vertical
}

for name, result in transformations.items():
    cv.imshow(name, result)
    cv.imwrite(result_path.replace("08", f"08_{name}"), result)
    cv.waitKey(0)
