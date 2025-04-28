import matplotlib
matplotlib.use('Agg') # fix por causa de erro de backend

import cv2 as cv
import matplotlib.pyplot as plt
from numpy import ndarray
import os
import tempfile

# Diretorios devem estar no mesmo nivel do arquivo
IMAGE_FOLDER  = os.path.join(os.path.dirname(__file__), 'images')
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), 'results')

def get_image_path(image_name: str) -> str:
    """
    Retorna o caminho completo da imagem especificada pelo nome.
    O nome da imagem deve incluir a extensao (ex: 'waterfall.png').
    """
    return os.path.join(IMAGE_FOLDER, image_name)

def isgray(image: ndarray) -> bool:
    """
    Verifica se a imagem eh monocromatica.
    Retorna True se a imagem for monocromatica, False caso contrario.

    Adaptado de:
    https://stackoverflow.com/questions/23660929/
    """
    if len(image.shape) < 3: return True
    if image.shape[2]  == 1: return True
    b,g,r = image[:,:,0], image[:,:,1], image[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

def imread_auto(image_path: str) -> ndarray:
    """
    Retorna a imagem lida do caminho especificado.
    Se a imagem for monocromatica, retorna uma imagem com 1 canal.
    Se a imagem for colorida, retorna a imagem original.
    """
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    if isgray(image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image

def save_images(image_dict: dict[str, ndarray], exercise: int) -> None:
    """
    Salva as imagens do dicionario image_dict no diretorio RESULT_FOLDER.
    O nome do arquivo eh composto pelo numero do exercicio e o titulo da imagem.
    """
    str_ex = str(exercise).zfill(2) # adiciona um zero a esquerda se ex < 10

    if not os.path.exists(os.path.join(RESULT_FOLDER, str_ex)):
        os.makedirs(os.path.join(RESULT_FOLDER, str_ex))
    
    ex_folder = os.path.join(RESULT_FOLDER, str_ex)

    params = [cv.IMWRITE_PNG_STRATEGY, cv.IMWRITE_PNG_STRATEGY_DEFAULT]
    for title, image in image_dict.items():
        filename = f"ex{str_ex}_{title}.png"
        cv.imwrite(os.path.join(ex_folder, filename), image, params)

def display_images(image_dict: dict[str, ndarray], single: bool=False) -> bool:
    """"
    Exibe as imagens em uma janela, uma a uma, e aguarda que uma tecla seja pressionada.
    As possibilidades sao 'n' para interromper o exercicio atual, 'q' para sair
        e qualquer outra tecla para continuar.
    Retorna False se o usuario pressionar 'n', True caso contrario.
    """
    for title, image in image_dict.items():
        cv.imshow(title, image)
        if single:
            key = cv.waitKey(0)
            if key == ord('q'):
                print("\033[91mExecution interrupted by user (pressed 'q').\033[0m") # vermelho
                exit()
            elif key == ord('n'):
                return False
            cv.destroyAllWindows()
    if not single:
        key = cv.waitKey(0)
        cv.destroyAllWindows()
        if key == ord('q'):
            print("\033[91mExecution interrupted by user (pressed 'q').\033[0m")
            exit()
        elif key == ord('n'):
            return False
    return True
