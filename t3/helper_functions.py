import cv2 as cv
from numpy import ndarray
import os

# Diretorios devem estar no mesmo nivel do arquivo
IMAGE_FOLDER  = os.path.join(os.path.dirname(__file__), 'images')
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), 'results')

def get_image_path(image_name: str) -> str:
    """
    Retorna o caminho completo da imagem especificada pelo nome.
    O nome da imagem deve incluir a extensao (ex: 'waterfall.png').
    """
    return os.path.join(IMAGE_FOLDER, image_name)

def get_all_images() -> list[str]:
    """
    Retorna uma lista com os nomes de todas as imagens no diretorio IMAGE_FOLDER.
    """
    images = []
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            images.append(filename)
    return images

def save_images(image_dict: dict[str, ndarray], exercise: int | str) -> None:
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
        filename = f"{title}.png"
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
