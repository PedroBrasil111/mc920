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

def display_images(image_dict: dict[str, ndarray]) -> bool:
    """
    Desenvolvido com auxílio de IA.
    Salva uma figura com subplots (2, n/2) de imagens usando matplotlib,
    exibe a figura salva com OpenCV e espera por uma tecla.
    Retorna False se 'n' for pressionado, True caso contrario.
    """

    n = len(image_dict)
    if n == 0:
        print("\033[91mNo images to display.\033[0m")
        return True

    # Create matplotlib figure
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(10, 6))
    axes = axes.flatten()

    # Ensure "original" is processed first if it exists
    items = list(image_dict.items())
    items.sort(key=lambda x: x[0] != "original")

    for ax, (title, img) in zip(axes, items):
        title = title.capitalize().replace('_', ' ')
        if img.ndim == 2:  # grayscale
            ax.imshow(img, cmap='gray')
        else:  # color
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    # Hide any unused axes
    for ax in axes[len(image_dict):]:
        ax.axis('off')

    plt.tight_layout()

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        temp_path = tmpfile.name
    plt.savefig(temp_path, dpi=150)
    plt.close(fig)

    # Read with OpenCV and show
    img_cv = cv.imread(temp_path)
    os.remove(temp_path)  # clean up temp file immediately

    if img_cv is None:
        print("\033[91mFailed to load the saved figure.\033[0m")
        return True

    cv.imshow('Image Set', img_cv)
    key = cv.waitKey(0)
    cv.destroyAllWindows()

    if key == ord('q'):
        print("\033[91mExecution interrupted by user (pressed 'q').\033[0m")
        exit()
    elif key == ord('n'):
        return False
    return True
