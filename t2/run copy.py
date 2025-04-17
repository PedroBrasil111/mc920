import argparse
import cv2 as cv
from arrays import get_array, get_all_arrays
from helper_functions import (
    get_image_path, save_images, display_images
)
import numpy as np
import os

### 01
def meio_tom3(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE).astype(np.float32)
    floyd = get_array("floyd_steinberg")

    for r in range(image.shape[0]):
        iter_c = range(image.shape[1]) if r % 2 == 0 else range(image.shape[1] - 1, -1, -1)
        k = floyd if r % 2 == 0 else np.fliplr(floyd)
        for c in iter_c:
            old_pixel = image[r, c]
            new_pixel = 0 if old_pixel < 128 else 255
            error = old_pixel - new_pixel
            image[r, c] = new_pixel

            half_r, half_c = floyd.shape[0] // 2, floyd.shape[1] // 2
            for i in range(0, floyd.shape[0]):
                for j in range(-half_c, half_c + 1):
                    rr, cc = r + i, c + j
                    if 0 <= rr < image.shape[0] and 0 <= cc < image.shape[1]:
                        image[rr, cc] += error * k[i, j + half_c]

    image = np.clip(image, 0, 255).astype(np.uint8)
    cv.imshow("Floyd-Steinberg", image)
    cv.waitKey(0)

    return None


def meio_tom(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE).astype(np.float32)

    floyd = get_array("floyd_steinberg")
    flipped = np.fliplr(floyd)
    half_r, half_c = floyd.shape[0] // 2, floyd.shape[1] // 2

    for r in range(image.shape[0]):
        iter_c = range(image.shape[1]) if r % 2 == 0 else range(image.shape[1] - 1, -1, -1)
        k = floyd if r % 2 == 0 else flipped
        for c in iter_c:
            aux = np.where(image[r, c] < 128, 0, 1)
            err = image[r, c] - aux*255
            image[r, c] = aux

            for i in range(0, floyd.shape[0]):
                for j in range(-half_c, half_c + 1):

                    if (r + i >= 0 and r + i < image.shape[0] and
                            c + j >= 0 and c + j < image.shape[1]):
                        image[r + i, c + j] += err * k[i, j + half_c]

    image = (image * 255).astype(np.uint8)
    print(f"PIXELS NOT IN {0, 255}:", np.sum(np.where(~((image == 0) | (image == 255)), 1, 0)))
    #cv.imshow("Original", image)
    cv.imshow("Floyd-Steinberg", image)
    cv.waitKey(0)

    return None

### 02
def filtragem_frequencia(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    return None

def process_and_handle_exercise(
        exercise_function: callable,
        exercise_number: int,
        image_names: list[str],
        save: bool,
        display: bool
        ) -> None:
    """
    Processa a imagem usando a funcao exercise_function e lida com o resultado.
    Se o resultado for um dicionario, salva e/ou exibe as imagens.
    Se o resultado for None, encerra.
    """
    # Verifica quais imagens existem
    if image_names:
        for image_name in image_names:
            if not os.path.isfile(get_image_path(image_name)):
                print(f"\033[91mImage {image_name} not found.\033[0m") # vermelho
                image_names.remove(image_name)
                if not image_names:  # encerra se nenhuma imagem for encontrada
                    print("\033[91mNo valid images found.\033[0m") # vermelho
                    return

    # Caso especial para o exercicio 7, que combina duas imagens
    if image_names and exercise_number == 7:
        for i in range(0, len(image_names), 2):
            image_name_1 = image_names[i]
            if i + 1 < len(image_names):
                image_name_2 = image_names[i + 1]
                result = exercise_function(image_name_1, image_name_2)
            else:
                result = exercise_function(image_name_1)
            if not handle_result(result, exercise_number, save, display):
                return

    # Exercicios que requerem uma imagens
    else: 
        if image_names:
            for image_name in image_names:
                result = exercise_function(image_name)
                if not handle_result(result, exercise_number, save, display):
                    return
        else:  # usa a imagem padrao se nenhuma for fornecida
            result = exercise_function()
            if not handle_result(result, exercise_number, save, display):
                return

def handle_result(
        result: dict[str, np.ndarray],
        exercise_number: int,
        save: bool,
        display: bool
        ) -> bool:
    """
    Lida com o resultado da funcao de exercicio.
    Se o resultado for um dicionario, salva e/ou exibe as imagens.
    Se o resultado for None, encerra.
    """
    if result:
        if save:
            save_images(result, exercise_number)
        if display:
            return display_images(result) # retorna falso se o usuario pressionar 'n'
    return True

def main(args: argparse.Namespace) -> None:
    """
    Funcao principal que processa os argumentos e executa os exercicios.
    """
    exercises = {
        1: meio_tom,
        2: filtragem_frequencia,
    }
    if not (args.d or args.s): # acao nao especificada
        print("\033[91mNo action specified. Use -s to save images and/or -d to display them.\033[0m") # vermelho
        return
    
    if not args.e: # exercicios fora do intervalo
        print("\033[91mAllowed exercises are 1-10.\033[0m") # vermelho

    for exercise_number in args.e:
        print(f"\033[94mProcessing exercise {exercise_number}...\033[0m") # azul
        try:
            process_and_handle_exercise(
                exercises[exercise_number],
                exercise_number,
                args.i,
                args.s,
                args.d
            )
        except Exception as e: # programa nao encerra quando um exercicio falha
            print(f"\033[91mError processing exercise {exercise_number}: {e}\033[0m") # vermelho
        print("\033[92mDone\033[0m") # verde

if __name__ == '__main__':
    meio_tom()
    """
    parser = argparse.ArgumentParser(description='Process images with various exercises.')
    parser.add_argument('-i', nargs='+', type=str, help='List of image names (or all)')
    parser.add_argument('-e', nargs='+', type=int, help='List of exercise numbers (1-10) - default: all')
    parser.add_argument('-s', action='store_true', help='Save processed images')
    parser.add_argument('-d', action='store_true', help='Display processed images')
    args = parser.parse_args()

    # Todas as imagens
    if args.i and args.i[0].lower() == 'all':
        args.i = [file for file in os.listdir(get_image_path('')) if not file.startswith('.')]
    # Todos os exercicios por padrao, e limita entrada a 1-2
    args.e = range(1, 3) if not args.e else [ex for ex in args.e if 1 <= ex <= 2]

    main(args)
    """