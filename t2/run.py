import argparse
import cv2 as cv
from arrays import get_array, get_all_arrays
from helper_functions import (
    get_image_path, save_images, display_images, imread_auto
)
import numpy as np
import os

def spread_error(
        image: np.ndarray, row: int, col: int,
        error: np.ndarray | np.float64, kernel: np.ndarray
        ) -> None: 
    """
    Aplica a distribuicao de erro na vizinhanca de (row, col) na imagem.
    O erro eh distribuido para os pixels proximos de acordo com a matriz
    de distribuicao de erro (kernel).
    """
    kernel_nrows, kernel_ncols = kernel.shape
    half_kcols = kernel_ncols // 2

    # indices para slice da imagem
    row_end   = min(image.shape[0], row + kernel_nrows)
    col_start = max(0, col - half_kcols)
    col_end   = min(image.shape[1], col + half_kcols + 1)

    # slicing do kernel (apenas parte "dentro da imagem")
    col_offset = max(0, half_kcols - col)
    subkernel = kernel[:row_end - row,
                       col_offset:col_offset + (col_end - col_start)
                      ]
    if len(error.shape) > 0: # broadcasting quando imagem tem mais de um canal
        subkernel = subkernel[:, :, np.newaxis]

    # aplica a distribuicao de erro
    image[row:row_end, col_start:col_end] += subkernel * error

def apply_error_diffusion(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Aplica a tecnica de meio tom com difusao de erro na imagem usando
    a matriz de distribuicao de erro fornecida (kernel).
    Retorna a imagem resultante.
    """
    nrows, ncols = image.shape[:2]
    image = image.astype(np.float32) # difusao de erro requer operacoes com float
    output = image.copy() # difusao sera feita inplace no output
    flipped_kernel = np.fliplr(kernel)
    for r in range(nrows):
        # inverte direcao de varredura a cada linha
        col_range = range(ncols) if r % 2 == 0 else range(ncols - 1, -1, -1)
        current_kernel = kernel if r % 2 == 0 else flipped_kernel
        # varre a linha
        for c in col_range:
            # operacoes sao feitas em todos os canais
            thresholded = np.where(output[r, c] < 128, 0, 255)
            error = output[r, c] - thresholded
            output[r, c] = thresholded
            # distribui o erro para os pixels proximos
            spread_error(output, r, c, error, current_kernel)
    return output

### 01
def meio_tom(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    """
    Aplica a tecnica de meio tom com difusao de erro na imagem usando
    as matrizes de distribuicao de erro fornecidas.
    Retorna um dicionario com as imagens resultantes.
    """
    image = imread_auto(get_image_path(image_name)) # numero de canais baseado na imagem
    result_images = {} # armazena as imagens resultantes
    kernel_dict = get_all_arrays() # dicionario com as matrizes de distribuicao de erro

    for kernel_name, kernel in kernel_dict.items():
        output = apply_error_diffusion(image, kernel)
        result_images[f"meio_tom_{kernel_name}"] = output.astype(np.uint8)

    return result_images

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
                    continue
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
    Salva e/ou exibe as imagens resultantes.
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
