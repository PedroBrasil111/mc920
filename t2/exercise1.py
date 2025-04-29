import argparse
from error_dispersion_matrices import get_matrices
from helper_functions import (
    get_image_path, save_images, display_images, imread_auto
)
import numpy as np
import os

# Carrega todas as matrizes de distribuicao de erro
ERROR_DIFF_MATRICES = get_matrices()

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

def meio_tom(image: np.ndarray, matrices: list[str]) -> dict[str, np.ndarray]:
    """
    Aplica a tecnica de meios-tons com difusao de erro na imagem usando
    as matrizes de distribuicao de erro fornecidas.
    Retorna um dicionario com as imagens resultantes.
    """
    result_images = {} # armazena as imagens resultantes
    kernel_dict = { # dicionario com as matrizes de distribuicao de erro
        name: ERROR_DIFF_MATRICES[name] for name in matrices
    } 

    for kernel_name, kernel in kernel_dict.items():
        output = apply_error_diffusion(image, kernel)
        result_images[f"meios_tons_{kernel_name}"] = output.astype(np.uint8)

    return result_images

def handle_result(
        original: np.ndarray,
        result: dict[str, np.ndarray],
        save: bool,
        display: bool
        ) -> bool:
    """
    Lida com o resultado da funcao de exercicio.
    Salva e/ou exibe as imagens resultantes.
    """
    if result:
        if save:
            save_images(result, exercise=1)
        result["original"] = original
        if display:
            return display_images(result, single=False) # retorna falso se o usuario pressionar 'n'
    return True

def handle_args(args) -> None:
    """
    Trata os argumentos da linha de comando.
    """
    if args.image:
        if not os.path.exists(get_image_path(args.image)):
            raise ValueError(f"Image {args.image} not found.")
    else:
        args.image = "baboon_monocromatica.png"
    if args.matrix:
        for matrix_name in args.matrix:
            if matrix_name.lower() not in ERROR_DIFF_MATRICES:
                raise ValueError(f"Unknown matrix name: {matrix_name}")
    else:
        args.matrix = list(ERROR_DIFF_MATRICES.keys())
    if not args.save and not args.display:
        raise ValueError("No action specified. Use -s to save images and/or -d to display them.")

def run(args) -> None:
    """
    Executa o programa principal.
    """
    image = imread_auto(get_image_path(args.image)) # numero de canais baseado na imagem

    print("\033[93mProcessing...\033[0m") # amarelo

    # Aplica a filtragem
    result = meio_tom(image, args.matrix)

    # Lida com o resultado
    if args.save:
        print("\033[92mImages saved successfully.\033[0m")
    if args.display:
        print("\033[93mDisplaying images. Press 'q' to interrupt.\033[0m")
    handle_result(image, result, args.save, args.display)

    print("\033[92mDone.\033[0m") # verde

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply halftone dithering to images through the error diffusion method.")
    parser.add_argument("-i", "--image", type=str, help="Image name (with extension) - default: baboon_monocromatica.png")
    parser.add_argument("-m", "--matrix", nargs="+", type=str, help="Error diffusion matrix names ('floyd_steinberg', 'stevenson_arce', 'burkes', 'sierra', 'stucki', 'jarvis_judice_ninke', 'custom')- default: all")
    parser.add_argument("-s", "--save", action="store_true", help="Save processed images")
    parser.add_argument("-d", "--display", action="store_true", help="Display processed images")
    args = parser.parse_args()

    try:
        handle_args(args)
    except ValueError as e:
        print(f"\033[91mError: {e}\033[0m")
        exit(1)

    run(args)
