import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend

import argparse
import cv2 as cv
from arrays import get_array, get_all_arrays
from helper_functions import *
from exp_functions import *
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def distribute_error_for(image, row, col, error, kernel):
    kernel_nrows, kernel_ncols = kernel.shape
    half_kcols = kernel_ncols // 2

    for i in range(kernel_nrows):
        for j in range(-half_kcols, half_kcols + 1):
            if kernel[i, j + half_kcols] != 0:
                target_row = row + i
                target_col = col + j
                if 0 <= target_row < image.shape[0] and 0 <= target_col < image.shape[1]:
                    image[target_row, target_col] += error * kernel[i, j + half_kcols]

def distribute_error_slice(
        image: np.ndarray, row: int, col: int,
        error: np.ndarray | np.float64, kernel: np.ndarray
        ) -> None: 
    """
    Aplica a distribuicao de erro na vizinhanca de (row, col) na imagem.
    O erro eh distribuido para os pixels vizinhos de acordo com a matriz
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

def apply_error_diffusion(image: np.ndarray,
                          kernel: np.ndarray,
                          err_funct,
                          trocar) -> np.ndarray:
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
        if trocar: 
            col_range = range(ncols) if r % 2 == 0 else range(ncols - 1, -1, -1)
            current_kernel = kernel if r % 2 == 0 else flipped_kernel
        else:
            col_range = range(ncols)
            current_kernel = kernel
        # varre a linha
        for c in col_range:
            # operacoes sao feitas em todos os canais
            new_val = np.where(output[r, c] < 128, 0, 255)
            error = output[r, c] - new_val
            output[r, c] = new_val
            err_funct(output, r, c, error, current_kernel)
    return output

### 01
def meio_tom(image_name: str = "baboon_monocromatica.png", err_funct=distribute_error_slice) -> dict[str, np.ndarray]:
    """
    Aplica a tecnica de meio tom com difusao de erro na imagem usando
    as matrizes de distribuicao de erro fornecidas.
    Retorna um dicionario com as imagens resultantes.
    """
    image = imread_auto(get_image_path(image_name)) # numero de canais baseado na imagem
    result_images = {} # armazena as imagens resultantes
    kernel_dict = get_all_arrays() # dicionario com as matrizes de distribuicao de erro

    for kernel_name, kernel in list(kernel_dict.items())[:1]:
        output = apply_error_diffusion(image, kernel, err_funct, trocar=True)
        output2 = apply_error_diffusion(image, kernel, err_funct, trocar=False)
        result_images[f"meio_tom_{kernel_name}_troca"] = output.astype(np.uint8)
        result_images[f"meio_tom_{kernel_name}_sem"] = output2.astype(np.uint8)

    return result_images

def main_comparison():
    print("distribute_error_slice")
    res1 = meio_tom("baboon_colorida.png", distribute_error_slice)
    print("\ndistribute_error_for")
    res2 = meio_tom("baboon_colorida.png", distribute_error_for)

    for k in res1.keys():
        print("\nChecking", k)
        if not np.array_equal(res1[k], res2[k]): 
            print(f"Results differ")
        else:
            print("Results are the same")
        cv.imshow(k, res1[k])
        cv.imshow(k + "_for", res2[k])
        cv.waitKey(0)
        cv.destroyAllWindows()

def main_slice():
    print("distribute_error_slice")
    res = meio_tom("wavefront_512.png", distribute_error_slice)
    for k in res.keys():
        print("Displaying", k)
        cv.imshow(k, res[k])
    cv.waitKey(0)
    cv.destroyAllWindows()

def size_experiment():
    gray_images = [f"noise_{i}.png" for i in [16, 32, 64, 128, 256, 512, 1024]]
    rgb_images = [f"noise_{i}_RGB.png" for i in [16, 32, 64, 128, 256, 512, 1024]]
    slice_gray_times = []
    slice_rgb_times = []
    for_gray_times = []
    for_rgb_times = []

    print("distribute_error_slice")
    for image_name in gray_images:
        start_time = time.time()
        res = meio_tom(image_name, distribute_error_slice)
        slice_gray_times.append(time.time() - start_time)
        print(f"\nTime taken for {image_name}: {slice_gray_times[-1]} seconds")
    for image_name in rgb_images:
        start_time = time.time()
        res = meio_tom(image_name, distribute_error_slice)
        slice_rgb_times.append(time.time() - start_time)
        print(f"\nTime taken for {image_name}: {slice_rgb_times[-1]} seconds")
    print("\ndistribute_error_for")
    for image_name in gray_images:
        start_time = time.time()
        res = meio_tom(image_name, distribute_error_for)
        for_gray_times.append(time.time() - start_time)
        print(f"\nTime taken for {image_name}: {for_gray_times[-1]} seconds")
    for image_name in rgb_images:
        start_time = time.time()
        res = meio_tom(image_name, distribute_error_for)
        for_rgb_times.append(time.time() - start_time)
        print(f"\nTime taken for {image_name}: {for_rgb_times[-1]} seconds")
    print()
    print("Slice gray total time:", sum(slice_gray_times))
    print("Slice RGB total time:", sum(slice_rgb_times))
    print("\nFor gray total time:", sum(for_gray_times))
    print("For RGB total time:", sum(for_rgb_times))

    return slice_gray_times, slice_rgb_times

if __name__ == '__main__':
    main_slice()
    #main_comparison()
    #size_experiment()