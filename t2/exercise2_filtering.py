from argparse import ArgumentParser
from helper_functions import (
    imread_auto, get_image_path, save_images, display_images
)
import numpy as np
import os

FILTER_TYPES = ['ideal', 'butterworth', 'gaussian']
FILTER_MODES = ['lowpass', 'highpass', 'bandpass', 'bandstop']
FILTER_MODES_CUTOFF = FILTER_MODES[:2]
FILTER_MODES_BAND = FILTER_MODES[2:]

EPSILON = 1e-8 # para evitar divisao por zero

def center_distance_array(shape: tuple[int]) -> np.ndarray:
    """
    Retorna a matriz de distancias para o centro da imagem com dimensoes (shape[0], shape[1]).
    """
    center = (shape[0] // 2, shape[1] // 2)
    Y, X = np.ogrid[:shape[0], :shape[1]] # cria uma grade de coordenadas
    dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2) # distancia ao centro
    return dist

def ideal_cutoff_kernel(shape: tuple[int], mode: str, cutoff: int) -> np.ndarray:
    d = center_distance_array(shape)
    h = np.where(d <= cutoff, 1, 0)
    return h if mode == 'lowpass' else 1 - h

def ideal_band_kernel(shape: tuple[int], mode: str, cutoff: int, width: int) -> np.ndarray:
    d = center_distance_array(shape)
    h = np.where(
        (d > (cutoff - width/2)) & 
        (d < (cutoff + width/2)), 1, 0
    )
    return h if mode == 'bandpass' else 1 - h

def butterworth_cutoff_kernel(shape: tuple[int], mode: str, cutoff: int, order: int) -> np.ndarray:
    d = center_distance_array(shape)
    h = 1 / (1 + (d / cutoff)**(2 * order))
    return h if mode == 'lowpass' else 1 - h

def butterworth_band_kernel(shape: tuple[int], mode: str, cutoff: int, width: int, order: int) -> np.ndarray:
    d = center_distance_array(shape)
    h = 1 / (1 + ((d**2 - cutoff**2) / (d*width + EPSILON))**(2*order)) # evita divisao por zero
    return h if mode == 'bandpass' else 1 - h

def gaussian_cutoff_kernel(shape: tuple[int], mode: str, cutoff: int) -> np.ndarray:
    d = center_distance_array(shape)
    h = np.exp(-(d**2) / (2 * (cutoff**2)))
    return h if mode == 'lowpass' else 1 - h

def gaussian_band_kernel(shape: tuple[int], mode: str, cutoff: int, width: int) -> np.ndarray:
    d = center_distance_array(shape)
    h = np.exp(-((d**2 - cutoff**2) / (width * cutoff))**2)
    return h if mode == 'bandpass' else 1 - h

def build_frequency_kernel(
        shape: tuple[int], filter_type: str, filter_mode: str,
        cutoff: int, width: int = 1, order: int = 2) -> np.ndarray:
    """
    Cria um filtro para o dominio da frequencia.

    :param shape: dimensoes da imagem (nrows, ncols)
    :param filter_type: tipo de filtro ('ideal', 'butterworth', 'gaussian')
    :param filter_mode: modo de filtragem ('lowpass', 'highpass', 'bandpass', 'bandstop')
    :param cutoff: frequencia de corte
    :param width: largura do filtro (apenas para bandpass e bandstop)
    :param order: ordem do filtro (apenas para butterworth)
    :return: filtro desejado
    :raise ValueError: se o tipo de filtro ou modo de filtragem forem invalidos
    """

    filter_functions = {
        'ideal': (ideal_cutoff_kernel, ideal_band_kernel),
        'butterworth': (butterworth_cutoff_kernel, butterworth_band_kernel),
        'gaussian': (gaussian_cutoff_kernel, gaussian_band_kernel)
    }

    # Definicao de qual funcao usar e com quais parametros
    index = 0 if filter_mode in FILTER_MODES_CUTOFF else 1
    params = (shape, filter_mode, cutoff)
    if index == 1: # bandpass e bandstop requerem largura da banda
        params += (width,)
    if filter_type == 'butterworth':
        params += (order,)

    return filter_functions[filter_type][index](*params)

def apply_frequency_filter(
        frequency_image: np.ndarray, filter_type: str, filter_mode: str,
        cutoff: int, width: int = 1, order: int = 2) -> np.ndarray:
    """
    Dada uma imagem centralizada no dominio da frequencia, aplica um filtro
    de frequencia e retorna a imagem filtrada.

    :param frequency_image: imagem no dominio da frequencia
    :param filter_type: tipo de filtro ('ideal', 'butterworth', 'gaussian')
    :param filter_mode: modo de filtragem ('lowpass', 'highpass', 'bandpass', 'bandstop')
    :param cutoff: frequencia de corte
    :param width: largura do filtro (apenas para bandpass e bandstop)
    :param order: ordem do filtro (apenas para butterworth)
    :return (filtered_image, filtered_frequency): imagem filtrada e imagem no dominio da frequencia filtrada
    """
    axes = (0, 1)
    # Cria e aplica o filtro
    kernel = build_frequency_kernel(
        frequency_image.shape, filter_type, filter_mode,
        cutoff, width, order
    )
    filtered_freq = frequency_image * kernel
    # Obtem a imagem filtrada a partir da transformada inversa
    filtered_image = np.abs(np.fft.ifft2(
        np.fft.ifftshift(filtered_freq, axes=axes),
        axes=axes
    ))
    return filtered_image, filtered_freq

def get_magnitude_image(frequency_image: np.ndarray) -> np.ndarray:
    """
    Retorna o espectro de magnitude de uma imagem no dominio da frequencia.
    O espectro de magnitude eh calculado como 20*log10(|F(u,v)|),
    onde F(u,v) eh a transformada de Fourier da imagem.
    O espectro de magnitude eh normalizado para o intervalo [0, 255].
    """
    result = np.clip(20*np.log(np.abs(frequency_image),
                     where=(frequency_image!=0)),
                     0, 255
                    )
    return result

def handle_args(args):
    """
    Trata os argumentos da linha de comando.
    """
    if args.type:
        for filter_type in args.type:
            if filter_type not in FILTER_TYPES:
                raise ValueError(f"Invalid filter type: {filter_type}")
    else:
        args.type = FILTER_TYPES
    if args.mode:
        for filter_mode in args.mode:
            if filter_mode not in FILTER_MODES:
                raise ValueError(f"Invalid filter mode: {filter_mode}")
    else:
        args.mode = FILTER_MODES
    if args.cutoff <= 0:
        raise ValueError("Cutoff frequency must be greater than 0.")
    if args.width <= 0:
        raise ValueError("Width must be greater than 0.")
    if args.order <= 0:
        raise ValueError("Order must be greater than 0.")
    if args.image:
        if not os.path.exists(get_image_path(args.image)):
            raise ValueError(f"Image {args.image} not found.")
    if not args.save and not args.display:
        raise ValueError("No action specified. Use -s to save images and/or -d to display them.")

def handle_results(
        image: np.ndarray,
        results: dict[str, np.ndarray],
        save: bool,
        display: bool,
        prefix: str = ""
        ) -> None:
    """
    Exibe e/ou salva os resultados.
    """

    for filter_mode, filter_type_results in results.items():
        filter_type_results = {
            f"{prefix}_{filter_mode}_{filter_type}": res
            for filter_type, res in filter_type_results.items()
        }
        # Salva os resultados
        if save:
            save_images(filter_type_results, exercise=2)
        # Mostra os resultados e a imagem original
        filter_type_results["original"] = image
        if display and not display_images(filter_type_results):
            break # exibe imgs e interrompe o loop se o usuario pressionar 'q'

def run(args):
    image = imread_auto(get_image_path(args.image))

    results = {} # resultados no dominio espacial
    freq_results = {} # resultados no dominio da frequencia (apenas magnitude)
    c, w, n = args.cutoff, args.width, args.order

    transformed = np.fft.fft2(image) # aplica a fft
    fshift = np.fft.fftshift(transformed, axes=(0, 1)) # centraliza o espectro

    # Itera sobre os tipos de filtro e modos de filtragem
    for filter_mode in args.mode:
        results[filter_mode] = {}
        freq_results[filter_mode] = {}

        for filter_type in args.type:
            filtered_image, filtered_freq = apply_frequency_filter(
                fshift, filter_type, filter_mode,
                c, w, n
            )
            results[filter_mode][filter_type] = filtered_image.astype(np.uint8)
            freq_results[filter_mode][filter_type] = \
                get_magnitude_image(filtered_freq).astype(np.uint8)

    print("\033[92mProcessing completed successfully.\033[0m") # verde

    # Exibe as imagens filtradas
    if args.display:
        print("\033[93mDisplaying filtered images. Press 'q' to exit, 'n' to start displaying the magnitude images and any other key to display the next one.\033[0m")
    handle_results(image, results,
                   args.save, args.display, prefix="filtered")
    # Salva os resultados
    if args.save:
        print("\033[92mImages saved successfully.\033[0m") # verde 
    # Exibe espectros filtrados
    if args.display:
        # pressionar 'n' tambem skipa essa parte, mas nao eh um grande problema
        print("\033[93mDisplaying magnitude images. Press 'q' to exit and any other key to display the next one.\033[0m")
    handle_results(get_magnitude_image(fshift).astype(np.uint8)
                   , freq_results, False, args.display, prefix="spectrum")

    print("\033[92mDone.\033[0m") # verde 

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("-i", "--image", type=str, default="baboon_monocromatica.png", help="Image name (with extension) - default: baboon_monocromatica.png")
    args.add_argument("-t", "--type", nargs="+", type=str, help="List of filter types (ideal, butterworth, gaussian) - default: all")
    args.add_argument("-m", "--mode", nargs="+", type=str, help="List of filter modes (low, high, bandpass, bandstop) - default: all")
    args.add_argument("-c", "--cutoff", type=int, default=50, help="Cutoff frequency - default: 50")
    args.add_argument("-w", "--width", type=int, default=10, help="Width of the filter (only for bandpass and bandstop) - default: 10")
    args.add_argument("-o", "--order", type=int, default=2, help="Order of the filter (only for butterworth) - default: 2")
    args.add_argument("-s", "--save", action="store_true", help="Save the images")
    args.add_argument("-d", "--display", action="store_true", help="Display the images")
    args = args.parse_args()

    # Trata os argumentos da linha de comando
    try:
        handle_args(args)
    except ValueError as e:
        print(f"\033[91mError: {e}\033[0m")
        exit(1)

    # Aplica a filtragem
    run(args)
