from argparse import ArgumentParser
import cv2 as cv
from helper_functions import (
    imread_auto, get_image_path, save_images, display_images_subplots, display_images
)
import matplotlib.pyplot as plt
import numpy as np
import os

def calc_histogram(image: np.ndarray) -> np.ndarray:
    """
    Calcula o histograma da imagem e retorna como uma imagem RGB.
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    # Desenha o histograma
    ax.hist(image.flatten(), bins=256, range=(0, 256), color='blue')
    ax.set_xlim(0, 255)

    # Converte para imagem
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape((height, width, 4))  # RGBA
    rgb = cv.cvtColor(buf, cv.COLOR_RGBA2BGR)  # converte para BGR
    plt.close(fig)

    return rgb

def handle_args(args):
    """
    Trata os argumentos da linha de comando.
    """
    if args.image:
        if not os.path.exists(get_image_path(args.image)):
            raise ValueError(f"Image {args.image} not found.")
    if args.percentile < 0 or args.percentile > 100:
        raise ValueError("Percentile must be between 0 and 100.")
    if not args.save and not args.display:
        raise ValueError("No action specified. Use -s to save images and/or -d to display them.")

def apply_compression(image: np.ndarray, percentile: float) -> np.ndarray:
    """
    Aplica a compressao na imagem usando a FFT e o percentil especificado.
    Retorna a imagem comprimida.
    """
    transformed = np.fft.fft2(image)  # aplica a fft
    fft_img = np.fft.fftshift(transformed, axes=(0, 1))  # centraliza o espectro

    # calcula o limiar
    limiar = np.percentile(np.abs(fft_img), percentile)
    # remove os coeficientes abaixo do limiar
    thresholded = np.where(np.abs(fft_img) < limiar, 0, fft_img)
    # retorna para o dominio do espaco
    compressed_image = np.abs(
        np.fft.ifft2(np.fft.ifftshift(thresholded, axes=(0, 1)))  # aplica a fft inversa
    ).astype(np.uint8)

    return compressed_image

def handle_results(
        original: np.ndarray,
        compressed: np.ndarray,
        save: bool,
        display: bool,
    ) -> bool:
    """
    Lida com o resultado da funcao de compressao.
    Salva e/ou exibe as imagens resultantes.
    """
    if save:
        save_images({"comprimido": compressed}, exercise=2)
    if display:
        results = {
            "original": original,
            "comprimido": compressed,
            "histograma": calc_histogram(original),
            "histograma_comprimido": calc_histogram(compressed)
        }
        print("\033[93mDisplaying results. Press 'q' to exit.\033[0m")
        return display_images(results, single=False)
    return True

def run(args):
    # Le imagem e aplica compressao
    image = cv.imread(get_image_path(args.image), cv.IMREAD_GRAYSCALE) # le imagem
    compressed_image = apply_compression(image, args.percentile)

    # Lida com o resultado
    print("\033[92mProcessing completed successfully.\033[0m") # verde
    handle_results(
        original=image,
        compressed=compressed_image,
        save=args.save,
        display=args.display
    )
    print("\033[92mDone.\033[0m") # verde 

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("-i", "--image", type=str, default="baboon_monocromatica.png", help="Image name (with extension) - default: baboon_monocromatica.png")
    args.add_argument("-p", "--percentile", type=float, default=95., help="Percentile for thresholding - default: 95")
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
