from helper_functions import (
    get_image_path, get_all_images,
    save_images, display_images,
)
from alignment_methods import (
    rotate_image, add_histogram,
    horizontal_projection_align, hough_transform_align,
)
from argparse import ArgumentParser
import numpy as np
import cv2 as cv
import os

ANGLE_FUNCTIONS = {
    "hough": hough_transform_align,
    "projection": horizontal_projection_align
}

def handle_args(args):
    """
    Trata os argumentos da linha de comando.
    """
    if args.image:
        for img_name in args.image:
            if not os.path.exists(get_image_path(img_name)):
                raise ValueError(f"Image {args.image} not found.")
    else:
        args.image = get_all_images()
    if args.mode:
        for mode in args.mode:
            if mode not in ["hough", "projection"]:
                raise ValueError(f"Invalid mode '{mode}'. Use 'hough' or 'projection'.")
    else:
        args.mode = ["hough", "projection"]
    if not args.save and not args.display:
        raise ValueError("No action specified. Use -s to save images and/or -d to display them.")

def handle_results(
        results: dict[str, np.ndarray],
        mode: str,
        save: bool,
        display: bool
    ) -> bool:
    """
    Lida com o resultado da funcao de compressao.
    Salva e/ou exibe as imagens resultantes.
    """
    if save:
        save_images(results, exercise=mode)
    if display:
        print("\033[93mDisplaying results. Press 'q' to exit.\033[0m")
        return display_images(results, single=True)
    return True

def run(args) -> None:
    for mode in args.mode:
        print(f"Processing mode: {mode}") # verde
        results = {}
        for img_name in args.image:
            img = cv.imread(get_image_path(img_name), cv.IMREAD_GRAYSCALE) # le imagem
            angle = ANGLE_FUNCTIONS[mode](img)
            rotated = rotate_image(img, angle)
            results[f"aligned_{img_name}"] = rotated
        handle_results(results, mode, args.save, args.display)
    print("\033[92mDone.\033[0m") # verde

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("-i", "--image", nargs='+', type=str, help="List of image names (with extension) - default: all images")
    args.add_argument("-m", "--mode", nargs='+', type=str, help="Mode of alignment - 'hough' and/or 'projection' - default: both")
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
