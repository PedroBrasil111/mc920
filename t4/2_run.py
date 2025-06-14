from argparse import ArgumentParser
import cv2 as cv
from numpy import ndarray
import os
import time

from transformation_methods import (
    spatial_transformation, intensity_interpolation
)
from helper_functions import (
    get_image_path, save_images, display_images,
)

INTERPOLATION_METHODS = ["nn", "bilinear", "bicubic", "lagrange"]

def handle_args(args):
    """
    Trata os argumentos da linha de comando.
    """
    if not args.image:
        args.image = "baboon_colorida.png"
    else:
        if not os.path.exists(get_image_path(args.image)):
            raise ValueError(f"Image not found: {args.image}")
    if args.angle is None:
        args.angle = 0.
    if args.scale is None:
        args.scale = 1.
    if args.scale <= 0:
        raise ValueError("Scale factor must be greater than 0.")
    if args.methods is None or (type(args) == str and args.methods.lower() == 'all'):
        args.methods = INTERPOLATION_METHODS
    if not args.dimensions:
        args.dimensions = [None, None]
    else:
        args.methods = [m.lower() for m in args.methods]
        for method in args.methods:
            if method not in INTERPOLATION_METHODS:
                raise ValueError(f"Invalid method: {method}. Choose from {INTERPOLATION_METHODS}.")
    if not (args.save or args.display):
        raise ValueError("No action specified. Use -s to save images and/or -d to display them.")

def handle_results(
        results: dict[str, ndarray],
        exercise: str,
        save: bool,
        display: bool
    ) -> bool:
    """
    Salva e/ou exibe as imagens resultantes.
    """
    if save:
        save_images(results, exercise)
        print("\033[92mResults saved\033[0m")
    if display:
        print("\033[93mDisplaying results. Press any key to continue or 'q' to exit.\033[0m")
        return display_images(results, single=True)
    return True

def run(args):
    img = cv.imread(get_image_path(args.image), cv.IMREAD_COLOR)
    new_coords = spatial_transformation(
        img, args.scale, args.angle, args.dimensions[0], args.dimensions[1]
    )
    results = {}
    for method in args.methods:
        t_start = time.time()
        print(f"\033[94mCalculating {method} interpolation...\033[0m")
        out_img = intensity_interpolation(img, new_coords, method)
        print(f"\033[92m{method} interpolation completed in {time.time() - t_start:.2f} seconds\033[0m")
        results[method] = out_img
    handle_results(results, "02", args.save, args.display)

if __name__ == "__main__":
    args = ArgumentParser("Geometric Transformation (scale and rotation) Methods")
    args.add_argument("-i", "--image", type=str, help="Image name (with extension) - default: baboon_colorida.png")
    args.add_argument("-m", "--methods", type=str, nargs="+", help="List of methods for interpolation (nn, bilinear, bicubic, lagrange), or 'all' - default: all methods")
    args.add_argument("-a", "--angle", type=float, help="Rotation angle in degrees (positive for clockwise) - default: 0.0")
    args.add_argument("-D", "--dimensions", type=int, nargs="+", help="Output image dimensions (width height) - default: the dimensions for which every point in the input image is transformed")
    args.add_argument("-e", "--scale", type=float, help="Scale factor (1.0 for no scaling) - default: 1.0")
    args.add_argument("-s", "--save", action="store_true", help="Save the images")
    args.add_argument("-d", "--display", action="store_true", help="Display the images")
    args = args.parse_args()

    # Trata os argumentos da linha de comando
    try:
        handle_args(args)
    except ValueError as e:
        print(f"\033[91mError: {e}\033[0m")
        exit(1)

    # Executa o processamento
    run(args)