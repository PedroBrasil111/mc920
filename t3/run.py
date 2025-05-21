from argparse import ArgumentParser
import cv2 as cv
from numpy import ndarray
import os

from helper_functions import (
    get_image_path, get_all_images,
    save_images, display_images,
)
from alignment import (
    rotate_image, run_angle_detection,
)

def handle_args(args):
    """
    Trata os argumentos da linha de comando.
    """
    if not args.image or args.image[0] == "all":
        args.image = get_all_images()
    else:
        missing = [img for img in args.image if not os.path.exists(get_image_path(img))]
        if missing:
            raise ValueError(f"Image(s) not found: {', '.join(missing)}")
    valid_modes = ["hough", "projection"]
    if not args.mode:
        args.mode = valid_modes
    else:
        invalid_modes = [m for m in args.mode if m not in valid_modes]
        if invalid_modes:
            raise ValueError(f"Invalid mode(s): {', '.join(invalid_modes)}. Use 'hough' and/or 'projection'.")
    if not (args.save or args.display):
        raise ValueError("No action specified. Use -s to save images and/or -d to display them.")

def handle_results(
        results: dict[str, ndarray],
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
        print("\033[92mResults saved\033[0m")
    if display:
        print("\033[93mDisplaying results. Press 'q' to exit or 'n' to go to the next mode (if there is one).\033[0m")
        return display_images(results, single=True)
    return True

def run(args) -> None:
    for mode in args.mode:
        str_mode = "Hough transform" if mode == "hough" else "Horizontal projection"
        print(f"Processing mode: {str_mode}") # verde
        results = {}
        # Le imagem, processa e guarda resultado
        for img_name in args.image:
            str_img = img_name.split(".")[0]
            img = cv.imread(get_image_path(img_name), cv.IMREAD_GRAYSCALE)
            angle = run_angle_detection(img, mode)
            rotated = rotate_image(img, angle, remove_border=True)
            print(f"- Image {str_img} was rotated by {angle:.2f} degrees")
            results[f"aligned_{str_img}"] = rotated
        handle_results(results, mode, args.save, args.display)
    print("\033[92mDone.\033[0m") # verde

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("-i", "--image", nargs='+', type=str, help="List of image names, or 'all' (with extension) - default: all")
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
