import cv2 as cv
import os
import numpy as np
import argparse

IMAGE_FOLDER  = os.path.join(os.path.dirname(__file__), 'images')
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), 'results')

def save_images(image_dict: dict[str, np.ndarray], exercise: int) -> None:
    for title, image in image_dict.items():
        filename = f"ex{exercise}-{title}.png"
        cv.imwrite(os.path.join(RESULT_FOLDER, filename), image)

def display_images(image_dict: dict[str, np.ndarray]) -> None:
    for title, image in image_dict.items():
        cv.imshow(title, image)
        cv.waitKey(0)
        cv.destroyAllWindows()

def get_image_path(image_name: str) -> str:
    return os.path.join(IMAGE_FOLDER, image_name)

# 01
def esboco_lapis(image_name: str = "watch.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_COLOR)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (21, 21), 0)
    result = cv.divide(gray_image, blurred_image, gray_image, scale=256)
    return {"Esboco lapis": result}

# 02
def ajuste_brilho(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    return None

# 03
def mosaico(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    return None

# 04
def alteracao_cores(image_name: str = "watch.png") -> dict[str, np.ndarray]:
    return None

# 05
def transformacao_imagens_coloridas(image_name: str = "watch.png") -> dict[str, np.ndarray]:
    return None

# 06
def plano_bits(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    return None

# 07
def combinacao_imagens(image_name: str = "baboon_monocromatica.png", image_name_2: str = "butterfly.png") -> dict[str, np.ndarray]:
    return None

# 08
def transformacao_intensidade(image_name: str = "city.png") -> dict[str, np.ndarray]:
    return None

# 09
def quantizacao_imagens(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    return None

# 10 
def filtragem_imagens(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    return None

def process_and_handle_exercise(
        exercise_function: callable,
        exercise_number: int,
        image_names: list[str],
        save: bool,
        display: bool
        ) -> None:
    if exercise_number == 7:  # Special case for exercise 7 (requires two images)
        for i in range(0, len(image_names), 2):
            image_name_1 = image_names[i]
            if i + 1 < len(image_names):
                image_name_2 = image_names[i + 1]
                result = exercise_function(image_name_1, image_name_2)
            else:
                result = exercise_function(image_name_1)
            handle_result(result, exercise_number, save, display)
    else:  # Exercises that require one image
        if image_names:
            for image_name in image_names:
                result = exercise_function(image_name)
                handle_result(result, exercise_number, save, display)
        else:  # Default image name
            result = exercise_function()
            handle_result(result, exercise_number, save, display)

def handle_result(
        result: dict[str, np.ndarray],
        exercise_number: int,
        save: bool,
        display: bool
        ) -> None:
    if result:
        if save:
            save_images(result, exercise_number)
        if display:
            display_images(result)

def main(args: argparse.Namespace) -> None:
    exercises = {
        1: esboco_lapis,
        2: ajuste_brilho,
        3: mosaico,
        4: alteracao_cores,
        5: transformacao_imagens_coloridas,
        6: plano_bits,
        7: combinacao_imagens,
        8: transformacao_intensidade,
        9: quantizacao_imagens,
        10: filtragem_imagens,
    }

    for exercise_number in args.e:
        print(f"\033[94mProcessing exercise {exercise_number}...\033[0m")
        process_and_handle_exercise(
            exercises[exercise_number],
            exercise_number,
            args.i,
            args.s,
            args.d
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images with various exercises.')
    parser.add_argument('-i', nargs='+', type=str, help='List of image names')
    parser.add_argument('-e', nargs='+', type=int, help='List of exercise numbers (1-10) - default: all')
    parser.add_argument('-s', action='store_true', help='Save processed images')
    parser.add_argument('-d', action='store_true', help='Display processed images')
    args = parser.parse_args()

    # All exercises by default
    args.e = range(1, 11) if not args.e else [ex for ex in args.e if 1 <= ex <= 10]
    main(args)