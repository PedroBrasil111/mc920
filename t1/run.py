import cv2 as cv
import os
import numpy as np
import argparse

IMAGE_FOLDER  = os.path.join(os.path.dirname(__file__), 'images')
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), 'results')

def save_images(image_dict: dict[str, np.ndarray], exercise: int) -> None:
    str_ex = str(exercise).zfill(2) # adiciona um zero a esquerda se ex < 10
    params = [cv.IMWRITE_PNG_STRATEGY, cv.IMWRITE_PNG_STRATEGY_DEFAULT]
    for title, image in image_dict.items():
        filename = f"ex{str_ex}-{title}.png"
        cv.imwrite(os.path.join(RESULT_FOLDER, filename), image, params)

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
    ksize = (21, 21)
    sigma = 3.5
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, ksize, sigma)
    result = cv.divide(gray_image, blurred_image, scale=255).astype(np.uint8)
    return {"esboco-lapis": result}

# 02
def ajuste_brilho(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE)
    results = {}
    alpha_list = [0.5, 1., 1.5, 2.5, 3.5,]
    image = np.divide(image, 255)  # Normalizacao para [0, 1]
    for i, alpha in enumerate(alpha_list):
        result = np.floor(np.power(image, 1/alpha) * 255).astype(np.uint8)
        results[f"ajuste-brilho-{i+1}"] = result
    return results

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
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE)
    plain_list = [0, 4, 7]
    results = {}
    for plain in plain_list:
        result = np.bitwise_and(image, 0b1 << plain) # image & (1 << plain)
        cv.normalize(result, result, 0, 255, cv.NORM_MINMAX) # inplace
        results[f"plano-bit-{plain}"] = result
    return results

# 07
def combinacao_imagens(image_name_1: str = "baboon_monocromatica.png", image_name_2: str = "butterfly.png") -> dict[str, np.ndarray]:
    image_1 = cv.imread(get_image_path(image_name_1), cv.IMREAD_GRAYSCALE)
    image_2 = cv.imread(get_image_path(image_name_2), cv.IMREAD_GRAYSCALE)
    weights = [
        (0.2, 0.8),
        (0.5, 0.5),
        (0.8, 0.2)
    ]
    results = {}
    for i, (w1, w2) in enumerate(weights):
        result = (np.add(np.multiply(w1, image_1), np.multiply(w2, image_2))).astype(np.uint8)
        results[f"combinacao-{i+1}"] = result
    return results

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
    if image_names and exercise_number == 7:  # Special case for exercise 7 (requires two images)
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
    if not (args.d or args.s):
        print("\033[91mNo action specified. Use -s to save images and/or -d to display them.\033[0m")
        return

    for exercise_number in args.e:
        print(f"\033[94mProcessing exercise {exercise_number}...\033[0m")
        try:
            process_and_handle_exercise(
                exercises[exercise_number],
                exercise_number,
                args.i,
                args.s,
                args.d
            )
        except Exception as e:
            print(f"\033[91mError processing exercise {exercise_number}: {e}\033[0m")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images with various exercises.')
    parser.add_argument('-i', nargs='+', type=str, help='List of image names (or all)')
    parser.add_argument('-e', nargs='+', type=int, help='List of exercise numbers (1-10) - default: all')
    parser.add_argument('-s', action='store_true', help='Save processed images')
    parser.add_argument('-d', action='store_true', help='Display processed images')
    args = parser.parse_args()

    if args.i and args.i[0].lower() == 'all':
        args.i = os.listdir(IMAGE_FOLDER)

    # All exercises by default
    args.e = range(1, 11) if not args.e else [ex for ex in args.e if 1 <= ex <= 10]
    main(args)