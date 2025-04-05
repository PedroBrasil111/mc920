import cv2 as cv
from numpy import ndarray
import os

IMAGE_FOLDER  = os.path.join(os.path.dirname(__file__), 'images')
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), 'results')

def save_images(image_dict: dict[str, ndarray], exercise: int) -> None:
    str_ex = str(exercise).zfill(2) # adiciona um zero a esquerda se ex < 10
    params = [cv.IMWRITE_PNG_STRATEGY, cv.IMWRITE_PNG_STRATEGY_DEFAULT]
    for title, image in image_dict.items():
        filename = f"ex{str_ex}-{title}.png"
        cv.imwrite(os.path.join(RESULT_FOLDER, filename), image, params)

def display_images(image_dict: dict[str, ndarray]) -> bool:
    for title, image in image_dict.items():
        cv.imshow(title, image)
        key = cv.waitKey(0)

        cv.destroyAllWindows()

        if key == ord('q'):
            print("\033[91mExecution interrupted by user (pressed 'q').\033[0m")
            exit()
        elif key == ord('n'):
            print("\033[93mExercise interrupted by user (pressed 'n').\033[0m")
            return False
    return True

def get_image_path(image_name: str) -> str:
    return os.path.join(IMAGE_FOLDER, image_name)
