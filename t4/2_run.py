from argparse import ArgumentParser
import numpy as np
import os
import cv2 as cv

from transformation_methods import *
from helper_functions import (
    get_image_path, save_images, display_images,
)

##############

def translation_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=np.float32)

def scale_matrix(scale: float) -> np.ndarray:
    return np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, 1]], dtype=np.float32)

def rotation_matrix(angle: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle)
    return np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                     [np.sin(angle_rad),  np.cos(angle_rad), 0],
                     [0, 0, 1]], dtype=np.float32)

def calculate_output_dimensions(w: int, h: int, angle: float, scale: float) -> tuple[int, int]:
    angle_rad = np.deg2rad(angle)
    cos_angle = np.abs(np.cos(angle_rad))
    sin_angle = np.abs(np.sin(angle_rad))

    new_w = int((w * cos_angle + h * sin_angle) * scale)
    new_h = int((w * sin_angle + h * cos_angle) * scale)

    return new_w, new_h

def output_coordinates(out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
    j_grid, i_grid = np.meshgrid(np.arange(out_w), np.arange(out_h)) # X, Y
    ones = np.ones_like(j_grid) # W
    grid = np.stack([j_grid, i_grid, ones], axis=0).reshape(3, -1) # shape: (3, h*w)
    return grid

def homogeneous_coordinates(img: np.ndarray, scale: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    new_w, new_h = calculate_output_dimensions(w, h, angle, scale)
    cx, cy = w / 2, h / 2
    nx, ny = new_w / 2, new_h / 2

    trans_mat_center = translation_matrix(-cx, -cy)
    trans_mat_back = translation_matrix(nx, ny)
    scale_mat = scale_matrix(scale)
    rotation_mat = rotation_matrix(angle)
    transform = rotation_mat @ scale_mat

    transform_mat = trans_mat_back @ transform @ trans_mat_center
    inv_transform = np.linalg.inv(transform_mat)

    coords_out = output_coordinates(new_w, new_h)

    coords_in = inv_transform @ coords_out
    coords_in = coords_in[:2] # sem a linha homogênea
    coords_in = coords_in.reshape(2, new_h, new_w)

    return coords_in, transform_mat

def nearest_neighbor_interpolation(img: np.ndarray, out_position: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    output_img = np.zeros((out_position.shape[1], out_position.shape[2], 3), dtype=np.float32)
    print(output_img.shape)

    for r_out in range(out_position.shape[1]):
        for c_out in range(out_position.shape[2]):
            x_out, y_out = out_position[:, r_out, c_out]
            x_in, y_in = int(round(x_out)), int(round(y_out))
            if 0 <= x_in < w and 0 <= y_in < h:
                output_img[r_out, c_out] = img[y_in, x_in]

    return output_img.astype(np.uint8)

def bilinear_interpolation(img: np.ndarray, out_position: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    output_img = np.zeros((out_position.shape[1], out_position.shape[2], 3), dtype=np.float32)
    print(output_img.shape)

    for r_out in range(out_position.shape[1]):
        for c_out in range(out_position.shape[2]):
            x_out, y_out = out_position[:, r_out, c_out]
            x_in, y_in = int(np.floor(x_out)), int(np.floor(y_out))
            dx, dy = x_out - x_in, y_out - y_in

            if 0 <= x_in < w-1 and 0 <= y_in < h-1:
                val = (1 - dx) * (1 - dy) * img[y_in, x_in] + \
                        dx * (1 - dy) * img[y_in, x_in + 1] + \
                        (1 - dx) * dy * img[y_in + 1, x_in] + \
                        dx * dy * img[y_in + 1, x_in + 1]

                output_img[r_out, c_out] = np.clip(val, 0, 255)

    return output_img.astype(np.uint8)

def ramp(x: float):
    return x if x > 0 else 0

def cubic_bspline(s: float):
    return (ramp(s + 2)**3 - 4*ramp(s + 1)**3 + 6*ramp(s)**3 - 4*ramp(s - 1)**3) / 6

def bicubic_interpolation(img: np.ndarray, out_position: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    output_img = np.zeros((out_position.shape[1], out_position.shape[2], 3), dtype=np.float32)

    for r_out in range(out_position.shape[1]):
        for c_out in range(out_position.shape[2]):
            x_out, y_out = out_position[:, r_out, c_out]
            x_in, y_in = int(np.floor(x_out)), int(np.floor(y_out))
            dx, dy = x_out - x_in, y_out - y_in

            val_out = 0
            if 1 <= x_in < w - 2 and 1 <= y_in < h - 2:
                for n in range(-1, 3):
                    for m in range(-1, 3):
                        if 0 <= x_in + m < w and 0 <= y_in + n < h:
                            val_in = img[y_in + n, x_in + m]
                            val_out += val_in * cubic_bspline(m - dx) * cubic_bspline(dy - n)

            output_img[r_out, c_out] = np.clip(val_out, 0, 255)

    return output_img.astype(np.uint8)

def polynomial(d, vals):
    num1 = (-d * (d-1) * (d-2) * vals[0]) / 6
    num2 = ((d+1) * (d-1) * (d-2) * vals[1]) / 2
    num3 = (-d * (d+1) * (d-2) * vals[2]) / 2
    num4 = (-d * (d+1) * (d-1) * vals[3]) / 6
    return num1 + num2 + num3 + num4

def L(img, dx, n, x, y):
    vals = [img[y + n - 2, x + i] for i in range(-1, 3)]
    return polynomial(dx, vals)

def lagrange_polynomial(img, x, y, dx, dy):
    vals = [L(img, dx, n, x, y) for n in range(1, 5)]
    return polynomial(dy, vals)

def lagrange_interpolation(img: np.ndarray, out_position: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    output_img = np.zeros((out_position.shape[1], out_position.shape[2], 3), dtype=np.float32)
    print(output_img.shape)

    for r_out in range(out_position.shape[1]):
        for c_out in range(out_position.shape[2]):
            x_out, y_out = out_position[:, r_out, c_out]
            x_in, y_in = int(np.floor(x_out)), int(np.floor(y_out))
            dx, dy = x_out - x_in, y_out - y_in

            val_out = 0
            if 1 <= x_in < w - 2 and 1 <= y_in < h - 2:
                val_out = lagrange_polynomial(img, x_in, y_in, dx, dy)
            
            output_img[r_out, c_out] = np.clip(val_out, 0, 255)

    return output_img.astype(np.uint8)

INTERPOLATE_FUNC = {
    "nn": nearest_neighbor_interpolation,
    "bilinear": bilinear_interpolation,
    "bicubic": bicubic_interpolation,
    "lagrange": lagrange_interpolation,
}

def interpolate_image(img: np.ndarray, out_position: np.ndarray, method: str) -> np.ndarray:
    h, w = img.shape[:2]
    output_img = np.zeros_like(img, dtype=np.float32)

    for r_out in range(out_position.shape[1]):
        for c_out in range(out_position.shape[2]):
            x_out, y_out = out_position[:, r_out, c_out]
            x_in, y_in = int(np.floor(x_out)), int(np.floor(y_out))
            dx, dy = x_out - x_in, y_out - y_in

    return output_img.astype(np.uint8)



##############

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
        args.angle = 0.0
    if args.scale is None:
        args.scale = 1.0
    if args.dimension is None:
        args.dimension = [512, 512]
    if args.method is None:
        args.method = "nn"
    else:
        valid_methods = ["nn", "bilinear", "bicubic", "lagrange"]
        if args.method not in valid_methods:
            raise ValueError(f"Invalid method: {args.method}. Choose from {valid_methods}.")
    if not (args.save or args.display):
        raise ValueError("No action specified. Use -s to save images and/or -d to display them.")

def handle_results(
        results: dict[str, np.ndarray],
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
    import time

    img = cv.imread(get_image_path("ellipse.png"), cv.IMREAD_COLOR)
    angle = -30
    scale = 1.5
    print("Building transformation matrix...")
    transformed_coords, transf_matrix = homogeneous_coordinates(img, scale, angle)

    t = time.time()
    print("Applying Bicubic interpolation...")
    bicubic = bicubic_interpolation(img, transformed_coords)
    print(f"Time: {time.time() - t}\n")
    cv.imshow("Bicubic Interpolation", bicubic)
    cv.waitKey(0)

    t = time.time()
    print("Applying Lagrange interpolation...")
    lagrange = lagrange_interpolation(img, transformed_coords)
    print(f"Time: {time.time() - t}\n")
    cv.imshow("Lagrange Interpolation", lagrange)
    cv.waitKey(0)

    t = time.time()
    print("Applying Bilinear interpolation...")
    bilinear = bilinear_interpolation(img, transformed_coords)
    print(f"Time: {time.time() - t}\n")
    cv.imshow("Bilinear Interpolation", bilinear)
    cv.waitKey(0)

    t = time.time()
    print("Applying NN interpolation...")
    nn = nearest_neighbor_interpolation(img, transformed_coords)
    print(f"Time: {time.time() - t}\n")
    cv.imshow("Nearest Neighbor Interpolation", nn)
    cv.waitKey(0)

    cv.destroyAllWindows()

if __name__ == "__main__":

    run(args=None)
    exit(0)

    args = ArgumentParser("Geometric Transformation Methods")
    args.add_argument("-i", "--image", type=str, help="List of image names (with extension) - default: objetos3.png")
    args.add_argument("-a", "--angle", type=float, help="Rotation angle in degrees (positive for clockwise)")
    args.add_argument("-e", "--scale", type=float, help="Scale factor (1.0 for no scaling)")
    args.add_argument("-D", "--dimension", nargs="+", type=int, help="Output image dimensions (width height)")
    args.add_argument("-m", "--method", action="store_true", help="Use method for geometric interpolation -- one of [nn, bilinear, bicubic, lagrange]")
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