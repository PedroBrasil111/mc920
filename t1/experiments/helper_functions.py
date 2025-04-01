import numpy as np
import cv2 as cv
import os

def get_image_path(image_name: str) -> str:
    image_path = os.path.join(os.path.dirname(__file__), '..', 'images')
    return os.path.join(image_path, image_name)

def scale_conversion(image: np.ndarray, min_val, max_val) -> np.ndarray:
    min_img = image.min(axis=(0, 1), keepdims=True)
    max_img = image.max(axis=(0, 1), keepdims=True)

    scaled = (image - min_img) / (max_img - min_img) * (max_val - min_val) - min_val
    return np.floor(scaled).astype(np.uint8)

def resize_image(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h) 
    new_size = (int(w * scale), int(h * scale))
    return cv.resize(image, new_size)

def add_title(image, title):
    font = cv.FONT_HERSHEY_TRIPLEX
    font_scale = 1.5
    color = (0, 0, 0)  # White text
    thickness = 1

    # Get text size
    text_size = cv.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2  # Center the text horizontally
    text_y = 50  # Set vertical position of title

    # Put the title on the image
    cv.putText(image, title, (text_x, text_y), font, font_scale, color, thickness, lineType=cv.LINE_AA)
    
    return image

def display_image_grid(
        image_dict: dict[str, np.ndarray], grid_dimensions: tuple[int]
        ) -> np.ndarray:
    images = list(image_dict.values())
    titles = list(image_dict.keys())
    rows, cols = grid_dimensions

    if len(images) > rows * cols:
        raise ValueError(f"Number of images ({len(images)}) exceeds grid dimensions ({rows*cols})")

    # Set max screen size (adjust as needed)
    screen_width, screen_height = 2560, 1600 # 1600, 900

    # Resize images
    resized_images = []
    max_img_width = screen_width // cols
    max_img_height = screen_height // rows

    for i, img in enumerate(images):
        # Resize image
        resized_img = resize_image(img, max_img_width, max_img_height)
        # Add title to the image
        resized_img_with_title = add_title(resized_img, titles[i])
        resized_images.append(resized_img_with_title)

    # Pad with blank images if necessary
    blank_image = np.zeros_like(resized_images[0])  # Use resized blank images
    while len(resized_images) < rows * cols:
        resized_images.append(blank_image)

    # Stack images into a grid
    image_grid = np.vstack([
        np.hstack(resized_images[row * cols:(row + 1) * cols])
        for row in range(rows)
    ])

    cv.imshow("Images", image_grid)
    cv.waitKey(0)

    return image_grid