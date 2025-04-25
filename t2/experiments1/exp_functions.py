import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend


import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def display_image_grid(
        image_dict: dict[str, np.ndarray], grid_dimensions: tuple[int]
        ) -> np.ndarray:
    
    def resize_image(image, max_width, max_height):
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h) 
        new_size = (int(w * scale), int(h * scale))
        return cv.resize(image, new_size)

    images = list(image_dict.values())
    titles = list(image_dict.keys())
    rows, cols = grid_dimensions

    if len(images) > rows * cols:
        raise ValueError(f"Number of images ({len(images)}) exceeds grid dimensions ({rows*cols})")

    # Set max screen size (adjust as needed)
    screen_width, screen_height = 1600, 900

    # Resize images
    resized_images = []
    max_img_width = screen_width // cols
    max_img_height = screen_height // rows

    for i, img in enumerate(images):
        # Resize image
        resized_img = resize_image(img, max_img_width, max_img_height)
        resized_images.append(resized_img)

    # Pad with blank images if necessary
    blank_image = np.zeros_like(resized_images[0])  # Use resized blank images
    while len(resized_images) < rows * cols:
        resized_images.append(blank_image)

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(screen_width / 100, screen_height / 100))

    # Iterate over the axes and images
    for i, ax in enumerate(axes.flat):
        if i < len(resized_images):
            ax.imshow(resized_images[i], cmap='gray' if len(resized_images[i].shape) == 2 else None)
            ax.set_title(titles[i], fontsize=10, color='black', loc='center')
        ax.axis('off')  # Turn off axis

    plt.tight_layout()
    plt.savefig("experiments_results/grid_image1.png", dpi=600, bbox_inches='tight')

