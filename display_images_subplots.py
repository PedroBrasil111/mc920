import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import ndarray
import tempfile

def display_images_subplots(image_dict: dict[str, ndarray], dim: tuple=None) -> bool:
    """
    Desenvolvido com auxílio de IA.
    Salva uma figura com subplots (2, n/2) de imagens usando matplotlib,
    exibe a figura salva com OpenCV e espera por uma tecla.
    Retorna False se 'n' for pressionado, True caso contrario.
    """

    n = len(image_dict)
    if n == 0:
        print("\033[91mNo images to display.\033[0m")
        return True

    # Create matplotlib figure
    if not dim:
        dim = (2, (n + 1) // 2)
    fig, axes = plt.subplots(dim[0], dim[1], figsize=(10, 6))
    axes = axes.flatten()

    items = list(image_dict.items())
    items.sort(key=lambda x: x[0].lower() != "original")

    for ax, (title, img) in zip(axes, items):
        title = title.capitalize().replace('_', ' ')
        if img.ndim == 2:  # grayscale
            ax.imshow(img, cmap='gray')
        else:  # color
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    # Hide any unused axes
    for ax in axes[len(image_dict):]:
        ax.axis('off')

    # Adjust layout to avoid overlapping
    plt.tight_layout()
    #ax.set_aspect('auto')
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.05)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        temp_path = tmpfile.name
    plt.savefig(temp_path, dpi=150)
    plt.close(fig)

    # Read with OpenCV and show
    img_cv = cv.imread(temp_path)
    os.remove(temp_path)  # clean up temp file immediately

    if img_cv is None:
        print("\033[91mFailed to load the saved figure.\033[0m")
        return True

    cv.imshow('Image Set', img_cv)
    key = cv.waitKey(0)
    cv.destroyAllWindows()

    if key == ord('q'):
        print("\033[91mExecution interrupted by user (pressed 'q').\033[0m")
        exit()
    elif key == ord('n'):
        return False
    return True
