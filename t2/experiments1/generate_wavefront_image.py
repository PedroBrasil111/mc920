import numpy as np
import cv2 as cv

def generate_wavefront_image(size, ratio):
    """
    Generates a wavefront image of the specified size and ratio.
    The image is generated using a mathematical formula based on the coordinates of each pixel.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    max_val = 2*(size//ratio)**2

    for r in range(size):
        for c in range(size):
            img[r, c] = np.floor((r**2//ratio + c**2//ratio) / max_val * 255)

    return img

for size in np.power(2, range(4, 12)):
    img = generate_wavefront_image(size, 1)
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.imwrite(f"images/wavefront_{size}.png", img)
    cv.destroyAllWindows()

exit()

size = 512
img = np.zeros((size, size), dtype=np.uint8)
ratio = 2

max_val = 2*(size//ratio)**2

for r in range(size):
    for c in range(size):
        img[r, c] = np.floor((r**2//ratio + c**2//ratio) / max_val * 255) if r**2 + c**2 != 0 else 255

cv.imshow("image", img)
cv.waitKey(0)

cv.imwrite(f"images/wavefront_{size}.png", img)
cv.destroyAllWindows()

size = 512
img = np.zeros((size, size), dtype=np.uint8)
ratio = 2

max_val = 2*(size//ratio)**2

for r in range(size):
    for c in range(size):
        img[r, c] = np.floor((abs(size//2 - r)**2//ratio + abs(size//2 - c)**2//ratio) / max_val * 255) if r**2 + c**2 != 0 else 255

cv.imshow("image", img)
cv.waitKey(0)

cv.imwrite(f"images/wavefront_{size}.png", img)
cv.destroyAllWindows()