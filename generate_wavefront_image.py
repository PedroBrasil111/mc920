import numpy as np
import cv2 as cv

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