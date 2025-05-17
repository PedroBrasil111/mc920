import numpy as np
import cv2 as cv

size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

for s in size:
    img = np.random.randint(0, 256, (s, s, 3), dtype=np.uint8)
    cv.imwrite(f"images/noise_{s}_RGB.png", img)
    img = np.random.randint(0, 256, (s, s), dtype=np.uint8)
    cv.imwrite(f"images/noise_{s}.png", img)