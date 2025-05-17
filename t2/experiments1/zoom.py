import cv2 as cv
import os

name = "fuji_cinza"
for file in os.listdir(name):
    if file.endswith(".png"):
        imgname = file[:-4]
        print(f"Processing {imgname}")
        img = cv.imread(f"{name}/{imgname}.png", cv.IMREAD_GRAYSCALE)
        nrows, ncols = img.shape
        zoomed = img[nrows//2 - 50:nrows//2 + 50, ncols-100:]
        cv.imshow(imgname, zoomed)
original = cv.imread(f"../images/{name}.png", cv.IMREAD_GRAYSCALE)
zoomed_original = original[nrows//2 - 50:nrows//2 + 50, ncols-100:]
cv.imshow("original", zoomed_original)

cv.waitKey(0)
cv.destroyAllWindows()