import os
import numpy as np
import cv2 as cv

directory = os.fsencode(os.path.join("t1", "images"))
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image = cv.imread(os.path.join("t1", "images", filename), cv.IMREAD_COLOR)
    print(f"{filename} - {image.shape}")
    print(f"Desvio padrão: {np.std(image)}")
    print()