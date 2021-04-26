import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('')
plt.imshow(img[:,:,::-1])
plt.waitforbuttonpress()
plt.close()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)[1]
if (binary == 255).sum() < (binary == 0).sum():
    binary = 255 - binary 
kernel = np.ones((3,3),np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
plt.imshow(binary, cmap='gray')
plt.waitforbuttonpress()
plt.close()
