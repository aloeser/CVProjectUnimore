import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

(h, w) = img.shape[:2]
(cX, cY) = (w // 2, h // 2)
M = cv.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv.warpAffine(img, M, (w, h), borderMode=cv.BORDER_REPLICATE)
plt.imshow(rotated[:,:,::-1])
plt.waitforbuttonpress()
plt.close()