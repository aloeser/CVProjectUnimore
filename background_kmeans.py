import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2 as cv

img = cv.imread('')
img = img[:,:,::-1]
plt.imshow(img)
plt.waitforbuttonpress()
plt.close()

w, h, c = img.shape
data = np.reshape(img, newshape=(w * h, 3))

initial_centers = np.array([data[0], data[len(data)//2]])
labels = KMeans(2).fit_predict(data)
labels = np.reshape(labels, (h, w)).astype(np.uint8)
if (labels == 1).sum() < (labels == 0).sum():
    labels = 1 - labels 
kernel = np.ones((3,3),np.uint8)
labels = cv.morphologyEx(labels, cv.MORPH_CLOSE, kernel)
plt.imshow(labels, cmap='gray')
plt.waitforbuttonpress()
plt.close()

img_background = img.copy()
img_background[labels == 0] = np.array([128, 128, 128], dtype=np.uint8)
#img_background[labels == 0] = img.mean(axis=(0,1))
plt.imshow(img_background)
plt.waitforbuttonpress()
plt.close()
