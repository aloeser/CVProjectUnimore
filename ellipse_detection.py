import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def center_crop(img, new_h, new_w):
    old_h, old_w = img.shape[:2]
    off_w = (old_w - new_w) // 2
    off_h = (old_h - new_h) // 2
    return img[off_h:off_h+new_h, off_w:off_w+new_w, :]


def find_edges(img):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    t, _ = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    std = img.std()
    edges = cv.Canny(img, min(255, t + 2 * std), max(0, t - 2 * std))
    return edges


def detect_circles(img, edges, n_circles=5, mode='both'):
    assert(mode in ['both', 'draw', 'mask'])
    img = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w))
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 1, 
                              param1=1, param2=1,
                              minRadius=h//8, maxRadius=h//2)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        n_circles = min(n_circles, len(circles[0]))
        for i in circles[0, :n_circles]:
            center = (int(i[0]), int(i[1]))
            radius = int(i[2])
            if mode in ['both', 'draw']:
                cv.circle(img, center, radius, (255, 0, 255), 2)
                cv.circle(img, center, 1, (0, 100, 100), 2)
            if mode in ['both', 'mask']:
                cv.circle(mask, center, radius, 255, -1)
    return img, mask
    

def morph_kernel(size):
    if size == 3:
        return np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], dtype=np.uint8)
    elif size == 4:
        return np.array([[1, 0, 0, 1],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [1, 0, 0, 1]], dtype=np.uint8)
    elif size == 5:
        return np.array([[1, 0, 0, 0, 1],
                        [0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0],
                        [1, 0, 0, 0, 1]], dtype=np.uint8)
    else:
        return cv.getStructuringElement(cv.MORPH_RECT, (size, size))


def segment_coin(img, blur_size=7, morph_size=4):
    img = cv.GaussianBlur(img, (blur_size, blur_size), 0, 0)
    _, mask = detect_circles(img, find_edges(img))
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, morph_kernel(morph_size))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (morph_size, morph_size))        
    mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)
    return mask


if __name__ == '__main__':
    orig = cv.imread('')
    orig = orig[:,:,::-1]
    plt.imshow(orig)
    plt.waitforbuttonpress()
    plt.close()

    w, h, c = orig.shape
    orig = center_crop(orig, min(h,w), min(h,w))
    w, h, c = orig.shape
    img = cv.GaussianBlur(orig, (7, 7), 0, 0)
    img_circles, _ = detect_circles(img, find_edges(img))
    mask = segment_coin(orig).astype(np.bool)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_circles, cmap='gray')
    ax2.imshow(mask, cmap='gray')
    plt.waitforbuttonpress()
    plt.close()

    out = np.zeros((h, w, c), dtype=np.uint8)
    out[mask] = orig[mask]
    out = cv.GaussianBlur(out, (5, 5), 0, 0)
    out = cv.resize(out, (64,64), interpolation = cv.INTER_AREA)
    plt.imshow(out, cmap='gray')
    plt.waitforbuttonpress()
    plt.close()
