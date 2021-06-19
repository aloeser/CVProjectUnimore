import cv2 as cv
import numpy as np
import random
import os

def random_retrieval_image(path='retrieval_dataset'):
    retrieval_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    retrieval_file = random.choice(retrieval_files)

    return cv.imread(retrieval_file)


def hough_circles(img,  hough_dp=1, minDist=20):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred_gray_img = cv.GaussianBlur(gray_img, (3, 3), 1, 1)
    #blurred_gray_img = cv.medianBlur(gray_img, 5)
    #blurred_gray_img = cv.GaussianBlur(gray_img, (5, 5), 2, 2)
    circles = cv.HoughCircles(blurred_gray_img, cv.HOUGH_GRADIENT, hough_dp, circles=1, param1=50, param2=30, minDist=minDist, minRadius=10, maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles.shape)
        print(circles)
    else:
        raise Exception("actually no circles found!")

    if circles is None or circles.shape == (4,1):
        raise Exception("no circles found")


    img_circles = img.copy()
    for (x,y,r) in circles[0, :]:
        cv.circle(img_circles, (x,y), r, (0, 200, 0), 2)

    cv.imshow("circles", img_circles)
    return circles

def create_masks(x_ctr, y_ctr, r, x=250, y=250):
    mask = np.zeros((y, x, 3), np.uint8)
    center = (x_ctr, y_ctr)
    # cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
    cv.circle(mask, center, r, color=(255, 255, 255), thickness=-1, lineType=8, shift=0) # thickness=-1 => filled circle
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    return mask

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

# returns 64x64x3 image with the coin on a black background
def extract_coin(img, x, y, r, do_morph=False):
    assert x >= r and y >= r and x+r < img.shape[1] and y+r < img.shape[0]
    excerpt = img[y-r:y+r, x-r:x+r, :]
    excerpt = cv.resize(excerpt, (64, 64), interpolation=cv.INTER_CUBIC)


    mask = create_masks(31, 31, 31, 64, 64)

    if do_morph:
        morph_size = 4
        mask = cv.morphologyEx(mask, cv.MORPH_DILATE, morph_kernel(morph_size))
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (morph_size, morph_size))
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)

    coin = cv.bitwise_and(excerpt, excerpt, mask=mask)

    result = np.zeros((64, 64, 3), np.uint8)
    result = cv.add(result, coin)
    return result

def extract_coins(img, circles):
    assert circles is not None and circles.shape != (4,1), "weird shape"
    coin_images = []
    for (x,y,r) in circles[0, :]:
        coin_images.append(extract_coin(img, x,y,r))

    return np.array(coin_images)

def get_coins_from_retrieval_image(retrieval_img):
    circles = hough_circles(retrieval_img)
    coin_images = extract_coins(retrieval_img, circles)
    return coin_images

def main():
    retrieval_img = random_retrieval_image()
    #retrieval_img = cv.imread('/home/aloeser/Downloads/opencv2.png')
    cv.imshow("original", retrieval_img)
    coin_images = get_coins_from_retrieval_image(retrieval_img)
    for index, image in enumerate(coin_images):
        cv.imshow(f"Coin Image {index}", image)
    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()