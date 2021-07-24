from __future__ import print_function
import cv2 as cv
import numpy as np
import random
random.seed(12345)


def thresh_callback(val):
    threshold = val
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the rotated rectangles and ellipses for each contour
    minRect = [None] * len(contours)
    minEllipse = [None] * len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)

    # Draw contours + rotated rects + ellipses
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(contours):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        # contour
        cv.drawContours(drawing, contours, i, color)
        # ellipse
        if c.shape[0] > 5:
            cv.ellipse(drawing, minEllipse[i], color, 2)
        # rotated rectangle
        box = cv.boxPoints(minRect[i])
        box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv.drawContours(drawing, [box], 0, color)
    cv.imshow('Contours', drawing)

def main():
    # read shifted image
    shifted_img = cv.imread('retrieval_dataset/0.png', cv.IMREAD_UNCHANGED)
    cv.namedWindow('Source')
    cv.imshow("shifted image", shifted_img)

    global src_gray

    # Convert image to gray and blur it
    src_gray = cv.cvtColor(shifted_img, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    # parameters
    max_thresh = 256
    thresh = 100  # initial threshold
    cv.createTrackbar('Canny Thresh:', 'Source', thresh, max_thresh, thresh_callback)
    thresh_callback(thresh)

    # window management
    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()