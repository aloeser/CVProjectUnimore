import cv2 as cv
import numpy as np

def corr_pic(shifted_img, blur_strgth):
    # test image (shifted image) + create output_pic
    draw_pic = shifted_img.copy()

    # greying the input picture
    shifted_img_grey = cv.cvtColor(shifted_img, cv.COLOR_BGR2GRAY)
#

    # blurring with 2 options
    if blur_strgth == "low":
        shifted_img_grey_blurred = cv.GaussianBlur(shifted_img_grey, (3, 3), 1, 1)
    elif blur_strgth == "high":
        shifted_img_grey_blurred = cv.GaussianBlur(shifted_img_grey, (9, 9), 2, 2)

    # set threshold and find contours
    ret, thresh = cv.threshold(shifted_img_grey_blurred, 90, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)  # source img, contour retrieval mode, contour approximation method
    # contours: python list with all contours in the image (each contour is np array of (x,y) coordinates of boundary points of the object)

    # finding a good contour (green)
    cv.drawContours(draw_pic, contours, 23, (0, 255, 0), 2)  # 18, 22, 23 ; pic 2.png
    good_cnt = contours[23]  # (82, 1, 2) <- 82 (x,y) coordinates

    # fitting ellipse (blue)
    ellipse = cv.fitEllipse(good_cnt)
    #              center (?)                               scale                              angle
    # ((76.10218811035156, 119.91889953613281), (44.374759674072266, 62.04508590698242), 167.4716796875)
    cv.ellipse(draw_pic, ellipse, (255, 0, 0), 1)

    # bounding box (red)
    rect = cv.minAreaRect(good_cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # cv.drawContours(draw_pic, [box], 0, (0, 0, 255), 1)

    ## correcting the ellipse to circle

    # parameters
    angle = ellipse[2]
    scale_tmp = ellipse[1]
    scale = scale_tmp[0] / scale_tmp[1]

    # rotation matrix
    M = cv.getRotationMatrix2D((shifted_img.shape[0] / 2, shifted_img.shape[1] / 2), angle, 1)  # 2x3 matrix
    # adding the scaling
    M[:, 0:2] = np.array([[1, 0], [0, scale]]) @ M[:, 0:2]
    # moving the ellipse so it doesn't end up outside the image (it's not correct to keep the ellipse in the middle of the image)
    M[1, 2] = M[1, 2] * scale

    # apply transform
    corr_img = cv.warpAffine(shifted_img, M, (256, 256), borderValue=(0,0,0))
    corr_img_grey_blurred = cv.warpAffine(shifted_img_grey_blurred, M, (256, 256))

    return corr_img_grey_blurred, corr_img, shifted_img_grey_blurred, draw_pic

def main():
    # test image (shifted image) + create output_pic
    shifted_img = cv.imread('retrieval_dataset/0.png', cv.IMREAD_UNCHANGED)

    # correcting the image (+ grey blurred version); shifted_img.. is before contour, draw pic.. shows contour
    corr_img_grey_blurred, corr_img, shifted_img_grey_blurred, draw_pic = corr_pic(shifted_img, "low")

    # printing pictures
    cv.imshow("shifted image", shifted_img)
    cv.imshow('gray/blurred', shifted_img_grey_blurred)
    cv.imshow('draw pic (contours)', draw_pic)
    cv.imshow('corr img', corr_img)
    cv.imshow('corr img gray/blurred', corr_img_grey_blurred)
    # window management
    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()