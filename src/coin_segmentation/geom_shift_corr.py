import cv2 as cv
import numpy as np

def grey_n_blur(inp_pic, blur_strgth):
    """
    Takes input picture, greys it and blurs it with low or high strength.
    :param inp_pic: input picture
    :param blur_strgth: blur strength
    :return: grey blurred input picture
    """
    # greying the input picture
    inp_pic_grey = cv.cvtColor(inp_pic, cv.COLOR_BGR2GRAY)
    #

    # blurring with 2 options
    if blur_strgth == "low":
        inp_pic_grey_blurred = cv.GaussianBlur(inp_pic_grey, (3, 3), 1, 1)
    elif blur_strgth == "high":
        inp_pic_grey_blurred = cv.GaussianBlur(inp_pic_grey, (9, 9), 2, 2)
    return inp_pic_grey_blurred
def get_thresh_based_contours(inp_pic, threshold):
    """
    Returns all found contours of the input image with respect to a given threshold.
    :param inp_pic: input picture
    :param threshold: threshold value
    :return: list of all contours in the image (each contour is np array of (x,y) coordinates of boundary points of the object)
    """
    ret, thresh = cv.threshold(inp_pic, threshold, 255, 0)
    # source img, contour retrieval mode, contour approximation method
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)
    return contours
def get_max_cnt_idx(contours):
    """
    Finding the contour out of all found contours with the biggest contour area.
    :param contours: list of all contours
    :return: np array of contour with maximal area and its index in the contours list
    """
    max_contour = max(contours, key=cv.contourArea)

    for idx, cnt in enumerate(contours):
        if np.array_equal(cnt, max_contour):
            cnt_idx = idx
    return max_contour, cnt_idx

def find_corr_matrix(shifted_img, obj):
    """
    Finding a correcting matrix for the shifted image depending from the fitted object (ellipse/rectangle).
    :param shifted_img: shifted input image
    :param ellipse: ((center coordinates x, y), (scale 1, 2), angle)
    :return: matrix to correct fitted ellipse in shifted input image
    """
    # parameters
    angle = obj[2]
    """
    finding smallest turning angle to correct maybe helpful..
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    """
    scale_tmp = obj[1]
    scale = scale_tmp[0] / scale_tmp[1]

    # rotation matrix
    M = cv.getRotationMatrix2D((shifted_img.shape[0] / 2, shifted_img.shape[1] / 2), angle, 1)  # 2x3 matrix
    # adding the scaling
    M[:, 0:2] = np.array([[1, 0], [0, scale]]) @ M[:, 0:2]
    # moving the ellipse so it doesn't end up outside the image (it's not correct to keep the ellipse in the middle of the image)
    M[1, 2] = M[1, 2] * scale
    mtrx = M
    return mtrx

def corr_pic(shifted_img, blur_strgth):
    # copy for a picture to show drawings
    draw_pic = shifted_img.copy()

    # prepare picutre for contour detection
    shifted_img_grey_blurred = grey_n_blur(shifted_img, blur_strgth)

    # depending on the threshold, find contours
    contours = get_thresh_based_contours(shifted_img_grey_blurred, 90)

    # finding the biggest contour (green)
    max_contour, cnt_idx = get_max_cnt_idx(contours)
    cv.drawContours(draw_pic, contours, cnt_idx, (0, 255, 0), 2)

    # fitting ellipse to max_contour (blue)
    ellipse = cv.fitEllipse(max_contour)
    #              center coordinates                           scale                         angle
    # ((76.10218811035156, 119.91889953613281), (44.374759674072266, 62.04508590698242), 167.4716796875)
    cv.ellipse(draw_pic, ellipse, (255, 0, 0), 1)

    ## find bounding box (red)
    # get rect (coordinates, scales, angle)
    rect = cv.minAreaRect(max_contour)
    # get box corner coordinates
    box = cv.boxPoints(rect)
    box = np.int0(box) # [[x1,y1],..,[x4,y4]] np arrays
    # draw bounding box
    cv.drawContours(draw_pic, [box], 0, (0, 0, 255), 1)

    #correcting the ellipse to circle
    # mtrx = find_corr_matrix(shifted_img, rect)
    mtrx = find_corr_matrix(shifted_img, ellipse)

    # apply transform
    corr_img = cv.warpAffine(shifted_img, mtrx, (256, 256), flags=cv.INTER_CUBIC, borderValue=(0,0,0))
    corr_img_grey_blurred = cv.warpAffine(shifted_img_grey_blurred, mtrx, (256, 256))

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