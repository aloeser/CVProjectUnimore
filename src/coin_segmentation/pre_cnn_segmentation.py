import cv2 as cv
import numpy as np
import os
import cnn_test
import geom_shift_corr

##

# KEY fct 1 + print
def hough_circle_segmentation(inp_pic, blur_strgth="low"):
    """
    Gets random input picture from retrieval_dataset with multiple coins.
    Returns all found circles in list of (x,y,r) circle format.
    :param inp_pic: random input picture
    :param blur_strgth: low or high
    :return: list of (x,y,r) circles
    """
    # Correcting the geometric shift
    corr_img_grey_blurred, corr_img, shifted_img_grey_blurred, draw_pic = geom_shift_corr.corr_pic(inp_pic, blur_strgth)

    # printing pictures
    cv.imshow("shifted image", inp_pic)
    cv.imshow('gray/blurred', shifted_img_grey_blurred)
    cv.imshow('draw pic (contours)', draw_pic)
    cv.imshow('corr img', corr_img)
    
    cv.imshow('corr img gray/blurred', corr_img_grey_blurred)

    # HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
    circles = cv.HoughCircles(corr_img_grey_blurred, cv.HOUGH_GRADIENT, dp=1, minDist=20, param1 = 200, param2 = 30, minRadius=0, maxRadius=0)

    if (circles is None):
        print("No circles found.")
    else:
        print("Circle/s found.")
        circles = np.uint16(np.around(circles))
        return corr_img, circles[0,:], corr_img_grey_blurred
def print_circles_in_pic(inp_pic, circles):
    """
    Copies input picture and draws given circles in it and shows them.
    :param inp_pic: input picture without drawn circles
    :param circles: circle information in format [(x,y,r),..]
    :return: output picture with circles and centers drawn in
    """
    out_pic = inp_pic.copy()
    for i in circles:
        # draw the outer circle
        cv.circle(out_pic, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(out_pic, (i[0], i[1]), 2, (0, 0, 255), 3)
    return out_pic

##

# help functions for KEY fct 2
def resize_pic(inp_pic, x=64, y=64):
    """
    Resize a 250x250 input picture to 64x64 output picture.
    :param inp_pic: input picture
    :param x: x, width
    :param y: y, height
    :return: output picture
    """
    out_pic = cv.resize(inp_pic, (y, x), interpolation=cv.INTER_AREA)
    return out_pic
def create_masks(x_ctr, y_ctr, r, x=250, y=250):
    """
    Creates mask (and inverted mask) with specific size (y, x) and white (black) circle (x_ctr, y_ctr, r) in it.
    :param x: width of mask
    :param y: height of mask
    :param x_ctr: x-coordinate of circle center
    :param y_ctr: y-coordinate of circle center
    :param r: radius of circle
    :return: mask (white filled circle) and mask_inv (black filled circle)
    """
    mask = np.zeros((y, x, 3), np.uint8)
    center = (x_ctr, y_ctr)
    # cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
    cv.circle(mask, center, r, color=(255, 255, 255), thickness=-1, lineType=8, shift=0) # thickness=-1 => filled circle
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    mask_inv = cv.bitwise_not(mask)
    return mask, mask_inv

# KEY fct 2 + print
def generate_output_vec(inp_pic, circles):
    """
    Collects all detected circles/coins for a single input picture
    sizes them 64x64 with black background and saves them in output vector(segmentation)/input vector(cnn).
    :param inp_pic: input picture with coins
    :param circles: circle data
    :return: output vector
    """
    all_coin_outputs = []
    mask = create_masks(31, 31, 31, 64, 64)[0]  # black background, white circle mask
    for (x, y, r) in circles:
        if r > x or r > y or x + r >= inp_pic.shape[1] or y + r >= inp_pic.shape[0]:
            continue
        assert x >= r and y >= r and x + r < inp_pic.shape[1] and y + r < inp_pic.shape[0]
        curr_coin_area = inp_pic[y-r : y+r, x-r : x+r]                     # select roi (region of interest)
        curr_coin = resize_pic(curr_coin_area)                             # 64 x 64 resize
        curr_coin_output = cv.bitwise_and(curr_coin, curr_coin, mask=mask) # resized roi with black background
        all_coin_outputs.append(curr_coin_output)
    output_vector = np.array(all_coin_outputs)
    return output_vector
def print_output_coins_conc(output_vector):
    """
    Returns a output picture of all images from output_vector/cnn_input_vector horizontally concatenated.
    :param output_vector: input image vector
    :return: concatenated output picture
    """
    image = output_vector[0]
    for next_img in output_vector[1:]:
        image_c = np.concatenate((image, next_img), axis=1)
        image = image_c
    return image

##

# help functions for KEY fct 3
def create_empty_coin_dic():
    """
    Creating an empty coin dictionary. All coin 'keys' in cents all 'values' initial set to zero.
    :return: coin dictionary
    """
    coin_dic = {
        "100": 0,
        "1": 0,
        "200": 0,
        "2": 0,
        "5": 0,
        "10": 0,
        "20": 0,
        "50": 0
    }
    return coin_dic
def get_amt_sum(coin_dic):
    """
    Returns amount of coins and predicted sum in the picture.
    :param coin_dic: coin dictionary with predictions
    :return: amount of coins, predicted sum
    """
    keys = [int(i) for i in coin_dic.keys()]
    values = list(coin_dic.values())
    coin_amt = sum(values)
    pred_sum = np.dot(keys, values)
    return coin_amt, pred_sum

# KEY fct 3 (cnn)
def get_pred_data(output_vector):
    """
    Returns all predicted data from given output vector of an input picture.
    :param output_vector: vector containing all coins with 64x64x3 shape
    :return: coin amount, predicted sum, coin dictionary
    """
    coin_dic = create_empty_coin_dic()
    for img in output_vector:
        cent = cnn_test.get_prediction(img)
        coin_dic[str(cent)] += 1
    coin_amt, pred_sum = get_amt_sum(coin_dic)
    return coin_amt, pred_sum, coin_dic

# PRINT (one pic)
def print_one_pic_sol(inp_pic):
    """
    Prints the CNN predicted data, shows input picture (also with detected circles) and associated output vector.
    :param inp_pic: single input picture
    :return: prints input picture, input picture with detected circles, concatenated output vector images
    """
    # correct input picture and get all found circles [(x, y, r)] through hough_circle_detection
    corr_img, circles, corr_img_grey_blurred = hough_circle_segmentation(inp_pic)

    # show found circles in corr_img_grey_blurred
    cv.imshow("detected circles", print_circles_in_pic(corr_img_grey_blurred, circles))

    # generate output vector with found circles in input picture
    output_vector = generate_output_vec(corr_img, circles)

    # predict with cnn the sum from the coin dictionary
    coin_amt, pred_sum, coin_dic = get_pred_data(output_vector)
    print('# detected coins:', coin_amt, '\npredicted sum:', pred_sum, 'cent', '\ncoin dictionary:', coin_dic)

    # generate input picture with circles
    inp_pic_circles = print_circles_in_pic(corr_img, circles)
    # generate output pictures with concatinated coins
    conc_output_vector = print_output_coins_conc(output_vector)

    cv.imshow('inp pic with circles', inp_pic_circles)
    cv.imshow('conc output vector', conc_output_vector)

    # window management
    cv.waitKey(0)
    cv.destroyAllWindows()

# PRINT (n pic)
def test_n_input_pic(n):
    """
    Testing the (0, ..., n-1) retrieval dataset input pictures through the 'Coin Segmentation' and the 'CNN prediction'.

    :param n: amount of input pictures that should get tested from the retrieval_dataset starting from the first
    :return: 'print_one_pic_sol' prints for every retrieval dataset input picture
    """
    data_len = len(os.listdir("retrieval_dataset")) - 1

    for i in range(data_len)[0:n]:
        pic_path = 'retrieval_dataset/%s.png' % i
        curr_pic = cv.imread(pic_path, cv.IMREAD_UNCHANGED)
        print_one_pic_sol(curr_pic)


def main():
    test_n_input_pic(n=1)


if __name__ == "__main__":
    main()
