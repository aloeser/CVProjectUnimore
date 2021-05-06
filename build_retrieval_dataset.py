import cv2 as cv
import numpy as np
import sys
import random
import os

# Returns a random coin
def random_coin(countries=['Germany', 'Italy'], cents=[200, 100, 50, 20, 10, 5, 2, 1], data_folder='data', side=None):

    coin_values = {
        200: '2',
        100: '1',
        50: '50ct',
        20: '20ct',
        10: '10ct',
        5: '5ct',
        2: '2ct',
        1: '1ct'
    }
    country = random.choice(countries)
    coin_value = coin_values[random.choice(cents)]
    path = os.path.join(data_folder, country, coin_value)

    coin_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if side is not None:
        coin_files = list(filter(lambda x: side in x, coin_files))
    else:
        assert False

    coin_file = random.choice(coin_files)
    assert side in coin_file
    print(f"'{side}' in {coin_file}")
    print(f"using {coin_file}")

    return cv.imread(coin_file)

def insert_coin_threshold_based(img1, y, x, img2, thresh1=10, thresh2=220):
    #cv.imshow('img1', img1)
    #cv.imshow('img2', img2)

    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0+y:rows+y, 0+x:cols+x]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    img2gray_inv = cv.bitwise_not(img2gray)
    ret, mask = cv.threshold(img2gray_inv, thresh1, 255, cv.THRESH_BINARY)

    # Take only region of logo from logo image.
    img2_fg = cv.bitwise_and(img2, img2, mask=mask)

    # additional fine tunging / correcting mask with img2_fg
    img2_fg_gray = cv.cvtColor(img2_fg, cv.COLOR_BGR2GRAY)
    ret2, mask2 = cv.threshold(img2_fg_gray, thresh2, 255, cv.THRESH_BINARY)
    mask2_inv = cv.bitwise_not(mask2)

    better_mask = cv.bitwise_and(mask, mask2_inv)
    better_mask_inv = cv.bitwise_not(better_mask)

    # Now black-out the area of logo in ROI
    better_img1_bg = cv.bitwise_and(roi, roi, mask=better_mask_inv)

    # Take only region of logo from logo image.
    better_img2_fg = cv.bitwise_and(img2, img2, mask=better_mask)

    dst = cv.add(better_img1_bg, better_img2_fg)
    img1[0+y:rows+y, 0+x:cols+x] = dst

# Creates a image with the given height, width, and background
def create_bgr_image(height, width, bg=(0,0,0)):
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = bg
    return img


def extract_rotated_resized_coin(inp_pic, y=64, x=64, phi=0):
    # resize 250x250 to 64x64
    r = x / y
    dim = (x, int(x * r))
    inp_pic_res = cv.resize(inp_pic, dim, interpolation=cv.INTER_AREA)
    cv.imshow('inp_pic_res', inp_pic_res)

    # rotation
    matrix = cv.getRotationMatrix2D((x / 2, y / 2), phi, 1.0)
    inp_pic_rot = cv.warpAffine(inp_pic_res, matrix, (x, y))
    cv.imshow('inp_pic_rot', inp_pic_rot)

    # hough circle detection
    inp_pic_grey = cv.cvtColor(inp_pic_rot, cv.COLOR_BGR2GRAY)
    inp_pic_grey = cv.GaussianBlur(inp_pic_grey, (3, 3), 1, 1)
    cv.imshow('inp_pic_grey', inp_pic_grey)
    # HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
    circles = cv.HoughCircles(inp_pic_grey, cv.HOUGH_GRADIENT, dp=3, minDist=20, minRadius=25, maxRadius=35)

    # draw circle in mask
    mask = np.zeros((y, x, 3), np.uint8)
    if (circles is not None):
        if len(circles) == 1:
            circles = np.round(circles[0, :].astype("int"))
            x_, y_, r = circles[0]
            cv.circle(mask, (x_, y_), r, (255, 255, 255), -1, 8, 0)
            print('1 circle found. r: ', r)
        else:
            print('multiple circles.')
    else:
        print('no circle found.')

    # with mask filter input pic and get inverted mask
    cv.imshow('mask', mask)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    out_pic = cv.bitwise_and(inp_pic_rot, inp_pic_rot, mask=mask)
    mask_inv = cv.bitwise_not(mask)

    return out_pic, mask_inv


def generate_retrieval_image(data_path='data', h=256, w=256, coin_amt_mean=9):
    # background = background():
    # pic = gen_pic(y,x,background)
    retrieval_img = create_bgr_image(256, 256, bg=(0, 255, 0))

    # label-list (empty)
    labels = []

    # coin_amt = gaussian(coin_amt_mean, var=1)
    coin_amt = 3

    # while #coins < coin_amt
    hashtag_coins = 0
    while hashtag_coins < coin_amt:
        # get coin from data
        coin_img = random_coin()

        # add to label-list
        # TODO

        # scale, rotate, remove background
        tmp, inv_mask = extract_rotated_resized_coin(coin_img)

        # find position for insert
        #y, x = get_insertable_position()

        # insert coin into the retrieval image
        insert_pic(retrieval_img, tmp, inv_mask)

        # #coins++
        hashtag_coins += 1

    # geometrischer shift gesamt pic
    # TODO

    # return pic, label
    return retrieval_img, labels

def main():
    generate_retrieval_image()

if __name__ == "__main__":
    main()