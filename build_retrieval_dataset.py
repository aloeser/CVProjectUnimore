import math
import cv2 as cv
import numpy as np
import sys
import random
import os

# Returns a random coin
def random_coin(countries=['Germany', 'Italy'], cents=[200, 100, 50, 20, 10, 5, 2, 1], data_path='data', side=None):

    coin_names = {
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
    coin_value = random.choice(cents)
    coin_name = coin_names[coin_value]
    path = os.path.join(data_path, country, coin_name)

    coin_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if side is not None:
        coin_files = list(filter(lambda x: side in x, coin_files))

    coin_file = random.choice(coin_files)

    print(f"'{side}' in {coin_file}")
    print(f"using {coin_file}")

    return coin_value, cv.imread(coin_file)

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

# list of existing coins. Coins are given by their center coordinates (y, x) and their radius r.
# #The radius is assumed to be half the image width/height (we expect square images, so it does not matter)
EXISTING_COIN_POSITIONS = []
def insert_pic(retrieval_img, coin_img, inverted_mask):
    # returns y, x coordinates that satisfy two criteria:
    #  criteria 1: the coin is completely inside the image
    #  criteria 2: the coin does not overlap with any existing coin - using actual overlap, i.e., round rather than square bounding boxes
    def get_insertable_position(retrieval_img, coin_radius, existing_coin_positions, max_attempts=10):
        retrieval_h, retrieval_w, _ = retrieval_img.shape
        # Look for possible positions till we find one, then break the loop
        # Use no more than max_attempts attempts to find a suitable position
        attempt = 0
        while attempt < max_attempts:
            attempt += 1

            # restrict the random range the following way to enforce the criteria 1
            # (if coins are at least 1 radius away from the border, they cant go beyond)
            y = random.randint(coin_radius, retrieval_h - coin_radius)
            x = random.randint(coin_radius, retrieval_w - coin_radius)

            # check for criteria 2 - for each existing coin, check for overlaps
            overlap_found = False
            for (c_y, c_x, c_r) in existing_coin_positions:
                # calculate euclidian distance between the coins' center points
                distance_between_centers = math.sqrt((c_y - y)**2 + (c_x - x)**2)
                # the distance between the centers has to be at least as big as the sum of both radius,
                # as each coin is going to take exactly <radius> space in each direction
                if distance_between_centers < c_r + coin_radius:
                    # too close, we found an overlap
                    overlap_found = True

            if not overlap_found:
                # both checks pass -> we found a valid position, return it
                return y, x

        # if the loops exits without a return, we could not find a valid position. Return None
        return None, None

    # Step 1: calculate coin radius
    # not sure yet if all coin images are going to be 64x64 (making 1€ the same size as 1ct feels wrong),
    # so for now I assume that coin images might have different sizes, and with that, a different radius
    coin_h, coin_w, _ = coin_img.shape
    # TODO uncomment the following line as soon as roman's code is working
    #assert coin_h == coin_w, "coins should be square"
    coin_radius = (coin_h + 1) // 2

    # Step 2: try to find a valid position - this may fail if the image is too full
    center_y, center_x = get_insertable_position(retrieval_img, coin_radius, EXISTING_COIN_POSITIONS)
    # check if we found a valid position, if not raise an exception
    if center_y is None:
        raise Exception(f"could not find a position for the {len(EXISTING_COIN_POSITIONS)+1}th coin")

    # Step 3: memorize the coin's position, in case further coins need to be inserted later on
    EXISTING_COIN_POSITIONS.append((center_y, center_x, coin_radius))

    # Step 4: actually perform the insert - still hacky atm
    topleft_y = center_y - coin_radius
    topleft_x = center_x - coin_radius

    def insert_coin_to_position(target_img, y, x, coin_only_img, inverted_mask):
        ch, cw, _ = coin_only_img.shape
        background = cv.bitwise_and(target_img[y:y+ch, x:x+cw], target_img[y:y+ch, x:x+cw], mask=inverted_mask)
        final_image = cv.add(coin_only_img, background)
        target_img[y:y+ch, x:x+cw] = final_image

    insert_coin_to_position(retrieval_img, topleft_y, topleft_x, coin_img, inverted_mask)


"""
Takes a retrieval image as input and returns a copy of the image, with an homographic transformation applied.
By default, the output image has the same shape as the original image.
"""
def perform_homographic_transform(retrieval_img, target_shape = None):
    h, w, _ = retrieval_img.shape
    if target_shape is None:
        target_shape = retrieval_img.shape[:2]
    target_h, target_w = target_shape

    pts_src = np.array([[0, 0], [0, w], [h, 0], [h, w]])
    pts_dst = np.array([[0, 0], [int(0.2*target_w), target_h], [target_w, 0], [int(0.8*target_w), target_h]])
    h, status = cv.findHomography(pts_src, pts_dst)
    return cv.warpPerspective(retrieval_img, h, (target_w, target_h))


def generate_retrieval_image(data_path='data', h=256, w=256, coin_amt_mean=9):
    # background = background()    # for now: a random, 1-colored background
    background_colors = [(0, 200, 0), (200, 0, 0), (0, 0, 200) ]
    background_color = random.choice(background_colors)

    # pic = gen_pic(y,x,background)
    retrieval_img = create_bgr_image(h, w, bg=background_color)

    # label-list: a dictionary mapping coin cent values (integers) to their frequency,
    # and a 'sum' field containing the accumulated value. We could calculate the sum on the fly from the dictionary too,
    # but I'm lazy and one additional dictionary entry does not hurt
    labels = {200: 0, 100: 0, 50: 0, 20: 0, 10: 0, 5: 0, 2: 0, 1: 0, 'sum': 0, 'num_coins': 0, 'background': background_color}

    # coin_amt = gaussian(coin_amt_mean, var=1)
    coin_amt = coin_amt_mean

    # while #coins < coin_amt
    hashtag_coins = 0
    while hashtag_coins < coin_amt:
        # get coin from data
        coin_value, coin_img = random_coin(data_path=data_path)

        # add to label-list
        labels[coin_value] += 1
        labels['sum'] += coin_value
        labels['num_coins'] += 1

        # scale, rotate, remove background
        tmp, inv_mask = extract_rotated_resized_coin(coin_img)

        # find position for insert
        #y, x = get_insertable_position()

        # insert coin into the retrieval image
        insert_pic(retrieval_img, tmp, inv_mask)

        # #coins++
        hashtag_coins += 1

    # geometrischer shift gesamt pic
    img_out = perform_homographic_transform(retrieval_img)
    cv.imshow("warp", img_out)
    # return pic, label
    return retrieval_img, labels

def main():
    img, _ = generate_retrieval_image(h=256, w=256, coin_amt_mean=5)
    cv.imshow("result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()