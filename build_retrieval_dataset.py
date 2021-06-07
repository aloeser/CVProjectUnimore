import math
import cv2 as cv
import numpy as np
import random
import os
import background_generator

def random_coin(countries=['Germany', 'Italy'], cents=[200, 100, 50, 20, 10, 5, 2, 1], data_path='data', side=None):
    """
    Returns a random coin.
    :param countries:
    :param cents:
    :param data_path:
    :param side:
    :return:
    """

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

# BACKGROUND MERGE
def create_bgr_image(height, width, bg=(0,0,0)):
    """
    Creates a image with the given height, width, and background.
    :param height:
    :param width:
    :param bg:
    :return:
    """
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = bg
    return img




## rotate_extract_coin ##
def rotate_pic(inp_pic, phi):
    """
    Rotates an input picture counter-clockwise around its center with an given angle.
    :param inp_pic: input picture
    :param phi: rotation angle
    :return: output picture
    """
    y, x = inp_pic.shape[:2]
    matrix = cv.getRotationMatrix2D((x / 2, y / 2), phi, 1.0)
    out_pic = cv.warpAffine(inp_pic, matrix, (x, y))
    return out_pic
def hough_circle_detection(inp_pic, blur_strgth, hough_dp=1, minRadius=180, maxRadius=190):
    """
    Detects a circle (through Hough Transformation) and returns the coordinates (x_ctr, y_ctr, r)
    :param inp_pic: input picture
    :param blur_strgth: choose between "low" and "high" blurring
    :param hough_dp: Inverse ratio of the accumulator resolution to the image resolution.
    For example, if dp=1 , the accumulator has the same resolution as the input image.
    If dp=2 , the accumulator has half as big width and height.
    :param minRadius: minimum circle radius
    :param maxRadius: maximum circle radius
    :return: coordinates (x_ctr, y_ctr) and the circle radius r of the found circle
    """
    inp_pic_grey = cv.cvtColor(inp_pic, cv.COLOR_BGR2GRAY)
    if blur_strgth == "low":
        inp_pic_grey_blurred = cv.GaussianBlur(inp_pic_grey, (3,3), 1, 1)
    elif blur_strgth == "high":
        inp_pic_grey_blurred = cv.GaussianBlur(inp_pic_grey, (9,9), 2, 2)
    #cv.imshow("grey_blurred", inp_pic_grey_blurred)
    # HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
    circles = cv.HoughCircles(inp_pic_grey_blurred, cv.HOUGH_GRADIENT, hough_dp, 20, minRadius, maxRadius) #minDist = 20
    if (circles is None):
        print("No circles found.")
    elif len(circles) != 1:
        print("More than one circle found.")
    else:
        circles = np.round(circles[0, :].astype("int")) # rounding coordinates to integer values
        x_ctr, y_ctr, r = circles[0]
        #cv.circle(inp_pic, (125, 125), r, color=(0, 0, 0), thickness=4, lineType=8, shift=0)
        #cv.imshow('circle in inp_pic', inp_pic)
        print("1 circle found. radius: ", r, ", center coordinate: (", x_ctr, ",", y_ctr, ")")
        return x_ctr, y_ctr, r
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
def extract_coin(inp_pic, mask):
    """
    Takes an image and extracts the coin with respect to the given coin-mask.
    :param inp_pic: input picture
    :param mask: coin-mask
    :return: output picture of the coin without a background
    """
    out_pic = cv.bitwise_and(inp_pic, inp_pic, mask=mask)
    return out_pic
def resize_pic(inp_pic, x=64, y=64):
    """
    Resize a 250x250 input picture to 64x64 output picture.
    :param inp_pic: input picture
    :param x: x, width
    :param y: y, height
    :return: output picture
    """
    r = x / y
    dim = (x, int(x * r))
    out_pic = cv.resize(inp_pic, dim, interpolation=cv.INTER_AREA)
    return out_pic
####
def rotate_extract_coin(inp_pic, phi):
    """
    Rotates, detects circle, builds masks, extracts coin and resizes.
    :param inp_pic: input picture
    :param phi: rotation angle
    :return: output picture, inverse mask
    """
    # rotate
    rot_pic = rotate_pic(inp_pic, phi)
    #cv.imshow("rot_pic", rot_pic)

    # hough
    x_ctr, y_ctr, r = hough_circle_detection(rot_pic, "low")

    # masks
    y, x, _ = inp_pic.shape
    mask, mask_inv = create_masks(x_ctr, y_ctr, r, x, y)
    #cv.imshow("mask", mask)

    # extract coin / delete background
    coin_no_background = extract_coin(rot_pic, mask)
    #cv.imshow("coin_no_background", coin_no_background)

    # resize
    out_pic = resize_pic(coin_no_background)
    mask_inv = resize_pic(mask_inv)
    #cv.imshow("final_pic", final_pic)

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


def generate_retrieval_image(data_path='data', h=256, w=256, coin_amt_mean=9, do_homographic_transform=True):
    # background = background()    # for now: a random, 1-colored background
    #background_colors = [(0, 200, 0), (200, 0, 0), (0, 0, 200) ]
    #background_color = random.choice(background_colors)

    # pic = gen_pic(y,x,background)
    #retrieval_img = create_bgr_image(h, w, bg=background_color)

    retrieval_img = background_generator.gen_background()

    # label-list: a dictionary mapping coin cent values (integers) to their frequency,
    # and a 'sum' field containing the accumulated value. We could calculate the sum on the fly from the dictionary too,
    # but I'm lazy and one additional dictionary entry does not hurt
    labels = {200: 0, 100: 0, 50: 0, 20: 0, 10: 0, 5: 0, 2: 0, 1: 0, 'sum': 0, 'num_coins': 0} # 'background': background_color}

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
        phi = np.random.randint(0, 360)
        tmp, inv_mask = rotate_extract_coin(coin_img, phi)

        # find position for insert
        #y, x = get_insertable_position()

        # insert coin into the retrieval image
        insert_pic(retrieval_img, tmp, inv_mask)

        # #coins++
        hashtag_coins += 1

    # geometrischer shift gesamt pic
    if do_homographic_transform:
        retrieval_img = perform_homographic_transform(retrieval_img)

    # return pic, label
    return retrieval_img, labels

def main():
    img, _ = generate_retrieval_image(h=256, w=256, coin_amt_mean=5, do_homographic_transform=False)
    cv.imshow("result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()