import json
import math
import shutil
from tqdm import tqdm

import cv2 as cv
import numpy as np
import random
import os
import background_generator

def random_coin(countries=['Austria', 'Belgium', 'Finland', 'France', 'Germany', 'Ireland', 'Italy', 'Luxemburg', 'Netherlands', 'Portugal', 'Spain'], cents=[200, 100, 50, 20, 10, 5, 2, 1], data_path='data'):
    """
    Returns a random coin.
    :param countries: the list of countries to choose from
    :param cents: the list of coin values to choose from, in cents
    :param data_path: the folder containing the dataset
    :return: coin value, coin image
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
    coin_file = random.choice(coin_files)

    return coin_value, cv.imread(coin_file)

def create_bgr_image(height, width, bg=(0,0,0)):
    """
    Creates a image with the given height, width, and background.
    :param height: height of the picture
    :param width: width of the picture
    :param bg: background color, given as (b,g,r) tuple
    :return: an image of size height * width with bg as background color
    """
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = bg
    return img

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

def hough_circle_detection(inp_pic, blur_strgth, hough_dp=1, minRadius=120, maxRadius=130):
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
    # if circles=None no circles found
    circles = cv.HoughCircles(inp_pic_grey_blurred, cv.HOUGH_GRADIENT, hough_dp, circles=1, minDist=20, minRadius=minRadius, maxRadius=maxRadius)
    if (circles is None):
        print("No circles found.")
        raise Exception("No circles found.")
    elif circles.shape == (4,1):
        # print("More than one circle found.")
        # For some images, the detection fails and openCV returns a shape of (4,1).
        # I cannot find this behaviour in the documentation, so maybe it is a bug
        # Best fix so far: guess the circle's position
        y, x = inp_pic_grey.shape[:2]
        return int(x/2), int(y/2), int(min(x,y) / 2 * 0.95)
    else:
        circles = np.round(circles[0, :].astype("int")) # rounding coordinates to integer values
        x_ctr, y_ctr, r = circles[0]
        #cv.circle(inp_pic, (125, 125), r, color=(0, 0, 0), thickness=4, lineType=8, shift=0)
        #cv.imshow('circle in inp_pic', inp_pic)
        # print("1 circle found. radius: ", r, ", center coordinate: (", x_ctr, ",", y_ctr, ")")
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
    out_pic = cv.resize(inp_pic, (y, x), interpolation=cv.INTER_AREA)
    return out_pic

def rotate_extract_coin(inp_pic, phi, new_size):
    """
    Rotates, detects circle, builds masks, extracts coin and resizes.
    :param inp_pic: input picture
    :param phi: rotation angle
    :new_size: new size of the coin (still squared)
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
    out_pic = resize_pic(coin_no_background, x=new_size, y=new_size)
    mask_inv = resize_pic(mask_inv, x=new_size, y=new_size)
    #cv.imshow("final_pic", final_pic)

    return out_pic, mask_inv

def get_insertable_position(target_img_shape, coin_radius, existing_coin_positions, max_attempts=10):
    """
    Returns y, x center coordinates of a circle/coin that satisfy two criteria:
      criteria 1: the coin is completely inside the image
      criteria 2: the coin does not overlap with any existing coin
        - we use actual overlap, i.e., round rather than square bounding boxes
    If no such position can be found in at most max_attempts attempts, the function returns None, None

    :param target_image_shape: the shape of the target image (we need height and width)
    :param coin_radius: radius of the coin to insert
    :param existing_coin_positions: list of existing coin positions, i.e., their center coordinates and radius
    :param max_attempts: the maximum number of attempts to find a valid position for the new coin
    :return: y,x center coordinates of the coin position, or None, None if no valid position was found
    """
    retrieval_h, retrieval_w, _ = target_img_shape
    # Look for possible positions till we find one, then break the loop
    # Use no more than max_attempts attempts to find a suitable position
    for _ in range(max_attempts):
        # restrict the random range the following way to enforce the criteria 1
        # (if coins are at least 1 radius away from the border, they cant go beyond)
        y = random.randint(coin_radius, retrieval_h - coin_radius)
        x = random.randint(coin_radius, retrieval_w - coin_radius)

        # check for criteria 2 - for each existing coin, check for overlaps
        overlap_found = False
        for (c_y, c_x, c_r) in existing_coin_positions:
            # calculate euclidian distance between the coins' center points
            distance_between_centers = math.sqrt((c_y - y) ** 2 + (c_x - x) ** 2)
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

def insert_coin_to_position(target_img, center_y, center_x, coin_radius, coin_only_img, inverted_mask):
    """
    Inserts the coin image into the target image
    :param target_img: the image where the coin should be inserted
    :param center_y: y position of the coin's center
    :param center_x: x position of the coin's center
    :param coin_radius: radius of the coin
    :param coin_only_img: an image containing only the coin, with a blackened background
    :param inverted_mask: a mask with the same shape as the coin, but black in the area of the coin and white otherwise
    :return: None
    """
    topleft_y = center_y - coin_radius
    topleft_x = center_x - coin_radius
    ch, cw, _ = coin_only_img.shape
    background = cv.bitwise_and(target_img[topleft_y:topleft_y+ch, topleft_x:topleft_x+cw], target_img[topleft_y:topleft_y+ch, topleft_x:topleft_x+cw], mask=inverted_mask)
    final_image = cv.add(coin_only_img, background)
    target_img[topleft_y:topleft_y+ch, topleft_x:topleft_x+cw] = final_image

def perform_homographic_transform(retrieval_img, target_shape = None):
    """
    Takes a retrieval image as input and returns a copy of the image, with an homographic transformation applied.
    By default, the output image has the same shape as the original image.
    Currently there is only one hardcoded transform available, i.e., not random.
    :param retrieval_img: the image to transform
    :param target_shape: the shape of the new image
    :return: the transformed image
    """
    h, w, _ = retrieval_img.shape
    if target_shape is None:
        target_shape = retrieval_img.shape[:2]
    target_h, target_w = target_shape

    pts_src = np.array([[0, 0], [0, w], [h, 0], [h, w]])
    pts_dst = np.array([[0, 0], [int(0.2*target_w), target_h], [target_w, 0], [int(0.8*target_w), target_h]])
    h, status = cv.findHomography(pts_src, pts_dst)
    return cv.warpPerspective(retrieval_img, h, (target_w, target_h))

def get_real_coin_size(coin_value, two_euro_reference_size=64):
    """
    Returns the size physical size of a coin, assuming a 2€ coin is two_euro_reference_size pixels wide
    :param coin_value: the value of the coin
    :param two_euro_reference_size: width/height of a 2€ coin
    :return: the size of a coin of the given value, with respect to the size of a 2€ coin
    """
    # using https://de.wikipedia.org/wiki/Eurom%C3%BCnzen as reference for coin sizes
    coin_sizes_mm = {200: 2575, 100: 2325, 50: 2425, 20: 2225, 10: 1975, 5: 2125, 2: 1875, 1: 1625}
    scale_factor = coin_sizes_mm[coin_value] / coin_sizes_mm[200]
    return int(scale_factor * two_euro_reference_size)

def center_crop(img, new_h, new_w):
    old_h, old_w = img.shape[:2]
    off_w = (old_w - new_w) // 2
    off_h = (old_h - new_h) // 2
    return img[off_h:off_h+new_h, off_w:off_w+new_w, :]

def generate_retrieval_image(data_path='data', h=256, w=256, coin_amt_mean=9, do_homographic_transform=True):
    """
    Returns an retrieval image with the given height, width and amount of coins; and a set of labels.
    :param data_path: the location of the dataset
    :param h: height of the generated image
    :param w: width of the generated image
    :param coin_amt_mean: the number of coins
    :param do_homographic_transform: whether to apply an homographic transform
    :return: retrieval image, labels
    """
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
    coin_amt = np.ceil(np.random.normal(coin_amt_mean, scale=2))

    # List of existing coins, given by their center coordinates and radius
    existing_coin_positions = []

    # while #coins < coin_amt
    hashtag_coins = 0
    while hashtag_coins < coin_amt:
        # get coin from data
        coin_value, coin_img = random_coin(data_path=data_path)
        new_size = min(coin_img.shape[0], coin_img.shape[1])
        coin_img = center_crop(coin_img, new_size, new_size)
        assert coin_img.shape[0] == coin_img.shape[1]

        # scale, rotate, remove background
        phi = np.random.randint(0, 360)
        physical_coin_size = get_real_coin_size(coin_value)
        coin_without_background, inv_mask = rotate_extract_coin(coin_img, phi, physical_coin_size)

        # find position for insert,
        # and insert coin into the retrieval image
        coin_radius = (coin_without_background.shape[0] + 1) // 2
        center_y, center_x = get_insertable_position(retrieval_img.shape, coin_radius, existing_coin_positions)
        if center_y is not None:
            existing_coin_positions.append((center_y, center_x, coin_radius))
            insert_coin_to_position(retrieval_img,center_x, center_y, coin_radius, coin_without_background, inv_mask)

            # add coin to label-list
            labels[coin_value] += 1
            labels['sum'] += coin_value
            labels['num_coins'] += 1
        else:
            # could not find a valid position, maybe the image is too full already
            break

        # #coins++
        hashtag_coins += 1

    # geometrischer shift gesamt pic
    if do_homographic_transform:
        retrieval_img = perform_homographic_transform(retrieval_img)

    # return pic, label
    return retrieval_img, labels

def generate_retrieval_dataset(path='retrieval_dataset', num_images=50, format="png", do_homographic_transform=False):
    # Delete the existing dataset
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    metadata = {}
    for img_index in tqdm(range(num_images)):
        try:
            img, meta = generate_retrieval_image(h=256, w=256, coin_amt_mean=9, do_homographic_transform=do_homographic_transform)
            metadata[img_index] = meta
            cv.imwrite(os.path.join(path, f"{img_index}.{format}"), img)
        except Exception:
            pass

    with open(os.path.join(path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def test_images():
    from pathlib import Path

    failed_paths = []
    for path in Path('data').rglob('*.jpg'):
        print(f"trying path {path}")
        img = cv.imread(str(path))
        x = hough_circle_detection(img, "low")


        if x is None:
            failed_paths.append(path.name)

    if len(failed_paths) > 0:
        print("coin extraction failed for the following paths:")
        for path in failed_paths:
            print("  "  + path)

def main():
    generate_retrieval_dataset(path='retrieval_dataset', num_images=50, do_homographic_transform=False)
    #test_images()

if __name__ == "__main__":
    main()