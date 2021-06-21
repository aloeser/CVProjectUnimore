import random
import cv2 as cv
import numpy as np

def create_bg_img(col_value, ch, x=256, y=256):
    """
    Creates a background image with a given (color) value or chooses a random grey value through the "rnd"-option.
    :param col_value: color value (for example BGR) or grey-value in (0 - 255)
    :param ch: amount of picture channels (1 or 3)
    :param x: width
    :param y: height
    :return: background image
    """
    if col_value == "rnd":
        rnd_value = -1
        while rnd_value < 0:
            rnd_value = np.random.normal(209, 23)  # 2 sigma / 95.45% (163, 255)
            # rnd_value = np.random.normal(210,15) # 3 sigma / 99.73% (165, 255)
            col_value = min(255, int(rnd_value))   # grey value
    bg_img = np.full((y, x, ch), col_value, np.uint8)
    return bg_img

def add_noise(inp_img, sigma, ratio=1):
    """
    Adding gaussian noise image (mean = 0, given sigma) to the input image.
    If ratio parameter is specified,
    noise will be generated for a lesser image and then it will be up-scaled to the original size.
    In that case noise will generate larger square patterns.
    To avoid multiple lines, the upscale uses interpolation.

    :param inp_img: input image
    :param sigma: gaussian noise sigma
    :param ratio: scale ratio for noise
    """
    x, y, ch = inp_img.shape

    # up-scaling for bigger noise particles
    if ratio > 1:
        h = int(y / ratio)
        w = int(x / ratio)
        small_noise = np.random.normal(0, sigma, (w, h, ch))
        noise = cv.resize(small_noise, dsize=(x, y), interpolation=cv.INTER_LINEAR)\
            .reshape((x, y, ch))
    else:
        noise = np.random.normal(0, sigma, (x, y, ch))

    out_img = np.clip(inp_img + noise, 0, 255)
    return out_img

def texture(inp_img, sigma, turbulence):
    """
    Consequently applies noise patterns to the original image from big to small.

    :param inp_img: input image
    :param sigma: defines bounds of noise fluctuations
    :param turbulence: defines how quickly big patterns will be replaced with the small ones.
                       The lower value - the more iterations will be performed during texture generation.
    :return: output image
    """
    x, y, ch = inp_img.shape

    ratio = x
    while not ratio == 1:
        inp_img = add_noise(inp_img, sigma, ratio)
        ratio = (ratio // turbulence) or 1
    out_img = np.clip(inp_img, 0, 255)
    return out_img

def gen_background(x=256, y=256, print_values=False):
    """
    Generate a brown 3-channel uint8 background image (randomized with specific parameters).
    :param x: width
    :param y: height
    :param print_values: printing all random parameters
    :return: background image
    """
    # random sigma for add_noise
    noise_sigma = random.choice([0 , 1, 5, 10, 20, 30])

    # random sigma for texture
    texture_sigma = random.choice([4, 5, 6])

    # random turbulance for texture
    turb_value = random.choice([2, 3, 4, 10, 100]) # no zero or infinite runtime

    # 1 channel grey structure
    # tmp = add_noise(texture(create_bg_img("rnd", ch=1, x=x, y=y), sigma=5, turbulence=2), sigma=5).astype(np.uint8)
    tmp = add_noise(texture(create_bg_img("rnd", ch=1, x=x, y=y), sigma=texture_sigma, turbulence=turb_value),
                    sigma=noise_sigma).astype(np.uint8)
    # cv.imshow("1ch structure", tmp)

    # 3 channel grey structure
    grey_bg_structure = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
    # cv.imshow("3ch structure", grey_bg_structure)

    # pick random background color from brown BGR_colors
    bgr_burlywood   = (135, 184, 222)
    bgr_tan         = (140, 180, 210)
    bgr_sandybrown  = (96, 164, 244)
    bgr_peru        = (63, 133, 205)
    bgr_chocolate   = (30, 105, 210)
    bgr_saddlebrown = (19, 69, 139)
    bgr_sienna      = (45, 82, 160)
    bgr_brown       = (42, 42, 165)
    bgr_maroon      = (0, 0, 128)
    # hue (dominant wavelength), saturation (purity), value (intensity); H: 0-179, S: 0-255, V: 0-255
    brown_colors = [bgr_burlywood, bgr_tan, bgr_sandybrown, bgr_peru, bgr_chocolate,
                        bgr_saddlebrown, bgr_sienna, bgr_brown, bgr_maroon]
    bgr_color = random.choice(brown_colors)

    # 3 channel brown picture
    brown_pic = create_bg_img(bgr_color, 3)
    # cv.imshow("brown pic", brown_pic)

    # Blending: brown picture + structure with random alpha
    alpha = np.random.normal(0.7, 0.1)
    beta = (1.0 - alpha)
    # dst = alpha*(img1) + beta*(img2) + gamma
    dst = cv.addWeighted(src1=brown_pic, alpha=alpha, src2=grey_bg_structure, beta=beta, gamma=-40.0)

    if print_values:
        print('noise variance:', noise_sigma,
          '\ntexture noise variance:', texture_sigma,
          '\ntexture turbulence value:', turb_value,
          '\nbrown background color:', bgr_color,
          '\nblending value (percent):', alpha)

    return dst


def main():
    # testing background generation function
    test_background = gen_background()
    cv.imwrite("test background.jpg", test_background)

if __name__ == "__main__":
    main()
