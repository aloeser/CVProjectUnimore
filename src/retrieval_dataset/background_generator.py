import cv2 as cv
import numpy as np

def grey_bg_img(bg_grey, ch, x=256, y=256):
    """
    Creates a background image with a given grey-value or chooses one randomly through the "rnd"-option.
    :param bg_grey: background grey-value in (0 - 255)
    :param ch: amount of picture channels (1 - 3)
    :param x: width
    :param y: height
    :return: grey background image
    """
    if bg_grey == "rnd":
        rnd_value = -1
        while rnd_value < 0:
            rnd_value = np.random.normal(209, 23)  # 2 sigma / 95.45% (163, 255)
            # rnd_value = np.random.normal(210,15) # 3 sigma / 99.73% (165, 255)
            bg_grey = min(255, int(rnd_value))
    gr_bg_img = np.full((y, x, ch), bg_grey, np.uint8)
    return gr_bg_img

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

def gen_background(x=256, y=256):
    """
    Generate a brown 3-channel uint8 background image (randomized with specific parameters).
    :param x: width
    :param y: height
    :return: background image
    """
    # 1 channel structure
    tmp = add_noise(texture(grey_bg_img("rnd", ch=1, x=x, y=y), sigma=5, turbulence=2), sigma=5).astype(np.uint8)
    #cv.imshow("1ch structure", tmp)

    # 3 channel structure
    structure = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
    # cv.imshow("3ch structure", structure)

    # brown BGR_colors
    bgr_burlywood   = (135, 184, 222)
    bgr_tan         = (140, 180, 210)
    bgr_sandybrown  = (96, 164, 244)
    bgr_peru        = (63, 133, 205)
    bgr_chocolate   = (30, 105, 210)
    bgr_saddlebrown = (19, 69, 139)
    bgr_sienna      = (45, 82, 160)
    bgr_brown       = (42, 42, 165)
    bgr_maroon      = (0, 0, 128)

    # 3 channel brown picture
    brown_pic = grey_bg_img(bgr_sienna, 3)
    # cv.imshow("brown pic", brown_pic)

    # Blending: brown picture + structure
    alpha = 0.7
    beta = (1.0 - alpha)
    # dst = alpha*(img1) + beta*(img2) + gamma
    dst = cv.addWeighted(src1=brown_pic, alpha=alpha, src2=structure, beta=beta, gamma=-40.0)

    return dst


def main():
    test_background = gen_background()
    cv.imwrite("test background.jpg", test_background)

    """# creating a random grey background
    grey_bg_img1 = grey_bg_img("rnd", ch=1)
    print("shape", grey_bg_img1.shape, "grey value: ", grey_bg_img1[0,0,0])
    cv.imshow("grey_bg_img", grey_bg_img1)


    ## adding noise
    # ratio 1
    grey_bg_img2 = add_noise(grey_bg_img1, sigma=10, ratio=1)
    #print(grey_bg_img2[0:3,0:3,0:3])
    cv.imwrite('noise2.jpg', grey_bg_img2)

    # ratio 2
    grey_bg_img3 = add_noise(grey_bg_img1, sigma=10, ratio=4)
    # print(grey_bg_img2[0:3,0:3,0:3])
    cv.imwrite('noise3.jpg', grey_bg_img3)


    # texture pic
    cv.imwrite('texture1.jpg', texture(grey_bg_img1, sigma=4, turbulence=4))

    # texture and noise
    cv.imwrite('texture-and-noise.jpg', add_noise(texture(grey_bg_img1, sigma=4, turbulence=2), sigma=10))

    # PARAM CHECK POINT
    # = add_noise(texture(grey_bg_img("rnd", ch=3), sigma=4, turbulence=2), sigma=10)
    cv.imwrite('final.jpg', add_noise(texture(grey_bg_img("rnd", ch=3), sigma=4, turbulence=2), sigma=10))
    """
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()


"""
structure = add_noise(texture(grey_bg_img("rnd", ch=3, x=x, y=y), sigma=10, turbulence=2), sigma=10).astype(np.uint8)
    hsv = cv.cvtColor(structure, cv.COLOR_BGR2HSV)

    # brown BGR_colors
    bgr_burlywood   = (135, 184, 222)
    bgr_tan         = (140, 180, 210)
    bgr_sandybrown  = (96, 164, 244)
    bgr_peru        = (63, 133, 205)
    bgr_chocolate   = (30, 105, 210)
    bgr_saddlebrown = (19, 69, 139)
    bgr_sienna      = (45, 82, 160)
    bgr_brown       = (42, 42, 165)
    bgr_maroon      = (0, 0, 128)

    saddlebrown = np.uint8([[[19, 69, 139]]])
    hsv_saddlebrown = cv.cvtColor(saddlebrown, cv.COLOR_BGR2HSV)
    h, s, v = hsv_saddlebrown[0,0]
    print("saddle hsv: ", h, s, v)

    test_pic = grey_bg_img(bgr_saddlebrown, 3)
    cv.imshow("test pic", test_pic)

    print("curr hsv", hsv[50, 50, :])
    #print("pic shape", hsv.shape)
    # hue (dominant wavelength), saturation (purity), value (intensity); H: 0-179, S: 0-255, V: 0-255
    curr_h = hsv[50, 50, 0]
    curr_s = hsv[50, 50, 1]
    curr_v = hsv[50, 50, 2]

    hsv[:, :, 0] += h - curr_h
    hsv[:, :, 1] += s - curr_s
    hsv[:, :, 2] += v - curr_v

    print(hsv[50,50,:])
    # background = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
"""
