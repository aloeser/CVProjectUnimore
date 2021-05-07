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


def main():
    # creating a random grey background
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
    cv.imwrite('final.jpg', add_noise(texture(grey_bg_img("rnd", ch=1), sigma=4, turbulence=2), sigma=10))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
