import cv2 as cv
import numpy as np
import torch
from moresnet import MOREsNet

def which_coin(pred_vector):
    """
    Gets predicted output vector and returns cent accordingly to highest probability.
    :param pred_vector: CNN predicted vector
    :return: cent value prediction of coin
    """
    # {'1': 0, '10ct': 1, '1ct': 2, '2': 3, '20ct': 4, '2ct': 5, '50ct': 6, '5ct': 7}
    cents = [100, 10, 1, 200, 20, 2, 50, 5]
    pos = np.argmax(pred_vector)
    cent = cents[pos]
    # print('pos', pos, '\ncoin', coin)
    return cent

def run_model(model, inp_pic):
    """
    Takes a single 64x64x3 coin picture, runs the given model and predicts coin value.
    :param model: model with pre-trained weights
    :param inp_pic: input picture of single coin
    :return: cent value prediction of coin
    """
    model.eval()
    with torch.no_grad():
        pred_vector = np.array(model(inp_pic)[0])
        # print('pred_vector', pred_vector)
    cent = which_coin(pred_vector)
    return cent

def img_preprocessing(img):
    """
    Takes input image from coin segmentation output vector and preprocesses for CNN model.
    :param img: single 64x64x3 coin image
    :return: CNN ready input image
    """
    img = img[..., ::-1] / 255
    input_img = torch.from_numpy(img.copy())
    input_img = torch.moveaxis(input_img, -1, 0).unsqueeze(0).float()
    return input_img

def get_prediction(img):
    """
    Predicts the cent value for a single coin picture through running a given model including image preprocessing.
    :param img: single 64x64x3 coin image
    :return: cent value prediction of coin
    """
    # load model and get predicted CNN output vector
    model = MOREsNet(in_channels=3, num_classes=8, load_pretrained='./moresnet.tar')
    input_pic = img_preprocessing(img)
    cent = run_model(model, input_pic)
    return cent

def main():
    # load model and get predicted CNN output vector for example picture
    model = MOREsNet(in_channels=3, num_classes=8, load_pretrained='./moresnet.tar')
    img = cv.imread('./cnn_dataset/root/1/italy-1-euro-2011-front.png')
    input_pic = img_preprocessing(img)
    cent = run_model(model, input_pic)

    # print CNN prediction, input image
    print(cent, 'cent')
    cv.imshow('inp_img', img)
    # windows management
    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
