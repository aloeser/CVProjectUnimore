import torch
import cv2 as cv
import matplotlib.pyplot as plt
from moresnet import MOREsNet


def main():
    img = cv.imread('./cnn_dataset/root/1/italy-1-euro-2011-front.png')
    img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
    img = img[...,::-1] / 255
    plt.imshow(img)
    plt.waitforbuttonpress()
    input = torch.from_numpy(img.copy())
    input = torch.moveaxis(input, -1, 0).unsqueeze(0).float()
    print(input.shape)
    print(input.dtype)
    model = MOREsNet(in_channels=3, num_classes=8, load_pretrained='./moresnet.tar')
    model.eval()
    with torch.no_grad():
        print(model(input))
    # {'1': 0, '10ct': 1, '1ct': 2, '2': 3, '20ct': 4, '2ct': 5, '50ct': 6, '5ct': 7}


if __name__ == '__main__':
    main()
