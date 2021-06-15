import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from PIL import Image
from ellipse_detection import segment_coin, center_crop


def get_label(img_name, label_map):
    key = ''
    for i in [1, 2, 5, 10, 20, 50]:
        q = (str(i) + '-euro')
        if q in img_name:
            key = q
            if 'cent' in img_name:
                key = key + '-cent'
            return label_map[key]
    # unknown label
    return -1


def process_images_and_save(file_names, out_res, out_dir):
    label_dict = {
        '1-euro-cent': 0,   '2-euro-cent': 1,   '5-euro-cent': 2,   '10-euro-cent': 3,
        '20-euro-cent': 4,  '50-euro-cent': 5,  '1-euro': 6,        '2-euro': 7
    }
    dirname_dict = {
        0: '1ct',   1: '2ct',   2: '5ct',   3: '10ct', 
        4: '20ct',  5: '50ct',  6: '1',     7: '2' 
    }
    for fname in tqdm(file_names):
        img = cv.imread(fname)
        h, w, c = img.shape
        img = center_crop(img, min(h, w), min(h, w))
        h, w, c = img.shape
        mask = segment_coin(img).astype(np.bool)
        img_out = np.zeros((h, w, c), dtype=np.uint8)
        img_out[mask] = img[mask]
        img_out = cv.GaussianBlur(img_out, (5, 5), 0, 0)
        img_out = cv.resize(img_out, out_res, cv.INTER_AREA)
        img_out = cv.cvtColor(img_out, cv.COLOR_BGR2RGB)
        # img_out = img_out.flatten()
        label = get_label(fname, label_dict)
        dir_to_save = out_dir + dirname_dict[label]
        if not os.path.isdir(dir_to_save):
            os.makedirs(dir_to_save)
        img = Image.fromarray(img_out)
        img.save(dir_to_save + '/' + fname.split('/')[-1].replace('jpg', 'png'))


def make_image_folder(data_dir, out_res, out_dir):
    dir_to_save = out_dir
    if dir_to_save[-1] != '/':
        dir_to_save += '/'
    if os.path.isdir(dir_to_save):
        print('Directory already exists! Exiting...')
        exit()
    elif not os.path.isdir(dir_to_save):
        dir_to_save = dir_to_save + 'root/'
        os.makedirs(dir_to_save)
    # country = 'Germany'
    file_names = [path.replace('\\', '/') + '/' + f 
                  for path, _, files in os.walk(data_dir) 
                  # if country in path 
                  for f in files]
    process_images_and_save(file_names, out_res, dir_to_save)


if __name__ == '__main__':
    data_dir = './data'
    out_dir = './cnn_dataset'
    out_res = (150, 150)
    make_image_folder(data_dir, out_res, out_dir)
