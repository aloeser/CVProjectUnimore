import os.path
import json

import cv2 as cv

from coin_segmentation.pre_cnn_segmentation import predict


class RetrievalEngine:

    def __init__(self, path):
        self.path = path
        with open(os.path.join(path,  'annotations.json')) as f:
            self.annotations = json.load(f)

    def most_similar(self, file, first_k):        # for now: using sum of coin values as similarity metric
        image = cv.imread(file)
        target_coin_value_sum = predict(image)['sum']

        # TODO: optimization potential: only store the first k
        scores = []
        for img_name in self.annotations:
            coin_value_sum = self.annotations[img_name]['sum']
            scores.append((abs(target_coin_value_sum - coin_value_sum), coin_value_sum, os.path.join(self.path, img_name + ".png")))

        scores = sorted(scores)
        return scores[:first_k]