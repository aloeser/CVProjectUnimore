import argparse
import cv2 as cv
import json
import os
import tqdm

from coin_segmentation.pre_cnn_segmentation import predict
from retrieval_dataset.build_retrieval_dataset import generate_retrieval_dataset

def annotate_retrieval_database(path):
    retrieval_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.png')]
    annotations = {}
    for f in tqdm.tqdm(retrieval_files, desc='Dataset Annotation'):
        image = cv.imread(f)
        file_name = os.path.splitext(os.path.basename(f))[0]
        #print(f"trying to annotate {file_name}")
        annotations[file_name] = predict(image)

    with open(os.path.join(path, 'annotations.json'), 'w') as out:
        json.dump(annotations, out, indent=2)


def build_retrieval_database(path, num_images, homographic_transform):
    generate_retrieval_dataset(path=path, num_images=num_images, do_homographic_transform=homographic_transform)
    annotate_retrieval_database(path)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input file', nargs=1,
                        help='name of the input file')
    parser.add_argument('--build_retrieval_database', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    build_retrieval_database('retrieval_database', 200, False)
    #args = parser.parse_args()
    #print(args.accumulate(args.integers))

if __name__ == "__main__":
    main()