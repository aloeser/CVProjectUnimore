import argparse
import cv2 as cv
import json
import os
import tqdm

from coin_segmentation.pre_cnn_segmentation import predict
from retrieval_dataset.build_retrieval_dataset import generate_retrieval_dataset
from retrieval_engine import RetrievalEngine

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


def generate_retrieval_database(path, num_images, homographic_transform, annotate_only):
    if not annotate_only:
        generate_retrieval_dataset(path=path, num_images=num_images, do_homographic_transform=homographic_transform)
    annotate_retrieval_database(path)


def perform_retrieval(file, retrieval_database_path, first_k,  dont_show_output):
    # TODO: if we call this function for more than one image, the retrieval engine should be stored
    retrieval_engine = RetrievalEngine(retrieval_database_path)
    first_k = retrieval_engine.most_similar(file, first_k)

    if not dont_show_output:
        query_img = cv.imread(file)
        target_coin_value_sum = predict(query_img)['sum']
        cv.imshow(f"Query Image, Sum: {target_coin_value_sum}", query_img)
        for position, (score, coin_sum, img_file) in enumerate(first_k):
            img = cv.imread(img_file)
            cv.imshow(f"Position {position+1}, Sum: {coin_sum}, Score: {score}, File: {os.path.basename(img_file).split('.')[0]}", img)
        cv.waitKey()
        cv.destroyAllWindows()
    else:
        return first_k

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="subparser_name")
    generator_parser = subparsers.add_parser('generate')
    generator_parser.add_argument('--path', help='path where the generated retrieval database is stored', default='retrieval_database')
    generator_parser.add_argument('-n', '--num_images', type=int, default=200)
    generator_parser.add_argument('--homographic_transform', action='store_true')
    generator_parser.add_argument('--annotate_only', action='store_true')

    retrieval_parser = subparsers.add_parser('run')
    retrieval_parser.add_argument('--input',
                                  help='path of the input image',
                                  required=True)
    retrieval_parser.add_argument('--retrieval_database_path', help='path where the generated retrieval database is stored',
                                  default='retrieval_database')
    retrieval_parser.add_argument('-k', '--first_k', type=int, default=10)
    retrieval_parser.add_argument('--dont_show_output', action='store_true')
    #bar_parser = subparsers.add_parser('retrieve')
    #build_retrieval_database('retrieval_database', 200, False, False)
    #perform_retrieval('retrieval_dataset/0.png', 'retrieval_database', 10, True)
    args = parser.parse_args()

    if args.subparser_name == "generate":
        generate_retrieval_database(path=args.path, num_images=args.num_images, homographic_transform=args.homographic_transform, annotate_only=args.annotate_only)
    elif args.subparser_name == "run":
        perform_retrieval(file=args.input, retrieval_database_path=args.retrieval_database_path, first_k=args.first_k, dont_show_output=args.dont_show_output)
    else:
        raise Exception("Invalid command, only generate and run are supported")



if __name__ == "__main__":
    main()