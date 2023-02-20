import os
import re

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf

from image_captioner import IC
from instance_segmentation import InstanceSegmenter
# from IterIC_seg import IterIC
from random_seg import RandomSegmenter
from IC_explaination import Explainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='flickr8k', type=str, required=False, help='image captioning model dataset: coco or flickr8k')
parser.add_argument('--seg_dataset', default='coco', type=str, required=False, help='instance segmentation model dataset: coco or lvis')
parser.add_argument('--seg_backbone', default='r50', type=str, required=False, help='instance segmentation model backbone: r50 or x101')
parser.add_argument('--weights_dir', default='./weights', type=str, required=False, help='directory path for the image captioning models weights')
parser.add_argument('--algorithm', default='quickshift', type=str, required=False, help='low level segmentation algorithm: quickshift, felzenszwalb or slic')
parser.add_argument('--kernel_size', default=20, type=int, required=False, help='low level segmentation kernel size (int)')
parser.add_argument('--max_dist', default=200, type=int, required=False, help='low level segmentation maximum distance (int)')
parser.add_argument('--ratio', default=0.2, type=float, required=False, help='low level segmentation ratio (float)')
parser.add_argument('--image_path', type=str, required=False, help='image path to be explained, it overwrites "test" option')
parser.add_argument('--test_dir', default='./data/test', type=str, required=False, help='directory path for test results, it activates "test" option')
parser.add_argument('--text_encoder', default='bert', type=str, required=False, help='the text encoder that is used in calculating the similarity score: \
                                                                                      count (sklearn CountVectorizer), bert or roberta')


if __name__=='__main__':

    args = parser.parse_args()

    CaptioningModel = IC(dataset=args.dataset, weights_dir=args.weights_dir)
    InstanceSegModel = InstanceSegmenter(dataset=args.seg_dataset, backbone=args.seg_backbone)
    RandomSegModel = RandomSegmenter(args.algorithm, kernel_size=args.kernel_size,
                                     max_dist=args.max_dist, ratio=args.ratio,
                                     random_seed=42)
    ExplainerModel = Explainer(captioner=CaptioningModel, instance_seg=InstanceSegModel, random_seg=RandomSegModel, text_encoder=args.text_encoder)

    if args.image_path:
        ExplainerModel.explain(args.image_path)
    else:
        images_dir = f'{args.test_dir}/{args.dataset}/images_dir/'
        save_dir = f'{args.test_dir}/{args.dataset}/save_dir/'
        ExplainerModel.test_images(images_dir=images_dir, save_dir=save_dir)

    # dataset = 'flickr8k'
    # instance_seg_dataset = 'coco' # coco or lvis
    # instance_seg_backbone = 'r50' # r50 or x101
    # CaptioningModel = IC(dataset=dataset, weights_dir='./weights')
    # InstanceSegModel = InstanceSegmenter(dataset=instance_seg_dataset, backbone=instance_seg_backbone)
    # RandomSegModel = RandomSegmenter('quickshift', kernel_size=20,
    #                                   max_dist=200, ratio=0.2,
    #                                   random_seed=42)
    # ExplainerModel = Explainer(captioner=CaptioningModel, instance_seg=InstanceSegModel, random_seg=RandomSegModel)

    # # image_name = '3'
    # # image_path = f'./XAI/data/original/{image_name}.jpeg'

    # # ExplainerModel.explain(image_path)

    # # Test:
    # # images_dir = f'./XAI/data/test/{dataset}/images_dir/'
    # # save_dir = f'./XAI/data/test/{dataset}/save_dir/'
    # images_dir = f'E:/PhD/XAI/data/test/{dataset}/images_dir/'
    # save_dir = f'E:/PhD/XAI/data/test/{dataset}/instance_seg_{instance_seg_dataset}/save_dir/'
    # ExplainerModel.test_images(images_dir=images_dir, save_dir=save_dir)