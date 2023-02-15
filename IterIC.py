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
parser.add_argument('--name', type=str, required=True)



if __name__=='__main__':

    args = parser.parse_args()
    print('Hello,', args.name)

    # dataset = 'flickr8k'
    # instance_seg_dataset = 'coco' # coco or lvis
    # instance_seg_backbone = 'r50' # r50 or x101
    # CaptioningModel = IC(dataset=dataset, weights_dir='./XAI/Iterative/weights')
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