import os
# import re

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# import tensorf low as tf

from image_captioner import IC
from instance_segmentation import InstanceSegmenter
from random_seg import RandomSegmenter
from iPIC import Explainer
from utils import log_print

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='flickr8k', type=str, required=False, help='image captioning model dataset: coco or flickr8k')
parser.add_argument('--seg_dataset', default='coco', type=str, required=False, help='instance segmentation model dataset: coco or lvis')
parser.add_argument('--seg_backbone', default='r50', type=str, required=False, help='instance segmentation model backbone: r50 or x101')
parser.add_argument('--weights_dir', default='./weights', type=str, required=False, help='directory path for the image captioning models weights')
parser.add_argument('--algorithm', default='quickshift', type=str, required=False, help='low level segmentation algorithm: quickshift, felzenszwalb or slic')
parser.add_argument('--seg_ksize', default='20', type=str, required=False, help='low level segmentation kernel size (str)')
parser.add_argument('--blur_ksize', default='120', type=str, required=False, help='blurring kernel size or "auto" (str)')
parser.add_argument('--max_dist', default='200', type=str, required=False, help='low level segmentation maximum distance (str)')
parser.add_argument('--ratio', default='0.2', type=str, required=False, help='low level segmentation ratio (str)')
parser.add_argument('--image_path', type=str, required=False, help='image path to be explained, it overwrites "test" option')
parser.add_argument('--test_dir', default='./data/test', type=str, required=False, help='directory path for test results, it activates "test" option')
parser.add_argument('--text_encoder', default='bert', type=str, required=False, help='the text encoder that is used in calculating the similarity score: \
                                                                                      count (sklearn CountVectorizer), bert or roberta')
parser.add_argument('--clip_mode', default='1', type=str, required=False, help='"0": do not use CLIP (PIC) \n \
                                                                                "1": use CLIP in stage 1 \n \
                                                                                "2": use CLIP in stage 2 \n \
                                                                                "both": use CLIP in both stages')
parser.add_argument('--pobj_mode', default='True', choices=('True','False'), required=False, help='applying object of preposition rule (True) or not (False)')
parser.add_argument('--improved_XIC', default='True', choices=('True','False'), required=False, help='using improved XIC (True) or not (False)')
parser.add_argument('--stage_hi_sim', default='True', choices=('True','False'), required=False, help='using stage result with higher similarity (True) or not (False)')
parser.add_argument('--exp_mode', default='False', choices=('True','False'), required=False, help='run all experiment variations\'s (Ture) or not (Flase)')
parser.add_argument('--test_mode', default='False', choices=('True','False'), required=False, help='run a test if segments are available (Ture) or not (Flase)')


if __name__=='__main__':

    args = parser.parse_args()

    dataset = args.dataset
    weights_dir = args.weights_dir

    seg_dataset = args.seg_dataset
    seg_backbone = args.seg_backbone

    algorithm = args.algorithm
    seg_ksize = int(args.seg_ksize)
    max_dist = int(args.max_dist)
    ratio = float(args.ratio)

    text_encoder = args.text_encoder
    blur_ksize = args.blur_ksize
    clip_mode = args.clip_mode
    pobj_mode = args.pobj_mode == 'True'
    improved_XIC = args.improved_XIC == 'True'
    stage_hi_sim = args.stage_hi_sim == 'True'

    exp_mode = args.exp_mode == 'True'
    test_mode = args.test_mode == 'True'

    test_dir = args.test_dir

    if exp_mode:
        log_print('Experiments mode :::')
        proceed = input('48 experiment will be executed!! Do you want to proceed? (y/n)')
        if proceed.lower()=='y':

            seg_datasets = ['coco', 'lvis']
            clip_modes = ['0', '1', 'both']
            pobj_modes = ['True', 'Flase']
            stage_hi_sims = ['True', 'Flase']
            blur_ksizes = ['auto', '120']

            for seg_dataset in seg_datasets:
                for clip_mode in clip_modes:
                    for pobj_mode in pobj_modes:
                        for stage_hi_sim in stage_hi_sims:
                            for blur_ksize in blur_ksizes:
                                log_print(f'Running: dataset={dataset}, seg_dataset={seg_dataset}, clip_mode={clip_mode}, pobj_mode={str(pobj_mode)}, blur_ksize={blur_ksize}, stage_hi_sim={stage_hi_sim}')

                                CaptioningModel = IC(dataset=dataset, weights_dir=weights_dir)

                                InstanceSegModel = InstanceSegmenter(dataset=seg_dataset, backbone=seg_backbone)

                                RandomSegModel = RandomSegmenter(algo_type=algorithm, kernel_size=seg_ksize, \
                                                                max_dist=max_dist, ratio=ratio, random_seed=42)

                                ExplainerModel = Explainer(captioner=CaptioningModel, instance_seg=InstanceSegModel, random_seg=RandomSegModel, \
                                                        text_encoder=text_encoder, blur_ksize=blur_ksize, clip_mode=clip_mode, stage_hi_sim=stage_hi_sim, \
                                                        pobj_mode=pobj_mode, improved_XIC=improved_XIC)


                                images_dir = f'{test_dir}/{seg_dataset}/images_dir/'

                                exp_details = f'{blur_ksize}_{clip_mode}_{str(pobj_mode)}_{str(improved_XIC)}_{str(stage_hi_sim)}'
                                save_dir = f'{test_dir}/{seg_dataset}/save_dir/{exp_details}/'
                                
                                run_test = False

                                if not os.path.isdir(save_dir):
                                    log_print(f"creating new directory: {save_dir}")
                                    os.makedirs(save_dir)
                                    run_test = True
                                else:
                                    user_input = input(f"Directory: {save_dir} is already available, do you want to overwrite the content? (Y/N)\n")
                                    if (user_input.lower() == 'y') or (user_input.lower() == 'yes'):
                                        log_print(f"overwriting available directory: {save_dir}")
                                        run_test = True
                                    
                                if run_test: ExplainerModel.test_images(images_dir=images_dir, save_dir=save_dir)

                                log_print(f'Finished: dataset={dataset}, seg_dataset={seg_dataset}, clip_mode={clip_mode}, pobj_mode={str(pobj_mode)}, blur_ksize={blur_ksize}, stage_hi_sim={stage_hi_sim}')
    elif test_mode:
        pass

    else:

        CaptioningModel = IC(dataset=dataset, weights_dir=weights_dir)

        InstanceSegModel = InstanceSegmenter(dataset=seg_dataset, backbone=seg_backbone)

        RandomSegModel = RandomSegmenter(algo_type=algorithm, kernel_size=seg_ksize, \
                                        max_dist=max_dist, ratio=ratio, random_seed=42)

        ExplainerModel = Explainer(captioner=CaptioningModel, instance_seg=InstanceSegModel, random_seg=RandomSegModel, \
                                text_encoder=text_encoder, blur_ksize=blur_ksize, clip_mode=clip_mode, stage_hi_sim=stage_hi_sim, \
                                pobj_mode=pobj_mode, improved_XIC=improved_XIC)

        if args.image_path:
            ExplainerModel.explain(args.image_path)
        else:
            images_dir = f'{args.test_dir}/{args.dataset}/images_dir/'

            exp_details = f'{args.blur_ksize}_{args.clip_mode}_{str(pobj_mode)}_{str(improved_XIC)}'
            save_dir = f'{args.test_dir}/{args.dataset}/save_dir/{exp_details}/'

            run_test = False

            if not os.path.isdir(save_dir):
                print(f"creating new directory: {save_dir}")
                os.makedirs(save_dir)
                run_test = True
            else:
                user_input = input(f"Directory: {save_dir} is already available, do you want to overwrite the content? (Y/N)\n")
                if (user_input.lower() == 'y') or (user_input.lower() == 'yes'):
                    run_test = True
                
            if run_test: ExplainerModel.test_images(images_dir=images_dir, save_dir=save_dir)