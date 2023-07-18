import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from utils import log_print


def XIC(image_id, query_id, image, query, answer, captioner=None, clip_model=None, mode='XIC', b_ksize=120, save_dir='.\\test_results'):
    score = 0
    if (mode=='XIC') or (mode=='iXIC') or (mode=='iXIC_clip'):
        test_images = generate_test_images(image_id, query_id, image, answer, mode, b_ksize, save_dir)
        score, test_images_captions = evaluate_good_answer(image, test_images, query, captioner, clip_model, mode)
    else:
        log_print("Wrong metric mode! Only the following modes are available: [XIC, iXIC, iXIC_clip]")
        log_print("Abort!")
    return score, test_images_captions

def generate_test_images(image_id, query_id, image, answer, mode, b_ksize, save_dir):
    test_images = list()
    mask = answer

    if mode=='XIC':
        image = image/255.
        shape = image.shape
        black = np.full(shape, 0.0)
        white = np.full(shape, 1.0)
        grey = np.random.normal(loc=0.5, scale=0.1, size=shape)
        grey = np.clip(grey, 0, 1)

        black[mask] = image[mask]
        white[mask] = image[mask]
        grey[mask] = image[mask]

        test_images = [black*255, white*255, grey*255]

        plt.imsave(os.path.join(save_dir, f'{image_id}_{query_id}_black.jpg'), black)
        plt.imsave(os.path.join(save_dir, f'{image_id}_{query_id}_white.jpg'), white)
        plt.imsave(os.path.join(save_dir, f'{image_id}_{query_id}_grey.jpg'), grey)

    elif (mode=='iXIC') or (mode=='iXIC_clip'):
        blurred = cv2.blur(image, (b_ksize, b_ksize))
        blurred[mask] = image[mask]

        test_images = [blurred]

        plt.imsave(os.path.join(save_dir, f'{image_id}_{query_id}_blurred.jpg'), blurred)

    return test_images

def evaluate_good_answer(image, test_images, query, captioner, clip_model, mode='XIC'):
    score = 0
    if (mode=='XIC') or (mode=='iXIC'):
        captions = list()
        for test_image in test_images:
            caption = captioner.caption_image(test_image)
            captions.append(caption)
            if query in caption.lower():
                score = score or 1
            else:
                score = score or 0
        if mode=='iXIC': captions=captions*3
        score = score
    elif mode=='iXIC_clip':
        captions = ['CLIP']*3
        max_score = clip_model.measure_similarity(image, query)[0][0]
        query_score = clip_model.measure_similarity(test_images[0], query)[0][0]
        score = f'{str(query_score)}/{str(max_score)}'
    return score, captions
