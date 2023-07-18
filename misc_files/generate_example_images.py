import pickle as pkl
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np



def return_proposals(file_path, stage='stage 1'):
    segments_dict = pkl.load(open(file_path,'rb'))
    segments = segments_dict[f'{stage} segments']
    print(f'{len(segments)} segments are found')
    return segments

def get_proposed_mask(file_path, stage='stage 2'):
    segments_dict = pkl.load(open(file_path,'rb'))
    return segments_dict[f'{stage} proposed_mask']

def save_proposals(image_id, seg_id='', ds='lvis', stage='stage 1'):
    b_ksize = 20
    seg='instance' if stage=='stage 1' else 'low_level'

    images_dir = f'.\\data\\test\\{ds}\\images_dir'
    save_dir = f'C:\\Me\\PHD\\00PhD_Thesis\\00Publications\\2023_SACI\\presentation\\figs\\combinations\\{seg}'
    answers_dir = f'E:\\PhD\\Publications\\2023_SACI\\test_data\\data\\test\\flickr8k\\instance_seg_{ds}\save_dir'

    image_file = os.path.join(images_dir, f'{image_id}.jpg')
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if stage=='stage 1':
        proposals_file = os.path.join(answers_dir, f'{image_id}_{stage}.bin')
        proposals_list = return_proposals(proposals_file, stage)
        for idx, proposal in enumerate(proposals_list):
            blurred = cv2.blur(image, (b_ksize, b_ksize))
            blurred[proposal[-1]] = image[proposal[-1]]
            plt.imsave(os.path.join(save_dir, f'{image_id}_{idx}_blurred.jpg'), blurred)
        
        proposed_mask = get_proposed_mask(proposals_file, stage=stage)
        blurred = cv2.blur(image, (b_ksize, b_ksize))
        blurred[proposed_mask] = image[proposed_mask]
        plt.imsave(os.path.join(save_dir, f'{image_id}_answer.jpg'), blurred)

    elif stage=='stage 2':
        image_id = f'{image_id}_{seg_id}'
        proposals_file = os.path.join(answers_dir, f'{image_id}_{stage}.bin')
        proposals_list = return_proposals(proposals_file, stage)
        segments = proposals_list

        mask = get_proposed_mask(proposals_file, stage=stage)

        seg_proposals = np.full(image.shape, 0)
        seg_proposals[mask==True] = segments[mask==True]
        masks_ids = np.unique(seg_proposals)

        print(f'{len(masks_ids)} segments are found')
        for mask_id in masks_ids:
            proposal = np.full(seg_proposals.shape, False)
            proposal[seg_proposals==mask_id] = True
            proposal[mask!=True] = False # remove BG when mask_id is zero

            blurred = cv2.blur(image, (b_ksize, b_ksize))
            blurred[proposal] = image[proposal]

            
            plt.imsave(os.path.join(save_dir, f'{image_id}__{seg_id}_{mask_id}.jpg'), blurred)
        
        proposed_mask = get_proposed_mask(proposals_file, stage=stage)
        blurred = cv2.blur(image, (b_ksize, b_ksize))
        blurred[proposed_mask] = image[proposed_mask]
        plt.imsave(os.path.join(save_dir, f'{image_id}_{seg_id}_answer.jpg'), blurred)
            




if __name__=="__main__":
    image_id = '18'
    seg_id = '0'
    ds = 'lvis'
    stage='stage 2'
    save_proposals(image_id, seg_id, ds, stage)
