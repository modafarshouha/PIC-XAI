import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from time import perf_counter, gmtime, strftime
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

from utils import Chunker, log_print


class Explainer():
    def __init__(self, captioner, instance_seg, random_seg, text_encoder) -> None:
        self.CaptioningModel = captioner
        self.InstanceSegModel = instance_seg
        self.RandomSegModel = random_seg
        self.text_encoder = text_encoder

        self.image_path = None
        self.main_caption = None
        self.query = None
        self.image = None
        self.blurred = None

        self.current_stage = None
        self.current_stage_proposals = None

        self.stages_results = self.initialize_stages_results()

        self.chunker = Chunker()
        self.stops = stopwords.words('english')

        self.test_time = None

    def initialize_stages_results(self):
        stage_dict = {'selected_ids ': [], 'max_score': [], 'proposed_mask': [], 'proposal': [], 'segments':[], 'captions':[]}
        results_dict = {'stage 1': stage_dict.copy(), 'stage 2': stage_dict.copy()}
        return results_dict
    
    def explain(self, image_path, mode='loop', test_query='', save_dir='', image_id='', query_id=''):
        loop_cond = True
        self.validate_image_path(image_path)
        if self.image_path:
            if not self.main_caption: self.caption_image(self.image, main=True)
            # self.get_query() if mode=='loop' else self.set_query()
            while(loop_cond):
                if mode=='loop':
                    loop_cond = self.get_query()
                    if not loop_cond: break
                elif mode=='test':
                    self.query = test_query.lower()
                    loop_cond = False

                log_print(f"Query is {self.query}")
                log_print("Query accepted!")
                log_print("Generating segments!")
                start = perf_counter()
                self.instance_segmentation()
                self.random_segmentation()
                pref_time = perf_counter() - start
                log_print(f"Segments generation is done in less than {int(pref_time/60)+1} min ({pref_time} sec)!")
                start = perf_counter()
                log_print("Stage 1 ...")
                self.stageI()
                pref_time = perf_counter() - start
                log_print(f"Stage 1 is done in less than {int(pref_time/60)+1} min ({pref_time} sec)!")
                start = perf_counter()
                self.stageII()
                pref_time = perf_counter() - start
                log_print(f"Stage 2 is done in less than {int(pref_time/60)+1} min ({pref_time} sec)!")
                self.answer(save_dir, image_id, query_id)

    def validate_image_path(self, image_path):
        if os.path.isfile(image_path):
            self.image_path = image_path
            image = cv2.imread(self.image_path)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.blurred = cv2.blur(self.image, (20, 20))
        else:
            log_print('Image was not found!')

    def caption_image(self, image, main=False):
        caption = self.CaptioningModel.caption_image(image).lower()
        if main:
            self.main_caption = caption
            log_print(f'\nOriginal caption:\n{self.main_caption}\n')
        return caption
    
    def get_query(self):
        good_query = False
        while(not good_query):
            query = input('Enter query (from the original caption) or "NAN" to exit:\n').lower().strip()
            if (query != '') and (re.search(r"\b{}\b".format(query), self.main_caption.strip())):
                good_query = True
            elif query=='nan':
                log_print("Exit!")
                return False
            else:
                log_print(f'"{query}" is not a valid query!')
        self.query = query
        return True

    def instance_segmentation(self):
        if len(self.stages_results['stage 1']['segments'])==0:
            log_print("stage 1 segments are NOT available!")
            self.InstanceSegModel.predict(self.image)
            self.InstanceSegModel.create_proposals()
            self.stages_results['stage 1']['segments'] = self.InstanceSegModel.proposals
        else:
            log_print("stage 1 segments are available!")
            self.InstanceSegModel.proposals = self.stages_results['stage 1']['segments']

    def random_segmentation(self):
        if len(self.stages_results['stage 2']['segments'])==0:
            log_print("stage 2 segments are NOT available!")
            self.RandomSegModel.create_segments(self.image)
            self.stages_results['stage 2']['segments'] = self.RandomSegModel.segments
        else:
            log_print("stage 2 segments are available!")
            self.RandomSegModel.segments = self.stages_results['stage 2']['segments']

    def stageI(self):
        self.current_stage = 'stage 1'
        self.current_stage_proposals = None
        self.current_stage_proposals = self.InstanceSegModel.proposals
        self.nominate_proposal()

        plt.imshow(self.stages_results[self.current_stage]['proposal'])
        plt.title('Stage 1 proposal')
        plt.show()

    def nominate_proposal(self):
        ids, scores = self.calculate_similarity_scores()
        self.select_highest(ids, scores)
        self.merge_proposals() # useful when there is more than region with the max similarity score

    def calculate_similarity_scores(self):
        scores = list()
        ids = list()
        print("\n\n\n")
        log_print(f'{self.current_stage} calculate_similarity_scores starts')
        for idx, image_size, mask_size, proposed_segment, _ in tqdm(self.current_stage_proposals):
            ids.append(idx)
            if len(self.stages_results[self.current_stage]['captions'])>idx:
                caption = self.stages_results[self.current_stage]['captions'][idx]
            else:
                caption = self.caption_image(proposed_segment)
                self.stages_results[self.current_stage]['captions'].append(caption) 
            # weight = mask_size/image_size if self.current_stage=='stage 1' else 1
            weight = (mask_size/image_size) if self.current_stage=='stage 1' else 1 # Moda
            sim_score = self.calculate_similarity_score(caption, self.query, remove_stops=False)[0][0]
            # log_print(caption)
            # log_print(f'{str(sim_score)} / {str(weight)}')
            sim_score_w = sim_score/weight
            
            # print("caption ::: ", caption)
            # print("sim_score : ", sim_score)
            # print("sim_score_w : ", sim_score_w)
            # plt.imshow(proposed_segment)
            # plt.show()

            # log_print("weighted: ", sim_score_w)
            # Moda: just a try, two upper limits
            # if sim_score_w >= 2*sim_score: sim_score_w = 2*sim_score
            if sim_score_w >= 1: sim_score_w = 1
            # log_print("Final: ", sim_score_w)
            # plt.imshow(proposed_segment)
            # plt.title(str(idx))
            # plt.show()
            scores.append(sim_score_w)
        log_print(f'{self.current_stage} calculate_similarity_scores is done')
        ids = [x for _, x in sorted(zip(scores, ids), reverse=True)]
        scores.sort(reverse=True)
        log_print(f'{self.current_stage} score: {max(scores)}')
        return ids, scores

    def remove_stops(self, sent):
        words = [word for word in sent.split() if word.lower() not in self.stops]
        text = ' '.join(words)
        return text

    def calculate_similarity_score(self, text_1, text_2, remove_stops=True):
        if remove_stops:
            text_1 = self.remove_stops(text_1)
            text_2 = self.remove_stops(text_2)
        corpus = [text_1, text_2]

        if self.text_encoder=='count':
            text_encoder_model = CountVectorizer().fit_transform(corpus)
            corpus = text_encoder_model.toarray()
        elif self.text_encoder=='bert':
            text_encoder_model = SentenceTransformer('bert-base-nli-mean-tokens')
            corpus = text_encoder_model.encode(corpus)
        elif self.text_encoder=='roberta':
            text_encoder_model = SentenceTransformer('stsb-roberta-large')
            corpus = text_encoder_model.encode(corpus)
        
        vect_1 = corpus[0].reshape(1, -1)
        vect_2 = corpus[1].reshape(1, -1)

        return cosine_similarity(vect_1, vect_2)
    
    def select_highest(self, sorted_ids, scores):
        max_score = max(scores)
        selected_ids = list()
        for idx in range(len(sorted_ids)):
            if scores[idx] != max_score: break # the highest score only
            selected_ids.append(sorted_ids[idx])
        # self.stageI_selected_ids = selected_ids
        self.stages_results[self.current_stage]['selected_ids'] = selected_ids
        # self.stageI_max_score = max_score
        self.stages_results[self.current_stage]['max_score'] = max_score

    def merge_proposals(self):
        selected_ids = self.stages_results[self.current_stage]['selected_ids']
        selected_proposals = [self.current_stage_proposals[i] for i in selected_ids]
        proposed_mask = selected_proposals[0][4].copy() # [0] the first item in the list, [4] the binary mask
        for proposal in selected_proposals:
            proposed_mask = np.logical_or(proposed_mask, proposal[4])
        self.stages_results[self.current_stage]['proposed_mask'] = proposed_mask
        proposal = self.blurred.copy()
        proposal[proposed_mask] = self.image[proposed_mask]
        self.stages_results[self.current_stage]['proposal'] = proposal
        
    def stageII(self):
        self.current_stage = 'stage 2'
        self.current_stage_proposals = None
        proposed_mask = self.stages_results['stage 1']['proposed_mask']
        self.RandomSegModel.create_proposals(self.image, proposed_mask)
        self.current_stage_proposals = self.RandomSegModel.proposals
        self.nominate_proposal()
    
    def answer(self, save_dir='', image_id='', query_id=''):
        log_print(f"Caption: {self.main_caption}")
        log_print(f"Query: {self.query}")
        selected_stage = self.current_stage
        # selected_stage = 'stage 1' if self.stages_results['stage 1']['max_score'] > self.stages_results['stage 2']['max_score'] else 'stage 2'
        log_print(f"Answer similarity score: {self.stages_results[selected_stage]['max_score']}")

        masked_image = np.zeros_like(self.image)
        masked_image[:,:] = (255, 0, 0)
        masked_image[self.stages_results[selected_stage]['proposed_mask']!=True] = self.image[self.stages_results[selected_stage]['proposed_mask']!=True]

        if save_dir and image_id:
            save_path = str(os.path.join(save_dir, f'{image_id}_{query_id}_blurred.jpg'))
            plt.imsave(save_path, self.stages_results[selected_stage]['proposal'])
            save_path = str(os.path.join(save_dir, f'{image_id}_{query_id}_masked.jpg'))
            plt.imsave(save_path, masked_image)

            results_file = save_dir + f'results_{self.test_time}.txt'
            with open(results_file, 'a') as file:
                result_line = f"\n{image_id}, {self.main_caption}, {query_id}, {self.query}, "
                result_line += f"{'1' if self.is_good_answer() else '0'}, "
                result_line += f"{self.stages_results[selected_stage]['max_score']}, "
                result_line += f"{self.image.shape[0]*self.image.shape[1]}, "
                result_line += f"{int(np.count_nonzero(self.stages_results[selected_stage]['proposed_mask'])/3)}"
                file.write(result_line)

            stage = 'stage 1'
            results_file = save_dir + f'{image_id}_{stage}.bin'
            self.dump_results(results_file, stage)

            stage = 'stage 2'
            results_file = save_dir + f'{image_id}_{query_id}_{stage}.bin'
            self.dump_results(results_file, stage)

            # segments_dict = dict()
            # segments_dict['stage 1 segments'] = self.stages_results['stage 1']['segments']
            # segments_dict['stage 2 segments'] = self.stages_results['stage 2']['segments']
            # segments_dict['stage 1 proposal'] = self.stages_results['stage 1']['proposal']
            # segments_dict['stage 2 proposal'] = self.stages_results['stage 2']['proposal']
            # segments_dict['stage 1 proposed_mask'] = self.stages_results['stage 1']['proposed_mask']
            # segments_dict['stage 2 proposed_mask'] = self.stages_results['stage 2']['proposed_mask']
            # pkl.dump(segments_dict, open(save_dir + f'{image_id}_{query_id}.bin','wb'))
            # pkl.load(open(save_dir +  f'{image_id}_{self.test_time}.bin','rb'))   

        else:
            plt.imshow(self.image)
            plt.title('Original image')
            plt.show()
            
            plt.imshow(masked_image)
            plt.title('masked answer')
            plt.show()
            
            plt.imshow(self.stages_results[selected_stage]['proposal'])
            plt.title('Answer')
            plt.show()

    def dump_results(self, file_path, stage):
        segments_dict = dict()
        segments_dict[f'{stage} segments'] = self.stages_results[stage]['segments']
        segments_dict[f'{stage} proposal'] = self.stages_results[stage]['proposal']
        segments_dict[f'{stage} proposed_mask'] = self.stages_results[stage]['proposed_mask']
        segments_dict[f'{stage} captions'] = self.stages_results[stage]['captions']
        pkl.dump(segments_dict, open(file_path,'wb'))

        # segments_dict['stage 1 segments'] = self.stages_results['stage 1']['segments']
        # segments_dict['stage 2 segments'] = self.stages_results['stage 2']['segments']
        # segments_dict['stage 1 proposal'] = self.stages_results['stage 1']['proposal']
        # segments_dict['stage 2 proposal'] = self.stages_results['stage 2']['proposal']
        # segments_dict['stage 1 proposed_mask'] = self.stages_results['stage 1']['proposed_mask']
        # segments_dict['stage 2 proposed_mask'] = self.stages_results['stage 2']['proposed_mask']
        # pkl.dump(segments_dict, open(file_path,'wb'))

    def read_results(self, file_path, stage):
        segments_dict = pkl.load(open(file_path,'rb'))
        self.stages_results[stage]['segments'] = segments_dict[f'{stage} segments']
        self.stages_results[stage]['proposal'] = segments_dict[f'{stage} proposal']
        self.stages_results[stage]['proposed_mask'] = segments_dict[f'{stage} proposed_mask']
        self.stages_results[stage]['captions'] = segments_dict[f'{stage} captions']

        # self.stages_results['stage 1']['segments'] = segments_dict['stage 1 segments']
        # self.stages_results['stage 2']['segments'] = segments_dict['stage 2 segments']
        # self.stages_results['stage 1']['proposal'] = segments_dict['stage 1 proposal']
        # self.stages_results['stage 2']['proposal'] = segments_dict['stage 2 proposal']
        # self.stages_results['stage 1']['proposed_mask'] = segments_dict['stage 1 proposed_mask']
        # self.stages_results['stage 2']['proposed_mask'] = segments_dict['stage 2 proposed_mask']

    def test_images(self, save_dir, images_dir):
        if not os.path.isdir(images_dir):
            log_print('Invalid test directory!')
            return False

        self.test_time = strftime("%m_%d_%H_%M_%S", gmtime())
        results_file = save_dir + f'results_{self.test_time}.txt'
        if not os.path.isfile(results_file):
            with open(results_file, 'w') as file:
                file.write('image_id, caption, query_id, query, good_query, sim_score, image_size, answer_size')

        images_names = next(os.walk(images_dir), (None, None, []))[2]
        # partial test
        images_names = [int(name[:name.index('.')]) for name in images_names]
        images_names.sort()
        images_names = [str(name)+'.jpg' for name in images_names]
        print("images_names ::::: \n", images_names)
        start_idx = 0
        # last_idx = 22
        # images_names = images_names[start_idx:last_idx]
        images_names = images_names[start_idx:]
        # images_names = [images_names[start_idx]]
        #############
        for name in tqdm(images_names):
            segments_dict = dict()
            self.reset_parameters()
            ext_idx = name.index('.')
            image_id = name[:ext_idx]
            image_results = self.test_image(image_path=images_dir+name, save_dir=save_dir, image_id=image_id)     

    def reset_parameters(self):
        self.image_path = None
        self.main_caption = None
        self.query = None
        self.image = None
        self.blurred = None
        self.current_stage = None
        self.current_stage_proposals = None
        self.stages_results = self.initialize_stages_results()

    def test_image(self, image_path, save_dir, image_id):
        log_print(f'Testing image {image_id}')
        results_dict = dict()
        self.validate_image_path(image_path)
        if self.image_path: self.caption_image(self.image, main=True)
        queries = self.generate_test_quries()
        log_print(f"Queries: {queries}")

        stage = 'stage 1'
        results_file = save_dir + f'{image_id}_{stage}.bin'
        if os.path.isfile(results_file): self.read_results(results_file, stage)

        for query_id, query in enumerate(queries):
            stage = 'stage 2'
            results_file = save_dir + f'{image_id}_{query_id}_{stage}.bin'
            if os.path.isfile(results_file): self.read_results(results_file, stage)

            self.explain(image_path=self.image_path, mode='test', test_query=query, save_dir=save_dir, image_id=image_id, query_id=query_id)
            results_dict[query] = 1 if self.is_good_answer() else 0
        return results_dict

    def generate_test_quries(self):
        queries = self.chunker.get_chunks(self.main_caption)
        queries = list(set(queries))
        return queries

    def is_good_answer(self):
        test_images = self.generate_test_images()
        for image in test_images:
            caption = self.caption_image(image)
            if self.query in caption.lower(): return True
        return False

    def generate_test_images(self):
        mask = self.stages_results[self.current_stage]['proposed_mask']
        shape = self.image.shape
        black = np.full(shape, 0)
        grey = np.random.normal(0.5, 0.25, size=shape)
        white = np.full(shape, 1) 

        black[mask] = self.image[mask]
        grey[mask] = self.image[mask]
        white[mask] = self.image[mask]

        return [black, grey, white]


if __name__ == "__main__":
    # def calculate_similarity_score(text_1, text_2, remove_stops=True):
    #     # if remove_stops:
    #     #     text_1 = self.removeStops(text_1)
    #     #     text_2 = self.removeStops(text_2)
    #     corpus = [text_1, text_2]
    #     vectorizer = CountVectorizer().fit_transform(corpus)
    #     vectors = vectorizer.toarray()

    #     vect_1 = vectors[0].reshape(1, -1)
    #     vect_2 = vectors[1].reshape(1, -1)

    #     return cosine_similarity(vect_1, vect_2)
    
    # log_print(calculate_similarity_score('man in a blue shirt is standing on a wooden bench', 'wooden bench'))

    im = np.ones(shape=(400, 600, 3))
    log_print(im.shape)
    log_print(im[:,:,0].shape)
    im[:,:] = (255, 0, 0)
    plt.imshow(im)
    plt.show()