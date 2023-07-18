import numpy as np
import torch
from pkg_resources import packaging

print("Torch version:", torch.__version__)

import clip
print(clip.available_models())

model, preprocess = clip.load("ViT-B/32")
# # model.cuda().eval()
# input_resolution = model.visual.input_resolution
# context_length = model.context_length
# vocab_size = model.vocab_size

# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# print("Input resolution:", input_resolution)
# print("Context length:", context_length)
# print("Vocab size:", vocab_size)

# print(preprocess)

# print(clip.tokenize("Hello World!"))


import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch

import cv2

# # images in skimage to use and their textual descriptions
# descriptions = {
#     "page": "a page of text about segmentation",
#     "chelsea": "a facial photo of a tabby cat",
#     "astronaut": "a portrait of an astronaut with the American flag",
#     "rocket": "a rocket standing on a launchpad",
#     "motorcycle_right": "a red motorcycle standing in a garage",
#     "camera": "a person looking at a camera on a tripod",
#     "horse": "a black-and-white silhouette of a horse", 
#     "coffee": "a cup of coffee on a saucer"
# }

# original_images = []
# images = []
# texts = []
# plt.figure(figsize=(16, 5))

# for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
#     name = os.path.splitext(filename)[0]
#     if name not in descriptions:
#         continue

#     image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
  
#     plt.subplot(2, 4, len(images) + 1)
#     plt.imshow(image)
#     plt.title(f"{filename}\n{descriptions[name]}")
#     plt.xticks([])
#     plt.yticks([])

#     original_images.append(image)
#     images.append(preprocess(image))
#     texts.append(descriptions[name])

# plt.tight_layout()
# plt.show()


# image_input = torch.tensor(np.stack(images))
# text_tokens = clip.tokenize(["This is " + desc for desc in texts])

# with torch.no_grad():
#     image_features = model.encode_image(image_input).float()
#     text_features = model.encode_text(text_tokens).float()

# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

# count = len(descriptions)

# plt.figure(figsize=(20, 14))
# plt.imshow(similarity, vmin=0.1, vmax=0.3)
# # plt.colorbar()
# plt.yticks(range(count), texts, fontsize=18)
# plt.xticks([])
# for i, image in enumerate(original_images):
#     plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
# for x in range(similarity.shape[1]):
#     for y in range(similarity.shape[0]):
#         plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

# for side in ["left", "top", "right", "bottom"]:
#   plt.gca().spines[side].set_visible(False)

# plt.xlim([-0.5, count - 0.5])
# plt.ylim([count + 0.5, -2])

# plt.title("Cosine similarity between text and image features", size=20)

# plt.show()

# queries = ["two dogs", "two dog", "running", "snow", "woman", "in the snow two dogs are running", "in snow the running are two dogs"]
# caption = ["two dogs are running in the snow"]

# queries = clip.tokenize(queries)
# caption = clip.tokenize(caption)

# with torch.no_grad():
#     queries_features = model.encode_text(queries).float()
#     caption_features = model.encode_text(caption).float()

# queries_features /= queries_features.norm(dim=-1, keepdim=True)
# caption_features /= caption_features.norm(dim=-1, keepdim=True)

# similarity = queries_features.cpu().numpy() @ caption_features.cpu().numpy().T

# print(similarity)

import pickle as pkl
segments_dict = pkl.load(open('data\\test\\flickr8k\save_dir\\0_stage 1.bin','rb'))
mask = segments_dict['stage 1 segments'][-1][-1]
print(mask.shape)

blur_ksize = 100

original = Image.open(os.path.join('./data/clip_samples', '0.jpg')).convert("RGB")
black = Image.open(os.path.join('./data/clip_samples', '0_black.jpg')).convert("RGB")
white = Image.open(os.path.join('./data/clip_samples', '0_white.jpg')).convert("RGB")

original = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join('./data/clip_samples', '0.jpg')), cv2.COLOR_BGR2RGB))
blurred = Image.fromarray(cv2.blur(cv2.cvtColor(cv2.imread(os.path.join('./data/clip_samples', '0.jpg')), cv2.COLOR_BGR2RGB), (blur_ksize, blur_ksize)))
black = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join('./data/clip_samples', '0_black.jpg')), cv2.COLOR_BGR2RGB))
white = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join('./data/clip_samples', '0_white.jpg')), cv2.COLOR_BGR2RGB))
blurred_1 = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join('./data/clip_samples', '0_blurred_1.jpg')), cv2.COLOR_BGR2RGB))
blurred_2 = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join('./data/clip_samples', '0_blurred_2.jpg')), cv2.COLOR_BGR2RGB))
blurred_3 = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join('./data/clip_samples', '0_blurred_3.jpg')), cv2.COLOR_BGR2RGB))

blurred_4 = cv2.blur(cv2.cvtColor(cv2.imread(os.path.join('./data/clip_samples', '0.jpg')), cv2.COLOR_BGR2RGB), (blur_ksize, blur_ksize))
blurred_4[mask==True] = cv2.cvtColor(cv2.imread(os.path.join('./data/clip_samples', '0.jpg')), cv2.COLOR_BGR2RGB)[mask==True]
blurred_4 = Image.fromarray(blurred_4)



original_images = [original, black, white, blurred, blurred_1, blurred_2, blurred_3, blurred_4]

images = []
for image in original_images:
    images.append(preprocess(image))

queries = ["two dogs are running in the snow", "two dogs", "two dog", \
            "running", "snow", "woman", "in the snow two dogs are running", \
            "in snow the running are two dogs", "white", "black"]

# queries = ["two dogs are running in the snow"]

image_input = torch.tensor(np.stack(images))
# image_input = torch.tensor(np.expand_dims(images[0], axis=0))
text_tokens = clip.tokenize(queries)

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()


print("images : ", type(images[0]))
print("images : ", images[0])
print("images : ", images[0])
print("images shape : ", images[0].shape)
print("image_input shape : ", image_input.shape)
print("image_features shape : ", image_features.shape)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

print("images type : ", type(images[0]))
print("similarity shape : ", similarity.shape)
print("similarity : ", similarity)
print("similarity[0] : ", similarity[0][0])

count = len(queries)

plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)
# plt.colorbar()
plt.yticks(range(count), queries, fontsize=18)
plt.xticks([])
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
  plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity between text and image features", size=20)

plt.show()