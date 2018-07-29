from pathlib import Path
#from camera import take_picture
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import string
import json
import pickle
from Img_Search import Final_Code

#from IPython.core.display import Image
#from PIL import Image

from IPython.core.display import display

import io
import json
import requests
import numpy as np
from pathlib import Path
from PIL import Image as new_IMG

import torch
from torchvision.models import resnet18
from torchvision import transforms

from io import StringIO
import matplotlib.pyplot as plt



#annotations, image_id2id, glove, vectors_512, inds_img, weight, bias, loaded_file
file_path = Path.home()
path = file_path/"glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)
with open(file_path/"captions_train2014.json") as f:
    loaded_file = json.load(f)

with open(file_path/"resnet18_features_train.pkl", mode = 'rb') as f:
    resnet = pickle.load(f)

annotations = []
for annotation in loaded_file["annotations"]:
    annotations.append(annotation['caption'])
    
inds_img = []
vectors_512 = []
for i in resnet.keys():
    inds_img.append(i)
    vectors_512.append(resnet[i].numpy())
inds_img = np.array(inds_img)
vectors_512 = np.vstack(vectors_512)
image_id2id = {}

weight = np.load(file_path/"weight.npy")
bias = np.load(file_path/"bias.npy")

for i in range(len(loaded_file["images"])):
    image_id2id[loaded_file["images"][i]['id']] = i


def phrase_search(phrase):
    Final_Code.search_phrase(phrase, annotations, image_id2id, glove, vectors_512, inds_img, weight, bias, loaded_file)
    return

def image_search():
    Final_Code.search_image(annotations, image_id2id, glove, vectors_512, inds_img, weight, bias, loaded_file)
    return

