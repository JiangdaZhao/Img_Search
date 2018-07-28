from pathlib import Path
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import string
import json
import pickle
import re, string
#from IPython.core.display import Image
#from PIL import Image

from IPython.core.display import display

import io
import json
import requests
import numpy as np
from pathlib import Path
from PIL import Image as new_IMG
import mygrad as mg
import torch
from torchvision.models import resnet18
from torchvision import transforms
#from camera import take_picture
from io import StringIO
import matplotlib.pyplot as plt

def get_image(img_url):
    ''' Fetch an image from CoCo.
    
    Parameters
    ----------
    img_url : str
        The url of the image to fetch, in the format:
            http://images.cocodataset.org/--dataset--/---unique_image_id--.jpg
        
        `dataset` is the specific coco dataset you wish to use, such as train2014 or val2017
        `unique_image_id` is an alpha-numeric sequence specific to the image you want to fetch
        
    Returns
    -------
    PIL.Image
        The image.
    '''
    response = requests.get(img_url)
    return new_IMG.open(io.BytesIO(response.content))

def search_phrase(query, annotations, image_id2id, glove, vectors_512, inds_img, weight, bias, loaded_file):
    query = query.lower()

    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

    def strip_punc(corpus):
        return punc_regex.sub('', corpus)

    query = strip_punc(query)
    query_components = query.split()
    N = len(annotations)
    queries_in_dict = []
    for word in query_components:
        if word in glove:
            nt = 1
            for caption in annotations:
                nt += word in set(caption.split())
            idf = np.log10(N/nt)
            queries_in_dict.append(glove[word] * idf)

    final_embedding = np.mean(np.vstack(queries_in_dict), axis = 0)
    # Find the cosine distance between the 50D vectors and the embeddings
    #s = np.sum(vectors_50, axis = 1)
    print(type(vectors_512))
    print(type(vectors_512[0]))
    print(type(weight))
    vectors_50 = mg.matmul(vectors_512, weight) + bias
    #print(vectors_50[:k])
    #print(vectors_50[21697])
    vectors_50 = vectors_50/np.linalg.norm(vectors_50)
    #s = np.sum(final_embedding)
    final_embedding = final_embedding/np.linalg.norm(final_embedding)

    cos = np.dot(vectors_50, final_embedding)
    k = 4
    max_vals = np.argsort(cos)[-k:]

    fig, ax = plt.subplots(2,2)
    for ind, ima in enumerate(inds_img[max_vals]):
        
        url = loaded_file["images"][image_id2id[ima]]['coco_url']
        img = get_image(url)
        ax[ind // 2, ind % 2].imshow(img)

'''def search_image(annotations, image_id2id, glove, vectors_512, inds_img, weight, bias, loaded_file):

    class IdentityModule(torch.nn.Module):
        def forward(self, inputs):
            return inputs

    model = resnet18(pretrained=True)
    model.fc = IdentityModule()
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = new_IMG.fromarray(take_picture())
    processed_img = preprocess(img)[np.newaxis]
    feats = model(processed_img)
    vectors_512_copy = vectors_512.copy()
    vectors_512_copy /= np.sqrt(np.sum(vectors_512_copy**2))
    feat = feats.numpy()
    feat /= np.sqrt(np.sum(feat**2))
    cos = np.dot(vectors_512_copy, feat)
    k = 4
    max_vals = np.argsort(cos)[-k:]
    fig, ax = plt.subplots(2,2)
    for ind, ima in enumerate(inds_img[max_vals]):
        
        url = loaded_file["images"][image_id2id[ima]]['coco_url']
        img = get_image(url)
        ax[ind // 2, ind % 2].imshow(img)'''