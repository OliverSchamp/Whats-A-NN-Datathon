#create a dataset with 100 images of artwork (train) and 15 and 15 images of artwork (val and test) from each of the 10 artbench-10 art styles
#create the correct metadata.csv file to go along with this
#the filepaths should be folder/train/metadata.csv, folder/train/0001.png
import requests
from PIL import Image
import pandas as pd
import io
import os
import numpy as np
from transformers import ViTFeatureExtractor


#choose whether directed to the train or the test or the validation folder
filepath = r"C:\Users\OliverSchamp\Documents\Whats-A-NN-Datathon\no_more_colab\artbench10-vit\train"
train_size = 1000

df = pd.read_csv('ArtBench-10.csv')
print(df['is_public_domain'].iloc[0])
df = df.query("is_public_domain == True")
df = df[['name', 'url', 'label']]
for style in df['label'].unique():
    if style in ['ukiyo_e']: #not in dataset given to us I think
        continue
    df_style = df.query("label == '" + style + "'").iloc[:train_size+100, :]
    os.mkdir(filepath + "/" + style)
    i = 0
    j = 0
    print(style)
    while i < train_size:
        url = df_style['url'].iloc[j]
        response = requests.get(url)
        try:
            image = Image.open(io.BytesIO(response.content))
            i += 1
            j += 1
        except:
            j += 1
            continue
        image = image.resize((226, 226))
        if np.array(image).shape != (226, 226, 3):
            print(j)
            continue
        name = str(i) + ".png"
        image.save(filepath + "/" + style + "/" + name, "PNG")

# -------------------------------------------------------
filepath = r"C:\Users\OliverSchamp\Documents\Whats-A-NN-Datathon\no_more_colab\artbench10-vit\test"
test_size = 100

# df = pd.read_csv('ArtBench-10.csv')
print(df['is_public_domain'].iloc[0])
df = df.query("is_public_domain == True")
df = df[['name', 'url', 'label']]
for style in df['label'].unique():
    if style in ['ukiyo_e']: #not in dataset given to us I think
        continue
    df_style = df.query("label == '" + style + "'").iloc[train_size:, :]
    os.mkdir(filepath + "/" + style)
    i = 0
    j = 0
    print(style)
    while i < test_size:
        url = df_style['url'].iloc[j]
        response = requests.get(url)
        try:
            image = Image.open(io.BytesIO(response.content))
            i += 1
            j += 1
        except:
            j += 1
            continue
        image = image.resize((226, 226))
        if np.array(image).shape != (226, 226, 3):
            print(j)
            continue
        name = str(i) + ".png"
        image.save(filepath + "/" + style + "/" + name, "PNG")
