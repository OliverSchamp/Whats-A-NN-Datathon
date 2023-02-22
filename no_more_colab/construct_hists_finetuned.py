#import the ViT model
#import original stable diffusion
#import my fine-tuned version of stable diffusion

#have a constant artwork title e.g. starry night
#iterate through the list of artists, feeding both generation models the artwork title plus artist name prompt
#take the generated artworks from the models and use the ViT to classify them as an art category
#keep a cumulative list of how many times each category has come up
#construct a bar chart, histogram with number of artists vs type of artwork

from transformers import ViTForImageClassification
from PIL import Image
import numpy as np
from transformers import ViTFeatureExtractor
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

vit = ViTForImageClassification.from_pretrained("oschamp/vit-artworkclassifier")
vit.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit.to(device)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

def vit_classify(image):
    encoding = feature_extractor(images=image, return_tensors="pt")
    encoding.keys()

    pixel_values = encoding['pixel_values'].to(device)

    outputs = vit(pixel_values)
    logits = outputs.logits

    prediction = logits.argmax(-1)
    return prediction.item() #vit.config.id2label[prediction.item()]
    # print("Predicted class:", vit.config.id2label[prediction.item()])

# ---------------------------------

# model_id = "CompVis/stable-diffusion-v1-4"
# device = "cuda"
#
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to(device)
#
# def original_generate(prompt):
#     return pipe(prompt).images[0]

# ----------------------------------

model_path = "oschamp/pytorch_artwork_lora_finetuned"
# model_path = "AngelBubulubu/arttune"
pipe_ft = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe_ft.unet.load_attn_procs(model_path)
pipe_ft.to("cuda")

def finetuned_generate(prompt, seed):
    generator = torch.Generator("cuda").manual_seed(seed)
    return pipe_ft(prompt, generator=generator).images[0]
    #faster generation
    # return pipe_ft(prompt, num_inference_steps=15, generator=generator).images[0]

# ---------------------------------------------
#Histogram code
import pandas as pd

art_name = "How to Bake a Cake"

artists = pd.read_parquet('https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artist.parquet.gzip')

#make a list of say, 50 artists, but they have to comprise out of 16 famous ones

#grid of 9 artists
#famous_artists = ['Vincent Van Gogh', 'Claude Oscar Monet', 'Rembrandt Van Rijn', 'Michelangelo Buonarroti', 'Salvador Dali', 'Leonardo Da Vinci', 'Henri Matisse', 'Pablo Picasso', 'Jackson Pollock']

#random names
famous_artists = ['Bob', 'Dave', 'Sam', 'Charles', 'Oliver', 'Sean', 'Tom', 'Joe', 'Guy']

artist_list = pd.unique(artists['name'])

#sample an artist from artist list, if not already in artist-list, add them to the list
artist_list_withfamous = famous_artists
while len(artist_list_withfamous) < 50:
    randomartist = np.random.choice(artist_list)
    if randomartist not in artist_list_withfamous:
        artist_list_withfamous.append(randomartist)

prompts = []
for artist in artist_list_withfamous:
    p = art_name + " by " + artist
    prompts.append(p)

prompts = prompts[:9]
# hist1 = np.zeros(9)
hist2 = np.zeros(9)

first_9_famousartists = []
first_9_prompts = []
first_9_classes = []
seed = 42
for i, prompt in enumerate(prompts):
    print("Iteration: ", i+1)

    # image1 = original_generate(prompt)
    image2 = finetuned_generate(prompt, seed)

    #solving the NSFW issue
    while np.array(image2).mean() == 0:
        seed = seed + 1
        image2 = finetuned_generate(prompt, seed)


    # class1 = vit_classify(image1)
    class2 = vit_classify(image2)

    # hist1[class1] += 1
    hist2[class2] += 1

    if i < 9:
        first_9_famousartists.append(image2)
        first_9_prompts.append(prompt)
        first_9_classes.append(vit.config.id2label[class2])

# print(hist1)
print(hist2)
print(first_9_classes)

#----------------------------------
def plotImagesAndLabels(image_list, prompts, classes):
    fig, m_axs = plt.subplots(3, 3, figsize = (16, 16))
    for i, c_ax in enumerate(m_axs.flatten()):
        c_ax.imshow(image_list[i], vmin = -1.5, vmax = 1.5)
        c_ax.set_title(prompts[i] + "\n" + "Class: " + classes[i])
        c_ax.set_xlabel("Class: " + classes[i])
        c_ax.axis('off')
    fig.show()
    plt.show()

plotImagesAndLabels(first_9_famousartists, first_9_prompts, first_9_classes)

#write to a csv
df = pd.read_csv('histograms.csv')
#add an entry to the csv
df.loc[len(df.index)] = [art_name, 'finetuned', hist2]

print("SYOP")
#save the csv
df.to_csv('histograms.csv', index=False)