import pandas as pd
import sys
import io
import matplotlib.pyplot as plt
import requests
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

# artists = pd.read_parquet('https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artist.parquet.gzip')
# artworks = pd.read_parquet('https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artwork.parquet.gzip')
#
# subset_artworks = artworks[['name', 'artist', 'image_url']]
#
# for i in range(np.max(artworks['artist'])):
#   artist_name = str(artists.query('id == ' + str(i))['name'].iloc[0])
#   subset_artworks['artist'] = subset_artworks['artist'].replace(i, artist_name)
#
# subset_artworks['caption'] = subset_artworks['name'].astype(str) + " by " + subset_artworks["artist"].astype(str)
# subset_artworks = subset_artworks.drop(['name', 'artist'], axis=1)
# data = subset_artworks

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

model_path = "oschamp/pytorch_artwork_lora_finetuned"
#model_path = "AngelBubulubu/arttune"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

def finetuned_generate(prompt, seed):
  generator = torch.Generator("cuda").manual_seed(seed)
  return pipe(prompt, generator=generator).images[0]

# images = [2493, 539, 83, 3135, 243, 1672, 2925, 2225, 357, 419, 1844, 1676, 642, 1696, 1067, 1634]
#
# for j in range(3):
#   for i in images:
#     prompt = data['caption'].iloc[i]
#     print(prompt)
#     image = pipe(prompt).images[0]
#     image.save("images/" + str(i)+"_"+str(j)+".png")

#replace images with the artworks we fine-tuned on using a copy of the metadata file
dff = pd.read_csv('to_generate.csv')

for i in range(dff.shape[0]):
  seed = np.random.randint(0, 10000)
  prompt = dff['prompts'].iloc[i]
  print(prompt)
  image = finetuned_generate(prompt, seed)

  # solving the NSFW issue
  while np.array(image).mean() == 0:
    seed = seed + 1
    image = finetuned_generate(prompt, seed)

  image.save("generated_fine-tuned/" + str(dff['source_artwork'].iloc[i]) + ".png", "PNG")