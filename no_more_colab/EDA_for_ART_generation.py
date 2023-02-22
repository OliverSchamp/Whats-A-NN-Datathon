import pandas as pd
import sys
import io
import matplotlib.pyplot as plt
import requests
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

artists = pd.read_parquet('https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artist.parquet.gzip')
artworks = pd.read_parquet('https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artwork.parquet.gzip')

subset_artworks = artworks[['name', 'artist', 'image_url']]

for i in range(np.max(artworks['artist'])):
  artist_name = str(artists.query('id == ' + str(i))['name'].iloc[0])
  subset_artworks['artist'] = subset_artworks['artist'].replace(i, artist_name)

subset_artworks['caption'] = subset_artworks['name'].astype(str) + " by " + subset_artworks["artist"].astype(str)
subset_artworks = subset_artworks.drop(['name', 'artist'], axis=1)
data = subset_artworks

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

model_path = "oschamp/pytorch_artwork_lora_finetuned"
#model_path = "AngelBubulubu/arttune"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

images = [2493, 539, 83, 3135, 243, 1672, 2925, 2225, 357, 419, 1844, 1676, 642, 1696, 1067, 1634]
for j in range(3):
  for i in images:
    prompt = data['caption'].iloc[i]
    print(prompt)
    image = pipe(prompt).images[0]
    image.save("images/" + str(i)+"_"+str(j)+".png")