import pandas as pd

import io
import matplotlib.pyplot as plt
import requests
from PIL import Image
import numpy as np

artists = pd.read_parquet('https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artist.parquet.gzip')
artworks = pd.read_parquet('https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artwork.parquet.gzip')

subset_artworks = artworks[['id', 'name', 'artist', 'image_url']]

fakedetection_scores = pd.read_csv('fake_sorted.csv')

#csv already sorted from highest to lowest scores, so all we do is slice it
NUM_IMAGES = 500
fakedetection_top = fakedetection_scores.iloc[:NUM_IMAGES, :] #harddetect
# fakedetection_top = fakedetection_scores.iloc[-NUM_IMAGES:, :] #easydetect

def return_id_fromfile(file_path):
  slist = [*file_path]
  num_str = ""
  for char in slist:
    try:
      int(char)
      num_str = num_str + char
    except:
      continue

  return int(num_str)

filepaths = fakedetection_top['src'].iloc[:]

ids_to_download = []
for file_path in filepaths:
  id = return_id_fromfile(file_path)
  ids_to_download.append(id)

# mask = subset_artworks['id'].isin(ids_to_download)
# subset_artworks = subset_artworks.loc[~mask]

# subset_artworks['id'].mask(subset_artworks['id'].isin(ids_to_download).values)

subset_artworks = subset_artworks[subset_artworks.index.isin(ids_to_download)]

for i in range(np.max(artworks['artist'])+1):
  artist_name = str(artists.query('id == ' + str(i))['name'].iloc[0])
  subset_artworks['artist'] = subset_artworks['artist'].replace(i, artist_name)

subset_artworks['caption'] = subset_artworks['name'].astype(str) + " by " + subset_artworks["artist"].astype(str)
subset_artworks = subset_artworks.drop(['name', 'artist'], axis=1)
data = subset_artworks #[['image_url', 'caption']]

metadata = pd.DataFrame({'file_name': pd.Series(dtype='str'),
                         'text': pd.Series(dtype='str')})

# histograms = pd.DataFrame({'artwork': pd.Series(dtype='str'), 'model': pd.Series(dtype='str'),
#                          'histogram': pd.Series(dtype='int')})

folder = "harddetect-500/" #directory in which to save your image data

for i, url in zip(data['id'], data['image_url']):
  response = requests.get(url)
  image = Image.open(io.BytesIO(response.content))
  fname = str(i) + ".png"
  image.save(folder + fname,"PNG")
  #add a row to the dataframe
  metadata.loc[len(metadata.index)] = [fname, data['caption'][i]]

metadata.to_csv(folder + 'metadata.csv', index=False)