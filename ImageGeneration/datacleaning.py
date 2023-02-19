import pandas as pd

import io
import matplotlib.pyplot as plt
import requests
from PIL import Image
import numpy as np

artists = pd.read_parquet('https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artist.parquet.gzip')
artworks = pd.read_parquet('https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artwork.parquet.gzip')

subset_artworks = artworks[['name', 'artist', 'image_url']]

for i in range(np.max(artworks['artist'])+1):
  artist_name = str(artists.query('id == ' + str(i))['name'].iloc[0])
  subset_artworks['artist'] = subset_artworks['artist'].replace(i, artist_name)

subset_artworks['caption'] = subset_artworks['name'].astype(str) + " by " + subset_artworks["artist"].astype(str)
subset_artworks = subset_artworks.drop(['name', 'artist'], axis=1)
data = subset_artworks

metadata = pd.DataFrame({'file_name': pd.Series(dtype='str'),
                         'additional_feature': pd.Series(dtype='str')})

for i, url in enumerate(data['image_url']):
  response = requests.get(url)
  image = Image.open(io.BytesIO(response.content))
  fname = str(i) + ".png"
  image.save("imagefolder/" + fname,"PNG")
  #add a row to the dataframe
  metadata.loc[i] = [fname, data['caption'][i]]

metadata.to_csv('metadata.csv', index=False)