from matplotlib import pyplot as plt
import numpy as np
from transformers import ViTForImageClassification

barWidth = 0.4
fig = plt.subplots(figsize = (15, 8))

#starry night
# hist1 = [3, 0, 27, 3, 11, 1, 0, 0, 5]
# hist2 = [3, 0, 28, 0, 5, 0, 0, 1, 13]

#mona lisa
DDPM_original = [2, 0, 0, 0, 0, 0, 44, 0, 4]
DDPM_finetuned = [6, 0, 1, 0, 0, 0, 20, 0, 23]

br1 = np.arange(len(DDPM_original))
br2 = [x + barWidth for x in br1]

#plot
plt.bar(br1, DDPM_original, color = 'r', width=barWidth, edgecolor='grey', label='DDPM_original')
plt.bar(br2, DDPM_finetuned, color = 'b', width=barWidth, edgecolor='grey', label='DDPM_finetuned')

vit = ViTForImageClassification.from_pretrained("oschamp/vit-artworkclassifier")
vit.eval()

plt.xticks([r + barWidth/2 for r in range(len(DDPM_original))], [vit.config.id2label[0], vit.config.id2label[1], vit.config.id2label[2], vit.config.id2label[3], vit.config.id2label[4], vit.config.id2label[5], vit.config.id2label[6], vit.config.id2label[7], vit.config.id2label[8]])

plt.title('Classification of "Mona Lisa by [artist]" by Artbench-10-ViT')
plt.ylabel('Number of artists')
plt.xlabel('Art style (Artbench-10 classes)')
plt.legend()
plt.show()