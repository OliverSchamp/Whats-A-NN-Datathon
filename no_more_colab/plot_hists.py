from matplotlib import pyplot as plt
import numpy as np
from transformers import ViTForImageClassification

barWidth = 0.4
fig = plt.subplots(figsize = (12, 8))

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



# name = [vit.config.id2label[0], vit.config.id2label[1], vit.config.id2label[2], vit.config.id2label[3], vit.config.id2label[4], vit.config.id2label[5], vit.config.id2label[6], vit.config.id2label[7], vit.config.id2label[8]]
# price = hist1
#
# # Figure Size
# fig, ax = plt.subplots(figsize=(16, 9))
#
# # Horizontal Bar Plot
# ax.barh(name, price)
#
# # Remove axes splines
# for s in ['top', 'bottom', 'left', 'right']:
#     ax.spines[s].set_visible(False)
#
# # Remove x, y Ticks
# ax.xaxis.set_ticks_position('none')
# ax.yaxis.set_ticks_position('none')
#
# # Add padding between axes and labels
# ax.xaxis.set_tick_params(pad=5)
# ax.yaxis.set_tick_params(pad=10)
#
# # Add x, y gridlines
# ax.grid(b=True, color='grey',
#         linestyle='-.', linewidth=0.5,
#         alpha=0.2)
#
# # Show top values
# ax.invert_yaxis()
#
# # Add annotation to bars
# for i in ax.patches:
#     plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
#              str(round((i.get_width()), 2)),
#              fontsize=10, fontweight='bold',
#              color='grey')
#
# # Add Plot Title
# ax.set_title('Sports car and their price in crore',
#              loc='left', )
#
# # Add Text watermark
# fig.text(0.9, 0.15, 'Jeeteshgavande30', fontsize=12,
#          color='grey', ha='right', va='bottom',
#          alpha=0.7)
#
# # Show Plot
# plt.show()