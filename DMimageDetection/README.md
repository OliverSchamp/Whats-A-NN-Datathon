On the detection of synthetic images generated by diffusion models
========================
This is the official repository of the paper:
[On the detection of synthetic images generated by diffusion models](https://arxiv.org/abs/2211.00680) 
Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, Luisa Verdoliva

The synthetic images used as test can be downloaded from the following [link](https://drive.google.com/file/d/1grvgKiIq0ny8ImQzSUXPk3nd-AMEDjNb/view?usp=share_link) alongside a csv file stating the processing applied in the paper on each image. The real images can be downloaded from the following freely available datasets : [IMAGENET](https://image-net.org/index.php), [UCID](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/5307/0000/UCID-an-uncompressed-color-image-database/10.1117/12.525375.short),[COCO - Common Objects in Context](https://cocodataset.org/#home).
The real images should then be placed in a folder with the same name that has been recorded in the csv file
The directory containing the test set should have the following structure:
```
Testset directory
|--biggan_256
|--biggan_512
.
.
.
|--real_coco_valid
|--real_imagenet_valid
|--real_ucid
.
.
.
|--taming-transformers_segm2image_valid
```
In this repository it is also provided a python script to apply on each image the processing outlined by the csv file.

There are also provided the code to test the networks on the provided images.  
The networks weights can be downloaded from the following [link](https://drive.google.com/file/d/1sAoAuOGCWS4dAMBhDkRHgBf4SgBgvkVf/view?usp=share_link) 

In order to launch the scripts, create a conda enviroment using the enviroment.yml provided.

The commands can be launched as follows:

To generate the images modified according to the details contained in the csv file, launch the script as follows:

```
python csv_operations.py --data_dir /path/to/testset/dir --out_dir /path/to/output/dir --csv_file /path/to/csv/file
```
In order to calculate the outputs of each model launch the script as shown below
```
python main.py --data_dir /path/to/testset/dir --out_dir /path/to/output/dir --csv_file /path/to/csv/file

```
Finally to generate the csv files containing the accuracies and aucs calculated per detection method and per generator architecture launche the last script as described.
```
python metrics_evaluations.py --data_dir /path/to/testset/dir --out_dir /path/to/output/dir
```


The annotations used to generate images with text to images models belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License (https://cocodataset.org/#termsofuse).

