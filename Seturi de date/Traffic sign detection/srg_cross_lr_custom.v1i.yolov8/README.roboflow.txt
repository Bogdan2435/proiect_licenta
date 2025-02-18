
srg_cross_lr_custom - v1 2024-06-13 11:45am
==============================

This dataset was exported via roboflow.com on June 28, 2024 at 10:06 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2741 images.
Si are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 220x220 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random brigthness adjustment of between -11 and +11 percent
* Random exposure adjustment of between -9 and +9 percent
* Random Gaussian blur of between 0 and 0.9 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels


