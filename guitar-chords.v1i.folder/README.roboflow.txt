
guitar-chords - v1 2024-07-23 6:44pm
==============================

This dataset was exported via roboflow.com on July 23, 2024 at 6:52 PM GMT

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

The dataset includes 13597 images.
Guitar-chords are annotated in folder format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x360 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees
* Random shear of between -12° to +12° horizontally and -12° to +12° vertically
* Random exposure adjustment of between -10 and +10 percent
* Random Gaussian blur of between 0 and 1.4 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels


