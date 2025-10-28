# Plasmodium Vivax (malaria) parasite detector and counter using YOLOv8 Model

## Introduction

## Problem Statement

## Project Objectives

1. To detect uninfected red blood cells and leukocytes in human blood cells in a microscopic image from a blood smear
2. To count the number of uninfected red blood cells and leukocytes in human blood cells in a microscpic image from a blood smear
3. To detect the growth stage of a Plasmodium Vivax (malaria parasite) in human blood cells in a microscopic image from a blood smear
4. To count the number of detected growth stages of the plasmodium vivax parasites per microscopic image

## Methods Used

## Technologies

## Project Description

The project will entail conduct of the following:
1. Loading and understanding the structure of the image dataset
2. Cleaning and ppreparation of the image dataset for object detection and counting using a You Only Look Once (YOLO) v8 model
4. Training of the model
5. Running of the predictions and counting (model testing)
6. Evaluation of the model

## Image Dataset

Images obtained from Broad Bioimage Benchmark Collection webiste <https://bbbc.broadinstitute.org/BBBC041/>

The dataset consists of:
- image folder
- training json file
- test json file

The dataset consists of 1,328 microscopic images of blood smears. 

The various categories to be detected are 7.
- Red Blood Cell (uninfected)
- Trophozoite
- Schizont
- Difficult
- Ring
- Leukocyte (uninfected)
- Gametocyte

The Red Blood Cell and Leukocyte categories are blood cells that are not infected with the malaria parasite.

The Trophozoite, Schizont, Ring and Gametocyte are different growth stages of the malaria parasite

The difficult category is a growth stage that couldn't be grouped into either of the 4 growth categories.

Sample images from the dataset

<img width="1293" height="1990" alt="image" src="https://github.com/user-attachments/assets/9dc47e0e-80c5-4288-9cdb-bfd8a2b09387" />

<img width="1293" height="1990" alt="image" src="https://github.com/user-attachments/assets/6c98bade-f368-4803-a186-0deb5492f814" />

Upon downloading the dataset, a folder called labels was created and the two json files moved to the folder.

The training json file was renamed to train.json

## Image Data Cleaning and Preparation
 - Data cleaning will involve checking for corrupted images, duplicate images, visually similar images, and class imbalance in the various categories.
 - Data preparation will involve the following:
   - Splitting images in the image sub folder into 2 folders: test & train images as the YOLOv8 model requires a specific folder structure i.e.
   
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

   - Creation of a data.yaml File in the dataset's root directory, that describes the dataset, classes, and other necessary information.
     
## Machine learning Modelling
You Only Look Once (YOLO) v8 model will be used for object detection and counting. 

### Model Architecture


## Recommendations/Next Steps


## Folder structure

The folder contains the following files:

## Acknowledgements & Attributions

We used image set BBBC041v1, available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods, 2012)
