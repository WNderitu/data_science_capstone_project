# Plasmodium Vivax (malaria) parasite detector and counter

## Introduction

## Problem Statement

## Project Objectives

1. To detect Plasmodium Vivax (malaria parasite) in human cells
2. To count the number of detected plasmodium vivax parasites per image

## Methods Used

## Technologies

## Project Description

The project will entail conduct of the following:

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

## Image Data Cleaning and Image preprocessing
 - Data cleaning involved checking for corrupted images, duplicate images, visually similar images, and class imbalance in the various categories.
 - Image preprocessing will be done for data that will be used in the non-YOLO model.
 - Image preprocessing is inbuilt in the YOLO model

## Machine learning Modelling
You Only Look Once (YOLO) model and a Region Based Convolutional Neural Network will be used for object detection and counting. 

### Model Architecture

## Recommendations/Next Steps

## Folder structure

The folder contains the following files:

## Acknowledgements & Attributions

We used image set BBBC041v1, available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods, 2012)
