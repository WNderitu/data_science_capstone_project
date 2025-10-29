# Plasmodium Vivax (malaria) parasite detector and counter using YOLOv8 Model

## Project Overview

## Introduction

Malaria, a life-threatening disease caused by Plasmodium parasites transmitted by female Anopheles mosquitoes, is most commonly found in tropical and subtropical regions. In 2023, the African Region was home to 94% and 95% of malaria cases and deaths, respectively. Sub-Saharan Africa carries a disproportionately high share of the global malaria burden. Plasmodium vivax is one of the 5 parasites that cause malaria associated with disease relapses due to dormant liver stages, contributing significantly to morbidity.  

The Plasmodium vivax lifecycle alternates between humans and Anopheles mosquitoes. Infection begins when a mosquito injects sporozoites into a human, which then move to the liver to either develop or remain dormant. Merozoites released from the liver infect red blood cells, progressing through ring, trophozoite, and schizont stages, causing malaria symptoms. Some parasites develop into gametocytes, enabling further transmission via mosquitoes.

Diagnosis of Plasmodium vivax malaria relies on examining stained blood smears via microsocopy and rapid diagnostic tests, with accurate detection being key to effective treatment and relapse prevention. Advanced methods like deep learning models, such as  Regional Based CNN, YOLOv8, and others would offer improved reliability and efficiency in identifying infections.

## Problem Statement

The detection of Plasmodium vivax malaria remains challenging, primarily due to the morphological similarities between parasite stages and other blood components observed in microscopic images. Conventional microscopy is a labor-intensive process that is susceptible to human error and demands specialized expertise, which may be limited in under-resourced settings. Consequently, there is an urgent need for automated, efficient, and dependable solutions capable of detecting and quantifying Plasmodium vivax parasites and their developmental stages within blood smears. Such advancements would enhance diagnostic accuracy, promote effective treatment, and strengthen malaria control initiatives.

## Project Objectives

1.	To develop a computer vision model for object detection and counting
2.	To detect uninfected red blood cells and leukocytes in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
3.	To count the number of uninfected red blood cells and leukocytes in human blood cells in a microscpic image from a blood smear using the developed YOLOv8 model
4.	To detect the growth stage of a Plasmodium Vivax (malaria parasite) in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
5.	 To count the number of detected growth stages of the plasmodium vivax parasites per microscopic image using the developed YOLOv8 model

## Methodology
This project employs YOLOv8 for multiclass object detection, leveraging its advanced capabilities to accurately identify and localize multiple object types within images or video frames. The methodology comprises several key steps to ensure robust and reproducible results.
•	**Obtaining dataset**: Data will be obtained from Broad Bioimage Benchmark Collection webiste <https://bbbc.broadinstitute.org/BBBC041/>
•	**Data Cleaning & Preparation**
   - Data cleaning will involve checking for corrupted images, duplicate images, visually similar images, and class imbalance in the various categories.
   - Data preparation will involve the following:
   - Splitting images in the image sub folder into 2 folders: test & train images as the YOLOv8 model requires a specific folder structure i.e.
   - 
    <img width="139" height="157" alt="Screenshot 2025-10-28 at 21 52 49" src="https://github.com/user-attachments/assets/c936bc6b-1d99-4a4a-ab82-5d8928b31afa" />

   - Creation of a data.yaml File in the dataset's root directory, that describes the dataset, classes, and other necessary information.
   - Convert existing JSON annotation format to the YOLOv8 format.The YOLOv8 format is TXT files with normalized bounding box coordinates. 
•	**Model Configuration:** Select or customize a YOLOv8 architecture suitable for multiclass detection. Adjust hyperparameters such as learning rate, batch size, and number of epochs based on dataset size and complexity.
•	**Training:** Train the YOLOv8 model on the annotated dataset using GPU acceleration if available, monitoring performance metrics like mean Average Precision (mAP) and loss curves to assess learning progress.
•	**Evaluation:** Evaluate the trained model on the test set, analyzing precision, recall, and mAP for each object class to determine detection accuracy and robustness.
•	**Deployment:** Integrate the trained YOLOv8 model into the target application, optimizing for real-time inference if required. Test the model in live scenarios to validate its practical effectiveness.
•	**Post-processing:** Apply non-maximum suppression and confidence thresholding to filter overlapping detections and minimize false positives.

Throughout the project, regular validation and hyperparameter tuning are performed to maximize detection performance across all object classes. The methodology ensures scalability and adaptability for various computer vision applications.

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

Upon downloading the dataset, a folder called labels was created and the two json files moved to the folder.

The training json file was renamed to train.json
  
## Machine learning Model Architecture

You Only Look Once (YOLO) v8 model will be used for object detection and counting. 


## Results

## Recommendations/Next Steps

## Future work/Suggestions

## Acknowledgements & Attributions

We used image set BBBC041v1, available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods, 2012)

## References
