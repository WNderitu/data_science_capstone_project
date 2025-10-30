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

## Project Description
This project employs YOLOv8 for multiclass object detection, leveraging its advanced capabilities to accurately identify and localize multiple object types within images or video frames. The methodology comprises several key steps to ensure robust and reproducible results.
- **Obtaining dataset**: Data will be obtained from Broad Bioimage Benchmark Collection webiste <https://bbbc.broadinstitute.org/BBBC041/>
- **Data Cleaning & Preparation**
   - Data cleaning will involve checking for corrupted images, duplicate images, visually similar images, and class imbalance in the various categories.
   - Data preparation will involve the following:
   - Splitting images in the image sub folder into 2 folders: test & train images as the YOLOv8 model requires a specific folder structure i.e.
 
      <img width="152" height="166" alt="Screenshot 2025-10-29 at 21 02 23" src="https://github.com/user-attachments/assets/d663834b-99f3-4814-9645-d91fe7469887" />

   - Creation of a data.yaml file in the dataset's root directory, that describes the dataset, classes, and other necessary information.
   - Convert existing JSON annotation format to the YOLOv8 format.The YOLOv8 format is TXT files with normalized bounding box coordinates.
- **Model Configuration:** Select or customize a YOLOv8 architecture suitable for multiclass detection. Adjust hyperparameters such as learning rate, batch size, and number of epochs based on dataset size and complexity.
- **Training:** Train the YOLOv8 model on the annotated dataset using GPU acceleration if available, monitoring performance metrics like mean Average Precision (mAP) and loss curves to assess learning progress.
- **Evaluation:** Evaluate the trained model on the test set, analyzing precision, recall, and mAP for each object class to determine detection accuracy and robustness.
- **Deployment:** Integrate the trained YOLOv8 model into the target application, optimizing for real-time inference if required. Test the model in live scenarios to validate its practical effectiveness.
- **Post-processing:** Apply non-maximum suppression and confidence thresholding to filter overlapping detections and minimize false positives.

Throughout the project, regular validation and hyperparameter tuning will be performed to maximize detection performance across all object classes. 

## Image Dataset

Images obtained from Broad Bioimage Benchmark Collection webiste <https://bbbc.broadinstitute.org/BBBC041/>

The dataset consists of:
- image folder
- training json file
- test json file

The dataset consists of 1,328 microscopic images of blood smears. A class label and set of bounding box coordinates were given for each cell.  

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
  
## Deep learning Model Architecture

The You Only Look Once (YOLO) model is a single stage detector that predicts bounding boxes and class probabilities directly from the entire input image in a single forward pass, which makes the model faster than other object detetcion models. The model treats object detection as a single regression problem. 

The YOLO version 8 model (YOLOv8) will be used for object detection and counting. The model size to be used is yolov8n (Nano) which has about 3 million parameters, is the fastest, suitable for small datasets and computers with limited GPU. However, it's accuracy is lower than other bigger sizes of YOLOv8 models. 

The model is dividied into three main components:
- **Backbone (feature extractor)** - this consists of the CNN that is responsible for extracting hierarchical features from the input image.
- **Neck** - this merges/fuses feature maps from the different stages of the backbone to capture information at various scales.
- **Head** - this is responsible for making predictions. It takes the merged features from the neck and outputs bounding box coordinates, class probabilities, and confidence scores for detected objects. The Head typically consists of multiple detection heads, each connected to a different output scale from the Neck, enabling the prediction of objects at various sizes. Post-processing techniques like non-maximum suppression (NMS) are applied to filter out redundant or overlapping bounding box predictions, resulting in the final set of detected objects.

<img width="1207" height="1122" alt="image" src="https://github.com/user-attachments/assets/4665efe1-8dd4-4cbc-b2f1-d57c7475b34c" />


Evaluation metrics used will be interesection over union (IOU), precision, recall, average precison(AP) and mean average precision (mAP).
- Intersection over Union (IoU): This measures the overlap between the model's predicted bounding box and the actual ground truth bounding box. An IoU of 1 means perfect overlap, while 0 means no overlap. A common threshold (e.g., 0.5 or 0.75) is set to consider a detection as a True Positive.
- Precision: This is the ratio of correctly predicted positive detections (True Positives) to the total number of positive detections (True Positives + False Positives). It tells you how accurate the model is when it predicts an object is present.
- Recall: This is the ratio of correctly predicted positive detections (True Positives) to the total number of actual positive objects in the image (True Positives + False Negatives). It tells you how many of the actual objects the model was able to find.
- Average Precision (AP): This is the area under the Precision-Recall curve for a single class. It gives a single number that summarizes the precision and recall performance for that class across different confidence thresholds.
- Mean Average Precision (mAP): This is the average of the Average Precisions (APs) across all the different classes. It provides an overall measure of the model's performance across all object categories.

## Results

## Recommendations/Next Steps

## Future work/Suggestions

## Acknowledgements & Attributions

1. We used image set BBBC041v1, available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods, 2012)
2. https://docs.ultralytics.com/datasets/

## References
1. https://www.who.int/health-topics/malaria#tab=tab_1
2. https://www.datacamp.com/blog/yolo-object-detection-explained
3. https://docs.ultralytics.com/datasets/
