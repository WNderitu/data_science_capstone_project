# Plasmodium Vivax (malaria) parasite detector and counter using YOLOv8 Model

## Introduction

Malaria, a life-threatening disease caused by Plasmodium parasites transmitted by female Anopheles mosquitoes, is most commonly found in tropical and subtropical regions. In 2023, the African Region was home to 94% and 95% of malaria cases and deaths, respectively. Sub-Saharan Africa carries a disproportionately high share of the global malaria burden. Plasmodium vivax is one of the 5 parasites that cause malaria associated with disease relapses due to dormant liver stages, contributing significantly to morbidity.  

The Plasmodium vivax lifecycle alternates between humans and Anopheles mosquitoes. Infection begins when a mosquito injects sporozoites into a human, which then move to the liver to either develop or remain dormant. Merozoites released from the liver infect red blood cells, progressing through ring, trophozoite, and schizont stages, causing malaria symptoms. Some parasites develop into gametocytes, enabling further transmission via mosquitoes.

Diagnosis of Plasmodium vivax malaria relies on examining stained blood smears via microsocopy and rapid diagnostic tests, with accurate detection being key to effective treatment and relapse prevention. Advanced methods like deep learning models, such as  Regional Based CNN, YOLOv8, and others would offer improved reliability and efficiency in identifying infections.

<img width="755" height="472" alt="image" src="https://github.com/user-attachments/assets/d55d4034-1d38-4192-a656-e65af73a03a8" />


Plasmodium Vivax Lifecycle

## Problem Statement

The detection of Plasmodium vivax malaria remains challenging, primarily due to the morphological similarities between parasite stages and other blood components observed in microscopic images. Conventional microscopy is a labor-intensive process that is susceptible to human error and demands specialized expertise, which may be limited in under-resourced settings. Consequently, there is an urgent need for automated, efficient, and dependable solutions capable of detecting and quantifying Plasmodium vivax parasites and their developmental stages within blood smears. Such advancements would enhance diagnostic accuracy, promote effective treatment, and strengthen malaria control initiatives.

## Project Objectives

1.	To develop a computer vision model for object detection and counting
2.	To detect uninfected red blood cells and leukocytes in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
3.	To count the number of uninfected red blood cells and leukocytes in human blood cells in a microscpic image from a blood smear using the developed YOLOv8 model
4.	To detect the growth stage of a Plasmodium Vivax (malaria parasite) in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
5.	 To count the number of detected growth stages of the plasmodium vivax parasites per microscopic image using the developed YOLOv8 model

## Image Dataset

- Images obtained from Broad Bioimage Benchmark Collection website <https://bbbc.broadinstitute.org/BBBC041/>
- The dataset consists of image folder, training json file & test json file
- There are 1,328 microscopic images of blood smears.
- Image resolution:1600x1200
- Class label & set of bounding box coordinates given for each image.
- 7 Class labels: red blood cell (uninfected), trophozoite, gametocyte, schizont, difficult, ring & leukocyte (uninfected)
- The Red Blood Cell and Leukocyte classes are blood cells that are not infected with the malaria parasite.
- The Trophozoite, Schizont, Ring and Gametocyte are different growth stages of the malaria parasite
- The difficult class label is for observed growth stages that couldn't be grouped into either of the 4 growth classes.
- Sample images from the dataset

<img width="491" height="325" alt="image" src="https://github.com/user-attachments/assets/24c25e2a-f8b3-47c6-837b-bb9d800e3430" />

<img width="447" height="319" alt="image" src="https://github.com/user-attachments/assets/dfdab58d-827d-496b-8da6-bcee41ffbd42" />

## Project Description
This project employs YOLOv8 for multiclass object detection, leveraging its advanced capabilities to accurately identify and localize multiple object types within images or video frames.

## Project Overview

<img width="909" height="225" alt="image" src="https://github.com/user-attachments/assets/93ad134d-cf7d-479a-a33d-67afb28369d8" />


## Deep learning Model Architecture

The You Only Look Once (YOLO) model is a single stage detector that predicts bounding boxes and class probabilities directly from the entire input image in a single forward pass, which makes the model faster than other object detection models. The model treats object detection as a single regression problem. 

The YOLO version 8 model (YOLOv8) will be used for object detection and counting. The model size to be used is yolov8n (Nano) which has about 3 million parameters, is the fastest, suitable for small datasets and computers with limited GPU. However, it's accuracy is lower than other bigger sizes of YOLOv8 models. 

The model is dividied into three main components:
- **Backbone (feature extractor)** - this consists of the CNN that is responsible for extracting hierarchical features from the input image.
- **Neck** - this merges/fuses feature maps from the different stages of the backbone to capture information at various scales.
- **Head** - this is responsible for making predictions. It takes the merged features from the neck and outputs bounding box coordinates, class probabilities, and confidence scores for detected objects. The Head typically consists of multiple detection heads, each connected to a different output scale from the Neck, enabling the prediction of objects at various sizes. Post-processing techniques like non-maximum suppression (NMS) are applied to filter out redundant or overlapping bounding box predictions, resulting in the final set of detected objects.

<img width="1207" height="1122" alt="image" src="https://github.com/user-attachments/assets/4665efe1-8dd4-4cbc-b2f1-d57c7475b34c" />

**Object detection evaluation metrics** used will be precision, recall, F1 Score and mean average precision (mAP).
  
- **Precision**: This is the ratio of correctly predicted positive detections (True Positives) to the total number of positive detections (True Positives + False Positives).It tells you how accurate the model is when it predicts an object is present. High precision = fewer false detections.
  
Precision = TP / (TP + FP)

- **Recall**: This is the ratio of correctly predicted positive detections (True Positives) to the total number of actual positive objects in the image (True Positives + False Negatives). It tells you how many of the actual objects the model was able to find. High recall = fewer missed detections.

Recall = TP / (TP + FN)

- **F1 Score**: Harmonic mean of precision and recall.YOLOv8 often reports best F1 (at optimal confidence threshold).

<img width="214" height="35" alt="Screenshot 2025-11-02 at 17 57 44" src="https://github.com/user-attachments/assets/293697af-cd78-4848-9c66-97ac24540aca" />

- **mAP@0.5** — IoU threshold = 0.5 (i.e., boxes overlap ≥ 50% to count as correct) - mean average precision calculated at a fixed IOU threshold of 0.50. This generally assesses whether the model can generaly detect the presence and approximate location of an object, and is a less less stric metric. 
 
- **mAP@0.5:0.95** — Mean mAP across IoU thresholds 0.5 to 0.95 (step 0.05) - average of the mean average precision calculated across multiple IoU thresholds, ranging from 0.50 to 0.95 in steps of 0.05 (i.e 0.50, 0.55, 0.60,...,0.95). 

Other metrics to help understand mAP@0.5 & mAP@0.5-0.95 performance metrics in YOLOv8:
- **Intersection over Union (IoU)**: This measures the overlap between the model's predicted bounding box and the actual ground truth bounding box. An IoU of 1 means perfect overlap, while 0 means no overlap. A common threshold (e.g., 0.5) is set to consider a detection as a True Positive. Higher IoU = better localization accuracy. 

<img width="172" height="38" alt="Screenshot 2025-11-02 at 17 58 54" src="https://github.com/user-attachments/assets/7a788e3c-8038-46ee-90b7-c98468378cb3" />

## Results

### Data Preparation

Class Imabalance was noted in the train, val and test image subsets as shown in the charts below. The imbalance is severe with 96% of the objects being from the red blood cell class. This imbalance is inherent to human blood smears as they have more red blood cells than other cells found in blood. 

<img width="328" height="272" alt="image" src="https://github.com/user-attachments/assets/806b4cf5-4760-439a-af19-f9a34a1ab8d6" />

<img width="328" height="272" alt="image" src="https://github.com/user-attachments/assets/bc3f000b-9c7c-4c79-be59-1668a4564f4d" />

<img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/73eabde2-8a5d-4718-9cdf-9b7ed1bb566d" />

Data preparation that was done is outlined in the table below. 

<img width="911" height="403" alt="image" src="https://github.com/user-attachments/assets/1400c142-084a-4acb-b597-1d8babb4aa9a" />


### Model Training

Three trains/iterations were done using the YOLOv8N model. The parameters applied are outlined in the table below:

<img width="911" height="440" alt="image" src="https://github.com/user-attachments/assets/ed8e0b83-89cd-4e78-9834-0f438988efdc" />

### Model Evaluation

<img width="933" height="221" alt="image" src="https://github.com/user-attachments/assets/17ddd65d-c204-450c-af90-bda712e1f322" />

- Overall class precision : Train 2 highest (0.719) - fewest false positives detections
- Overall recall : Train 1 highest (0.74)  - largest fraction of actual objects detected
- Overall mAP50  & Overall mAP50-95 : Train 3 – highest (0.743 & 0.626 ). Has better bounding box localization & classification at IoU 0.5 & best at stricter localization, performing well across multiple IoU thresholds (0.5 to 0.95).
- The 3 models performs well on classes with many instances, but performance drops significantly for rare classes especially schizont & gametocyte classes.
- Only trophozoite parasite class with good performance

#### Error Analysis

<img width="310" height="229" alt="image" src="https://github.com/user-attachments/assets/8164c651-ff25-4769-bfd6-3c04f8d1d0b6" />

<img width="443" height="147" alt="Screenshot 2025-11-13 at 20 33 38" src="https://github.com/user-attachments/assets/bcbdc428-4114-4c61-909a-df2f2b8ea756" />

Train 1
- Rapid drop to ~2.9 (train) & ~2.4 (val), curves closely follow each other, val loss consistently below train loss, good convergence (~ epoch 100 – 150)  slight underfit to train set. Best results - epoch 209, early stop: 229/300

<img width="318" height="222" alt="image" src="https://github.com/user-attachments/assets/fbdd5a35-e733-43ec-84fa-e6b49563ce16" />

<img width="420" height="148" alt="Screenshot 2025-11-13 at 20 34 05" src="https://github.com/user-attachments/assets/aebc66ff-3fdd-4897-b8e2-c965ceddb51e" />

Train 2
- Rapid drop to ~ 3.5 (train) & ~4.1(val), curves closely follow each other – minimal overfitting & model learning well, good convergence, gap small & stable – good generalization. Best results - epoch 100, early stop: 250/300

<img width="324" height="226" alt="image" src="https://github.com/user-attachments/assets/b03bd8cf-4f97-4aa2-ada5-5b60ee9f456f" />

<img width="418" height="148" alt="Screenshot 2025-11-13 at 20 34 24" src="https://github.com/user-attachments/assets/aeec41f5-00fd-4e43-8ad0-b0a7e2856668" />

Train 3
- Gap between curves is  ↑ with ↑ epochs - overfitting. Best results- epoch 183, early stop: 483/500. Model finding it harder to generalize  than to memorize training data. High volatility – struggle to find optimal weights for val set
- Based on error analysis, train 2 performed the best.

### Model Evaluation on Test Images

<img width="936" height="229" alt="image" src="https://github.com/user-attachments/assets/91737308-6206-493a-80b5-f5f599248597" />

Performance was poor compared to the validation results
Train 3 could not detect ring, schizont, gametocyte & difficult classes – could not generalize
Train 1 & 2 could not detect schizont, gametocyte & difficult classes - could not generalize
Trophozites - Train 1 - highest mAP50 & mAP50-95, Train 3 - highest Precision & Train 2- highest recall
Ring - Train 1 – best precision, recall , mAP50 & mAP50-95
Based on this metrics, the best model in this context is Train 1. 

### Predicting on test images

When the best model was used to detect objects and count on test images, the model was noted to have some instances of false positives and classification errors. See image below.

<img width="715" height="439" alt="image" src="https://github.com/user-attachments/assets/f6dc94a0-fccf-45de-9d07-ce5d4e3e49d4" />
<img width="188" height="162" alt="Screenshot 2025-11-13 at 20 11 48" src="https://github.com/user-attachments/assets/3e2e98e5-2637-42ef-a6d0-d43ace544a6f" />

### Model Selection

Based on the evaluation metrics on the test set of images, the best model from train_1 iteration was selected & saved in ONNX format. 

### Deploymnet

The model is deployed on streamlit at <https://datasciencecapstoneproject-ecwfxwj4qhafsrktpxjvnv.streamlit.app/#class-counts-overview>

### Challenges
1. Running out of GPU resources on Colab.
2. Poor model performance (generalization) on test image subset, possibly due to:
Malaria infected blood smear images are inherently (naturally) highly imbalanced due to dominance of uninfected red blood cells. It therefore difficult to handle class imbalance in this context (not clear how to?)
3. Highly similar parasite morphological stages
4. Variations in shape of infected red blood cells containing different lifecycle stages
5. Use of smaller dataset
6. Choice of YOLOv8n model (smallest variant, lightweight, optimized for speed & low compute cost (usable on limited GPUs). This is because of the following reasons:
- The model has lower accuracy especially on complex or small objects. 
- The model is less robust to noisy/complex data such as small objects i.e microscopic images of parasite stages in red blood cells
- The model has limited learning capacity – has fewer parameters, may fail to capture nuanced visual features
- The model is sensitive to augmentation/hyperparameters, therefore careful tuning is needed

<img width="944" height="535" alt="Screenshot 2025-11-13 at 20 29 53" src="https://github.com/user-attachments/assets/bc6929fb-467f-4c23-80f5-6e068a751d8a" />

## Recommendations/Future work
1.Use malaria dataset as is
2. Use larger YOLOv8 model e.g. medium variant
3. Use 2 stages for detecting & classifying malaria parasites
  - 1st stage: detect uninfected red blood cells vs. infected red blood cells with YOLOv8 model
  - 2nd stage: classify infected red blood cells into various parasite stages with an image classifier model e.g. AlexNet, EfficientNet, GoogLeNet, ResNet , MobileNet, Vision Transformers (ViT) etc..

## Acknowledgements & Attributions
1. We used image set BBBC041v1, available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods, 2012)
2. Image of YOLOv8 model architecture from: https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5
3. Image of P.Vivax lifecylce. Quique Bassat, CC BY 4.0 <https://creativecommons.org/licenses/by/4.0>, via Wikimedia Commons

## References
1. https://www.who.int/health-topics/malaria#tab=tab_1
2. https://docs.ultralytics.com/datasets/
3. https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5]
4. Link to data: Broad Bioimage Benchmark Collection website https://bbbc.broadinstitute.org/BBBC041/

