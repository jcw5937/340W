# An open codebase for enhancing transparency in deep learning-based breast cancer diagnosis utilizing CBIS-DDSM data

Publication can be accessed from [here]( https://www.nature.com/articles/s41598-024-78648-0).

## Notice

The following is from the repository of the original paper which this project builds on: https://pmc.ncbi.nlm.nih.gov/articles/PMC11549440/pdf/41598_2024_Article_78648.pdf 

Some additional code documents were added which were needed to adapt the pipeline to fit a cluster environment. These have been added to the steps section.

## Innovation:

Our innovation was SHAP (SHapley Additive exPlanations), is an interpretability method that uses a heatmap to display which parts of the processed images were used in classification. The SHAP model was added to the model_development_and_evaluation.py file and the updated name is cnn_with_shap.py.

## Overview

Accessible mammography datasets and innovative machine learning techniques are at the forefront of computer-aided breast cancer diagnosis. However, the opacity surrounding private datasets and the unclear methodology behind the selection of subset images from publicly available databases for model training and testing, coupled with the arbitrary incompleteness or inaccessibility of code, markedly intensifies the obstacles in replicating and validating the model's efficacy. These challenges, in turn, erect barriers for subsequent researchers striving to learn and advance this field. To address these limitations, we provide a pilot codebase covering the entire process from image preprocessing to model development and evaluation pipeline, utilizing the publicly available Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) mass subset, including both full images and regions of interests (ROIs). We have identified that increasing the input size could improve the detection accuracy of malignant cases within each set of models. Collectively, our efforts hold promise in accelerating global software development for breast cancer diagnosis by leveraging our codebase and structure, while also integrating other advancements in the field.


Below conatins the overview of appling the CBIS-DDSM mass subset for breast cancer diagnosis:

<div style="text-align: center;">
  <img width="970" alt="image" src="https://github.com/lingliao/Transparency-in-CABCDTD/assets/91222367/93f7aa76-4a39-4534-be60-ba14a795155f">
</div>


## Dataset Availability
The data utilized in this study is downloaded from [here]( https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629#2251662935562334b1e043a3a0512554ef512cad).

Full image size and the cropped area per 598 by 598 pixels are plotted as below:

<div style="text-align: center;">
  <img width="900" alt="image" src="https://github.com/lingliao/Transparency-in-CABCDTD/assets/91222367/b191aeb1-d923-43bb-965c-58ac206a0d2c">
</div>

## Methods
In general, our methods include: 1) converting DICOM to PNG format without altering bit depth, 2) mapping ROIs to corresponding full images to identify and crop abnormal areas while ensuring size congruence, 3) confirming sufficient crop size coverage for most abnormal regions, 4) appending cropped images to the preliminary target **598 × 598** pixels with centered abnormal areas and removal of unwanted backgrounds, 5) performing data augmentation for enhanced diversity, 6) processing and splitting images into training, validation, and testing sets for model development, 7) optimizing computational efficient Xception network for model development, and 8) assessing effectiveness using multiple matrices and visualizations.

Steps to run the code we provided for model development:

1. Convert_DICOM_to_PNG.py
2. Convert_DICOM_to_PNG_full.py
3. Map_ROI_to_Full_Images.py
4. Cropping.py
5. Size_adjustment.py
6. Augmentation_change.py
7. pathology.py
8. getting_the_weights.py
9. cnn__with_shap.py

## Results
The model's performance evaluation is based on the checkpoint with the highest validation accuracy.

The best performed checkpoint can be downloaded from [here](https://drive.google.com/drive/folders/18RxhTm9Oxak1dA0d2xihnHWekmEVug6h?usp=sharing)

Accuracy, precision, recall, F1 score, and ROC AUC are shown below:

<img width="545" height="156" alt="Screenshot 2025-12-02 at 6 56 01 PM" src="https://github.com/user-attachments/assets/8ec26699-7093-402f-bdb4-4102dc88d553" />

## Example output
Our example outputs, including data processing, data visualization and model development and evaluation, are saved in folder example_output.

## Identified images with unwanted white edges
Identified cropped images with unwanted white backgrounds are stored in folder image_with_white_edge.

## Others
Other outputs that the authors think might be helpful can be found under folder Others, including the calculated full image size, percentage per 598 by 598 areas for corpped images, pathology info, identified mismatch info in sizewize between ROI and full images.

(598_percentage_all.csv,
all_mass_pathology.csv,
heaght_width_FULL.csv,
mismated_test_image_info(original).csv,
and
mismated_train_image_info(original).csv.)

## We appreciate you attention.

**AI for IMPROVEMENT and for EVERYONE.**
