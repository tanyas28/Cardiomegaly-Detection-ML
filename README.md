# Cardiomegaly-Detection-ML
## Overview
This repository contains code for a deep learning project aimed at detecting medical conditions from chest X-ray images using convolutional neural networks (CNNs). The project focuses on identifying cardiomegaly, a condition where the heart is enlarged, from a dataset of labeled chest X-ray images.

## Dataset
The dataset used consists of chest X-ray images labeled with medical conditions. Positive cases include images with cardiomegaly, while negative cases represent images without any findings.

## Model Architecture
The core of this project utilizes the InceptionV3 model, a state-of-the-art deep learning architecture pre-trained on the ImageNet dataset. InceptionV3 is known for its efficiency and performance in image classification tasks, leveraging a deep CNN with inception modules that capture features at various scales.

### Transfer Learning
Transfer learning is employed by fine-tuning the InceptionV3 model on our specific medical image dataset. By leveraging the pre-trained weights of InceptionV3, we can adapt the model to effectively classify chest X-ray images for cardiomegaly detection. This approach helps in achieving higher accuracy with less data and computational resources compared to training a model from scratch.

## Training and Evaluation
The dataset is split into training and testing sets, with data augmentation applied during training to enhance model generalization. The model is trained using binary cross-entropy loss and optimized with the Adam optimizer. Training progress and performance metrics such as accuracy and loss are visualized using matplotlib.

## Results and Validation
The trained model's performance is evaluated on a separate test set to assess its ability to generalize to unseen data. Metrics like sensitivity, specificity, and the receiver operating characteristic (ROC) curve are used to evaluate and visualize the model's performance in detecting cardiomegaly.

## Deployment and Usage
The final trained model is saved and can be deployed for inference on new chest X-ray images. A demonstration of model usage is included, showing how to load the saved model and classify a new image.

## Testing on New Images
The model's efficacy is demonstrated by testing on an image downloaded from the internet, which was not part of the original dataset. This test showcases the model's ability to correctly predict the presence of cardiomegaly.
## Acknowledgements
- The This code is based on the [medical-ai](https://github.com/adleberg/medical-ai) repository by [Jason Adleberg](https://github.com/adleberg).
