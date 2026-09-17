# 🧠 MRI Brain Tumor Detection

An AI/ML-based web application for detecting and classifying brain tumors from MRI images using Deep Learning and Transfer Learning.

The system classifies MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The trained model is integrated with a Flask web application where users can upload an MRI image and receive a predicted class with confidence. The project also includes model evaluation and Grad-CAM based visualization.

---

## 📌 Project Overview

Brain tumors require accurate diagnosis and timely medical attention. This project demonstrates how Deep Learning can be used to classify brain MRI images into different tumor categories.

The project uses **EfficientNetB0 with Transfer Learning**. The pretrained ImageNet model is used as the feature extractor, followed by a custom classification layer for the four MRI classes.

### Workflow

MRI Image  
↓  
Image Preprocessing  
↓  
Data Augmentation  
↓  
EfficientNetB0  
↓  
Feature Extraction  
↓  
Global Average Pooling  
↓  
Dropout  
↓  
Dense Layer (4 Classes)  
↓  
Softmax Prediction  
↓  
Predicted Tumor Class + Confidence

---

## ✨ Features

- MRI brain tumor classification
- Four-class classification
- EfficientNetB0 transfer learning
- Image augmentation
- Model training and validation
- Test-set evaluation
- Accuracy, Precision, Recall and F1-score
- Confusion Matrix
- Classification Report
- Grad-CAM visualization
- Flask-based web application
- Prediction history
- Trained model saved in Keras format

---

## 🧬 Tumor Classes

| Class | Description |
|---|---|
| Glioma | A tumor originating from glial cells |
| Meningioma | A tumor arising from the meninges |
| No Tumor | MRI image without a detected tumor class |
| Pituitary | Tumor involving the pituitary region |

---

## 🤖 Model Architecture

The project uses **EfficientNetB0** with Transfer Learning.

### Architecture

```text
Input Image (224 × 224 × 3)
          ↓
Data Augmentation
          ↓
EfficientNetB0
(ImageNet Weights)
          ↓
Global Average Pooling
          ↓
Dropout (0.3)
          ↓
Dense Layer (4 neurons)
          ↓
Softmax
          ↓
4-Class Prediction