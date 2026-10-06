# 🧠 MRI Brain Tumor Detection

An AI/ML-based web application for detecting and classifying brain tumors from MRI images using Deep Learning, Transfer Learning, and Explainable AI.

The system classifies MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

The trained deep learning model is integrated with a Flask web application where users can upload an MRI image and receive a predicted class with confidence. The project also includes model evaluation and Grad-CAM-based visualization for model interpretability.

---

## 📌 Project Overview

Brain tumors require accurate diagnosis and timely medical attention. This project demonstrates how Deep Learning and Transfer Learning can be used to classify brain MRI images into different categories.

Multiple CNN architectures were explored during development, including EfficientNetB0 and ResNet50.

The final evaluated model is **ResNet50**, which achieved a test accuracy of **87.63%** on 1,600 test images.

---

## ✨ Features

- MRI brain tumor classification
- Four-class classification
- Transfer Learning
- ResNet50 deep learning model
- EfficientNetB0 experimentation
- Image preprocessing
- Data augmentation
- Model training and validation
- Test-set evaluation
- Accuracy, Precision, Recall and F1-score
- Confusion Matrix
- Classification Report
- Grad-CAM visualization
- Flask-based web application
- Prediction history using SQLite
- Trained models saved in Keras format

---

## 🧬 Tumor Classes

| Class | Description |
|---|---|
| Glioma | Tumor originating from glial cells |
| Meningioma | Tumor arising from the meninges |
| No Tumor | MRI image classified as having no tumor |
| Pituitary Tumor | Tumor involving the pituitary region |

---

## 📊 Dataset

The dataset contains four classes:

```text
dataset/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│
├── validation/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
