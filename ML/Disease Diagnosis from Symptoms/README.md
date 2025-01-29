# Disease Diagnosis from Symptoms
## Overview
This project utilizes a **neural network** to predict potential diseases based on user-input symptoms. The model is trained on a dataset of symptoms and corresponding diseases, leveraging machine learning for accurate diagnosis.

## Features
✅ Predicts diseases based on multiple symptoms

✅ Uses a deep learning model with optimized architecture

✅ Implements batch normalization and regularization for better performance

✅ Provides a classification report and confusion matrix for evaluation

## Technologies Used
- **Python** 🐍
- **TensorFlow/Keras** (Neural Network)
- **Pandas & NumPy** (Data Processing)
- **Scikit-Learn** (Preprocessing & Evaluation)
- **Matplotlib & Seaborn** (Visualization)

## Dataset
**Kaggle Link**: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

The dataset consists of symptoms and their related diseases. It has been preprocessed for better accuracy.

## Model Architecture
- **Input Layer**: Encoded symptoms
- **Hidden Layers**: Dense layers with Batch Normalization and L2 Regularization
- **Output Layer**: Softmax activation for multi-class classification
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## Evaluation Metrics
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix** (Visual representation of predictions)
- **Training & Validation Loss/Accuracy Plots**
