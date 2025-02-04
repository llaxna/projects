# Lab 2: Logistic Regression One vs. All Multiclass Classifier
## Overview
This project implements a **multiclass classifier using Logistic Regression (One vs. All)**. The implementation is done in two parts:

1. **Without using sklearn** – Building logistic regression from scratch.
2. **Using sklearn** – Utilizing LogisticRegression from sklearn for classification.
   
## Data Description
- The dataset consists of three columns:
  - Feature_1 and Feature_2 (independent variables)
  - Class (target variable with three classes: 0, 1, and 2)
- Data is split into training and testing sets for model evaluation.
  
## Part 1: Logistic Regression Without Sklearn
This section implements logistic regression manually without using sklearn. The steps include:

#### 1. Data Visualization
- A scatter plot is created to visualize the distribution of the three classes based on Feature_1 and Feature_2.

#### 2. One vs. All Classification Setup
- The multiclass classification problem is converted into three separate binary classification problems (one classifier per class).
- For each classifier, class labels are modified so that the target class is assigned a value of 1 while all others are assigned 0.
  
#### 3. Adding a Bias Term and Splitting Data
- A column of ones is added to the feature matrix to account for the intercept term in logistic regression.
- The dataset is split into training and testing subsets.

#### 4. Defining the Sigmoid Function
- The sigmoid function is implemented to compute the probability of a given class.

#### 5. Computing the Cost Function
- The cross-entropy loss (logistic regression cost function) is implemented to measure the model’s performance.

#### 6. Gradient Descent Implementation
- The logistic regression parameters (betas) are updated iteratively to minimize the cost function.
- The model runs gradient descent until convergence is achieved.

#### 7. Decision Boundary Visualization
- The decision boundary for one of the classifiers is plotted to illustrate how the model separates different classes.

#### 8. Learning Curve Analysis
- A learning curve is plotted to show how the cost function decreases over iterations, helping assess the model’s convergence.

#### 9. Model Prediction and Evaluation
- The trained classifiers are used to predict class labels on the test data.
- Accuracy, precision, and recall metrics are computed to evaluate model performance.

## Part 2: Logistic Regression Using Sklearn
This section implements logistic regression using sklearn for efficiency. The steps include:

#### 1. Splitting Data
- The dataset is divided into training and test sets to assess model generalization.

#### 2. Creating and Training the Model
- A logistic regression model is initialized and trained using the training dataset.

#### 3. Predicting Class Labels
- The trained model predicts the class labels for the test set.

#### 4. Evaluating Model Performance
- The model’s accuracy, precision, and recall are calculated to measure classification effectiveness.

#### 5. Extracting Decision Boundary Coefficients
- The learned coefficients from the logistic regression model are printed to understand how the model separates different classes.

## Conclusion
This project demonstrates the implementation of a **One vs. All logistic regression classifier**, both manually and using sklearn. By comparing both approaches, we gain insight into how logistic regression functions internally while also appreciating the efficiency of using machine learning libraries.
