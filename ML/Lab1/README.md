# Regression Models with Normal Equation & Gradient Descent

## Overview

This project demonstrates the application of both **single-variate linear regression** and **quadratic regression** to predict housing prices based on a feature (`lstat`) from the Boston housing dataset. The models are built using the **normal equation** method and **gradient descent** algorithm, implemented without using high-level machine learning libraries like scikit-learn.

## **Dataset**

The dataset used in this project is **Boston.csv**, which contains various features related to housing data.  It includes the following columns:
- **lstat**: The percentage of the population in a given area with a lower status.
- **medv**: The median value of owner-occupied homes (in thousands of dollars).

The project is divided into the following parts:
- **Part 1**: Single-variate linear regression using the normal equation.
- **Part 2**: Quadratic regression using the normal equation.
- **Part 3**: Implementing and testing the gradient descent algorithm to estimate model coefficients.

## Prerequisites

Make sure you have the following Python packages installed:
- **numpy**
- **matplotlib**
- **scipy**
- **sklearn** (for data splitting)

## Results
- **Single-Varate Linear Regression**: The linear model’s performance is evaluated using cost and R-squared metrics.
- **Quadratic Regression**: The model fitting is enhanced with a quadratic term for better prediction accuracy.
- **Gradient Descent**: The optimization of the model coefficients is visualized with a learning curve showing the reduction in cost over iterations.

## Conclusion
The project provides a step-by-step implementation of regression models, from basic linear regression to quadratic regression and gradient descent. The use of the **normal equation** ensures optimal solutions without relying on high-level libraries, while **gradient descent** offers a hands-on approach to iterative optimization.
