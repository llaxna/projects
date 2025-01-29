# Subjective Wellbeing Questionnaire (SWQ) - Analysis

The Ministry of Health in Jordan conducted the Subjective Wellbeing Questionnaire (SWQ) survey to assess and enhance the subjective well-being of the population. The survey focuses on measuring different aspects of well-being, including **food security**, **resilience**, **mental health**, and **stress mindset**.

## Dataset Overview
The dataset is provided in an Excel file containing two sheets:

- **Sample Data Sheet**: Contains data from over 900 participants across Jordan.
- **Dictionary Sheet**: Provides a detailed explanation of each variable and its corresponding levels.

The primary goal of the analysis is to understand the various factors contributing to well-being, focusing on the following scales:

## Scales Used

### 1. Food Insecurity Experience Scale (FIES)
This scale measures access to adequate food resources through a series of 8 questions.

**Score Interpretation**:
- **0-1**: High food security
- **2-4**: Low food security
- **5-8**: Very low food security

### 2. Resilience Scale (RS)
This scale assesses participants' ability to recover from stress using a 5-point Likert scale for 6 statements.

**Score Interpretation**:
- **6-13**: Low resilience
- **14-21**: Normal resilience
- **22-30**: High resilience

### 3. Mental Disorder: Kessler Psychological Distress Scale (K10+)
This 10-question scale assesses psychological distress.

**Score Interpretation**:
- **10-19**: Likely to be well
- **20-29**: Likely to have a moderate disorder
- **30-50**: Likely to have a severe disorder

### 4. Stress Mindset (SMM)
This scale measures beliefs about the effects of stress through 8 statements rated on a 0-4 scale.

**Score Interpretation**:
- **0-10**: Likely to have a debilitating effect
- **11-21**: Likely to have a moderate effect
- **22-32**: Likely to have an enhancing effect

## Objectives

### 1. Statistical Analysis
- The first step was to identify and address any data discrepancies or missing values.
- Summary statistics were generated to gain insights into the cleaned dataset, helping in understanding the distribution of scores across different scales.
- The cleaned dataset was prepared for further analysis to ensure the quality and accuracy of the data.

### 2. Exploratory Data Analysis (EDA)
- An exploratory analysis was performed to summarize the cleaned dataset. This involved looking at the distribution of responses for each scale and identifying patterns or trends.
- Visualizations, such as histograms and bar charts, were created to better understand the distribution of food security, resilience, mental distress, and stress mindset scores.
- A map was generated to visualize participant distribution across various governorates of Jordan, which helped in identifying the geographical representation of the data.

### 3. Explanatory Data Analysis
- External factors like **gender**, **age**, and **location** were examined to see their influence on the well-being scores.
- Regression analysis was performed to understand the relationship between demographic factors (e.g., gender, age, physical activity) and scores on resilience and mental distress scales.
- **Gender** and **age** were found to have significant effects on resilience and mental health, with females generally reporting lower resilience scores and younger participants reporting higher distress.

### 4. Statistical Modeling
- Linear regression models were applied to assess the relationship between resilience scores and external factors such as **gender**, **age**, and **physical activity**.
- ANOVA tests were used to explore the differences in well-being measures across different demographic groups.
- Additional analysis, such as Q-Q plots, was performed to assess the normality of the data and validate the assumptions of the statistical tests.

## Data Description

### 1. Demographic Data
The dataset includes demographic details of over 900 participants from Jordan, such as:
- **Gender**: Male, Female
- **Age**: Various age groups (18-22, 23-29, 30-39, etc.)
- **Location**: Governorates of Jordan

### 2. Wellbeing Measures
The dataset contains the values for the scales used to measure:
- **Food Security**: Assessing access to adequate food resources.
- **Resilience**: Evaluating the ability to recover from stress.
- **Mental Health**: Using the Kessler Psychological Distress Scale (K10+).
- **Stress Mindset**: Evaluating the belief about the effects of stress.

## Conclusion

- The analysis revealed significant insights into the factors that influence well-being in Jordan, such as **gender**, **age**, and **physical activity**.
- **Gender** differences were noted, with females generally reporting lower resilience and higher levels of mental distress.
- **Age** played a role, with younger participants experiencing higher levels of distress compared to older individuals.
- **Geographical differences** were observed, with some governorates showing higher participant responses in resilience and food security scales.
- The findings from this analysis will help inform policies aimed at improving the overall well-being of the population in Jordan.


