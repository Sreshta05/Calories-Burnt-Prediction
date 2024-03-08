# Calories Burnt Prediction
This project aims to predict the calories burnt during workout sessions based on various features using machine learning regression models. It provides a web application interface built with Streamlit, allowing users to explore the dataset, visualize distributions, and evaluate different regression models for predicting calorie expenditure.

## Demo:

https://github.com/Sreshta05/Calories-Burnt-Prediction/assets/76899515/ec720775-7be7-4a8b-8017-9beba06f01bc


## Introduction

## Dataset Information

The dataset used for this project is sourced from Kaggle, specifically from the Calories Burnt Prediction Dataset. It consists of workout information for 15,000 individuals, including details such as age, gender, weight, height, calories burnt, and average heart rate and body temperature.

## Objective

The primary objective of this project is to predict the calories burnt during exercise sessions based on various demographic and physiological factors. By analyzing and modeling this data, individuals can gain insights into their calorie expenditure and make informed decisions to optimize their fitness routines.

## Dataset Overview

The project provides a detailed overview of the dataset, showcasing the first 15 individuals' workout information. It includes anonymized data on age, gender, weight, height, calories burnt, and average heart rate and body temperature. Additionally, statistical properties of numerical features are presented for a better understanding of the dataset's characteristics.

## Gender Distribution

The gender distribution among the first 15 individuals is visualized interactively using a countplot. This allows users to explore the proportion of male and female participants in the dataset, providing insights into the gender demographics of the study.

## Distribution Plot for Numerical Columns

Users can explore the distribution of numerical columns through an interactive sidebar selector. By selecting a specific numerical feature, a distribution plot is generated to illustrate the spread of data, enabling users to analyze the distribution patterns effectively.

## Correlation Heatmap

A correlation heatmap is provided to investigate the relationships between selected numerical columns. Users can select one or more columns from the sidebar to visualize the correlation matrix, facilitating the identification of potential correlations within the data.

## Combined Categorical and Numerical Distribution Plot

This section combines categorical and numerical variables to provide a holistic view of data distribution. Subplots display distributions for each variable, offering a comprehensive understanding of the dataset's characteristics and distribution patterns.

## Model Predictions and Evaluation

The project evaluates several regression models for predicting calories burnt during exercise sessions. Models such as XGBoost, Linear Regression, Decision Tree Regression, and Random Forest Regression are applied and evaluated based on their performance metrics, including R2 score, mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE). Additionally, distribution plots of residuals are provided to assess model performance visually.

## Dependencies: 

Streamlit

NumPy

Pandas

Matplotlib

Seaborn

TOML

XGBoost

## How To Use:

  Ensure the dataset files (calories.csv and exercise.csv) are placed in the cedata directory.

  Run the Streamlit app: streamlit run app.py
  
  Interact with the web application to explore the dataset, visualize distributions, and evaluate regression models for predicting calorie expenditure.
