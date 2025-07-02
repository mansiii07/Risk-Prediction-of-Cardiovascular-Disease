INTRODUCTION 
 
Machine Learning: 
Machine Learning (ML) is a branch of artificial intelligence that focuses on developing systems capable of learning and making decisions without being explicitly programmed. At its core, it uses algorithms and statistical models to analyze the data, visualize the data, find patterns, and improve its performance over time. 
 
Risk Prediction of Cardiovascular Disease: 
Risk prediction for cardiovascular disease (CVD) involves identifying individuals at higher risk of developing heart-related conditions based on various factors. These factors include age, gender, height, weight, BMI, smoking history, alcohol consumption, fruit consumption, green vegetables consumption, fried potato consumption and medical conditions such as depression, diabetes, and arthritis. 
 
 
 
 
PROJECT OVERVIEW:
 
Aim: The aim of this project is to develop a predictive model for identifying patients at risk of cardiovascular disease. 
 
Dataset: The dataset used in this project is CVD dataset, which includes features like  age, gender, height, weight, BMI, smoking history, alcohol consumption, fruit consumption, green vegetables consumption, fried potato consumption and medical conditions such as depression, diabetes, and arthritis. 
 
Objective:  
 
•	To clean and preprocess the CVD dataset. 
•	To perform exploratory data analysis to identify key risk factors. 
•	To build and evaluate classification models using logistic regression, knn and random forest. 
 
IMPORT NECESSARY LIBRARIES  
 
•	Pandas- import pandas as pd 
     For data manipulation, typically used for handling tables(DataFrames). 
•	Numpy- import numpy as np 
For numerical operations, especially with arrays and matrices. 
•	Seaborn- import seaborn as sns 
For making beautiful statistical plots, built on top of matplotlib. 
•	Matplolib- import matplotlib.pyplot as plt 
For creating visualizations like line plots, bar charts, etc. 
•	Warnings- import warnings 
To control or suppress warning messages. 
•	warnings.filterwarnings('ignore'): 
Disables warning messages from showing up during execution. 
 
LOAD DATASET 
 
Reading the dataset using the pd.read_csv (“ ”). 
This step allows us to load the raw data into a structured format (a DataFrame), which can be easily explored and manipulated using pandas. 
 
DATA EXPLORATION 
 
•	.head ( ):  It shows the first five rows of table.  
•	.tail ( ):  It shows last five rows of table.  
•	.shape:  It shows the total no. of rows and no. of columns of the data. 
•	.info ( ):  It shows the indexes, columns, data-types of each column, memory at once.  
•	.describe ( ):  It provides an overview of a dataset’s central tendency, dispersion and distribution.  
•	.isnull().sum():  Counts missing (NaN) values per column. 
 
•	.duplicated ( ).sum ():  To check the sum of duplicate values in data.  
DATA CLEANING 
 
	Removing Duplicate Records: There are 80 duplicate values in the dataset. The drop_duplicates () function is used to remove any rows that are exact duplicates in the dataset. 
	Correcting Data Inconsistencies:  Replacing valid data with incorrect or default values. 
 
EDA 
 
Exploratory Data Analysis (EDA) is the process of analyzing data sets to summarize their main features using statistics and visual tools. It helps to understand the structure, patterns, trends, and relationships in the data, and to detect missing values or outliers. EDA is an essential step before applying any machine learning or statistical modeling. 
 
Measure of Central Tendency- Mean, Median, Mode 
•	"Mean": df [num_cols].mean () Calculates the mean (average) of each numerical column.  
•	"Median": df [num_cols].median () Calculates the median (middle value) of each numerical column.  
•	"Mode": df [num_cols].mode ().iloc [0] Calculates the mode (most frequent value) of each numerical column. 
 
The above values are the calculated mean, median, and mode for each numerical column. 
 
DATA VISUALIZATION 
Visualization is a key step in Exploratory Data Analysis (EDA) to understand the underlying patterns, distributions, and relationships within the dataset. It helps identify trends, detect outliers, and uncover correlations between features. 
UNIVARIATE ANALYSIS: Analysis of a single variable to understand its distribution and characteristics. 
 
 
** COUNTPLOT:

1.The most individuals report their general health as either “Very Good” or “Good”, while very few rate it as “Poor”. 
2.The majority of individuals engage in physical exercise, while a smaller portion do not. 
3.The majority of individuals in the dataset do not have heart disease, indicating a class imbalance. 
4.The majority of individual do not have skin cancer, showing a class imbalance in data 
5.The most individuals in the dataset do not have a history of smoking. 


** HISTOGRAM: 
 
The histograms indicate that most individuals in the dataset have average height, weight, and BMI, while the majority show low consumption of alcohol, fried potato, and green vegetables with moderate fruit consumption 

   
** PIE CHART: 
 
The pie chart indicates that the majority of individuals in the dataset belongs to the age groups of 60-64 and 70-74, while other age groups fairly evenly spread. 
 
 
BIVARIATE ANALYSIS: Analysis the relationship between two variables. 
 
**NUMERICAL VS. NUMERICAL: The scatterplot and the lineplot shows a strong relationship between weight and BMI, indicating that BMI increases steadily with weight. 
**CATEGORICAL VS. CATEGORICAL:
1.The countplot shows that a higher number of both males and females report exercising, with females having a slightly greater proportion of exercise than males.  
2.The countplot shows that more females report depression than males, although most individuals of both males and females do not have depression. 
3.The countplot shows that more females have arthritis than males, but in both males and females, most individuals do not have arthritis. 
4.The countplot shows that more females do not have smoking history, while smoking and non-smoking counts are nearly balanced among males. 
**NUMERICAL VS. CATEGORICAL: 
1.The kde (kernal density estimate) plot shows that individuals who exercise tend to have lower BMI values compared to those who do not exercise. 
2.The kde plot shows that the individuals with better general health (Excellent/Very Good) tend to have lower BMI values, while higher BMI values are more associated with poor health rating. 
3.The kde plot shows that the individuals who exercise generally have lower body weights, while those who do not exercise tend to have higher weight distribution. 


Encoders 
 
An encoder in data science and machine learning refers to a process or tool that transforms categorical data (non-numeric) into a numeric format so it can be used effectively in machine learning algorithms. 
Label Encoder: used to encode target variable or features into integers labels. 
Library used to import label encoder: 
from sklearn.preprocessing import LabelEncoder 
 
Outliers 
Outliers are data points that significantly differ from the other observations in a dataset. They appear unusually high or low compared to the majority of the data and can result from measurement errors, variability in the data, or rare events. 
 
Identify Outliers: 
Outliers can be identified easily through visual representation i.e. Boxplot. 
Boxplots: Graphical identification using the interquartile range (IQR). 
 
The boxplot shows that Height, Weight, and BMI have a wide range of values with many outliers, while the Alcohol, Fruit, Green Vegetable, and Fried Potato Consumption have smaller ranges but still include several outliers in the dataset. 
TREAT OUTLIERS: 
 
Calculate Quartiles: 
Q1=  df.quantile(0.25) i.e. 25th percentile of the column.  
Q3=  df.quantile(0.75) i.e. 75th percentile of the column.  
 
IQR Method: IQR =Q3-Q1 q_high= Q3 + (1.5*IQR) 
q_low= Q1 – (1.5*IQR) 
 
Capping: 
df [col] = np.where (df [col] < q_low, q_low, np.where (df [col] > q_high, q_high, df [col])) Applies conditional logic: 
If the value is below the q_low, it is replaced with q_low. 
If the value is above the q_high, it is replaced with q_high. 
Otherwise, the value remains unchanged. 
 
 
 
Dispersion of Data- min, max, range, variance, standard 
deviation, coefficient of variation 
  
SKEWNESS AND KURTOSIS 
 
 
Skewness is a statistical measure that tells you how asymmetrical a distribution of data is. In simple terms, it shows whether the data is balanced around the mean or leans more to one side.  
Types of Skewness:  
1.	Zero Skewness (0): The distribution is perfectly symmetrical. Think of a normal bell curve.  
2.	Positive Skew (Right Skew): The tail on the right side is longer. Most data is on the left, but a few very high values stretch the curve to the right.  
3.	Negative Skew (Left Skew): The tail on the left side is longer. Most data is on the right, but a few low values stretch the curve to the left. 
 
Kurtosis gives deeper insights into the variability and extremities in your data beyond just mean and variance. Kurtosis measures the "tailedness" of a data distribution: 
1.	High kurtosis: Indicates heavy tails and more outliers. 
2.	Low kurtosis: Suggests lighter tails and fewer outliers. 
3.	Normal kurtosis: When the kurtosis is close to 3 (mesokurtic distribution).  
 
FEATURE SCALING 
Feature scaling is the process of normalizing the range of independent variables (features) in the dataset. It ensures that no single feature dominates others due to its scale. Standard scaler: 
A Standard Scaler is a preprocessing technique in machine learning it transforms the data so that it has a mean of 0 and a standard deviation of 1. 
Library used to import Standard Scaler: 
from sklearn.preprocessing import StandardScaler 
 
 
VIF 
 
 Variance Inflation Factor (VIF) is a measure used to detect multicollinearity. Multicollinearity occurs when independent variables in a dataset are highly correlated, which can distort the predictions and interpretation of models. 
 
VIF = 1: No correlation between the variable and others. 
VIF > 1: Some correlation exists. 
VIF > 5 or 10: High multicollinearity, often considered problematic. 
 
HEAT MAP 
Heatmap is a graphical representation of data where individual values are shown using color gradients. It's commonly used to visualize the correlation between variables in a dataset, making it easy to identify patterns, trends, or strong relationships at a glance.  
 
  
  
TRAIN-TEST-SPLIT 
Splitting the data ensures the model is trained on one portion and evaluated on another, preventing overfitting 
To evaluate our machine learning model effectively, we split the dataset into training and testing subsets. 
Training Set: Used to train the machine learning model. 
Testing Set: Used to evaluate the model's performance on unseen data. 
Library used to import train- test- split: 
from sklearn.model_selection import train_test_split 
 
SAMPLING 
Sampling in machine learning refers to the process of selecting a subset of data points from a larger dataset. This is done to reduce computational costs, manage class imbalances, for model evaluation. 
Library used to import sampling: 
from imblearn.over_sampling import SMOTE 
 
LOGISTIC REGRESSION 
Logistic Regression is a statistical method used in machine learning model to find the relationship between independent variables and a binary dependent variable (i.e., outcomes that have two categories, like "yes/no" or "0/1"). Despite its name, logistic regression is not a regression algorithm but rather a classification technique. 
Library used to import logistic regression: 
from sklearn.linear_model import LogisticRegression 
 
KNN 
 K-Nearest Neighbors (KNN) is a simple, yet powerful machine learning algorithm used for both classification and regression tasks. It operates based on the principle of similarity, where data points that are close to each other are likely to have similar characteristics. Library used to import KNN Classifier: 
from sklearn.neighbors import KneighborsClassifier 
 
RANDOM FOREST 
 A Random Forest is an ensemble machine learning algorithm used for both classification and regression tasks. It combines multiple decision trees to improve prediction accuracy and reduce overfitting. Library used to import Random Forest Classifier: 
from sklearn.ensemble import RandomForestClassifier 
 
EVALUATION METRICS 
Library used to import metrics: 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
 
 Accuracy Score: The proportion of correctly classified instances out of all instances. 
Confusion Matrix: A table showing the number of true positives, false positives, true negatives, and false negatives.  
 
Classification Report: A summary of metrics for each class, including precision, recall, F1-score, and support. 
 
•	Accuracy: Measures the percentage of correct predictions. 
 
•	Precision: Measures how many of the predicted positive cases (CVD risk) were actual positives. 
 
•	Recall: Measures how many actual positive cases (those with CVD) were correctly identified. 
 
•	F1 Score: A balance between precision and recall, important when class distribution is imbalanced. 
 
MODEL EVALUATION 
 
•	Logistic Regression achieved an accuracy of 72.01%. 
•	K Nearest Neighbor achieved an accuracy of 71.3%. 
•	Random Forest achieved an accuracy of 85.04%. 
 
 
CONCLUSION 
 
We successfully built a predictive model that can estimate the risk of cardiovascular disease based on the medical features.  
 
The Random Forest model performed the best overall. 
 
