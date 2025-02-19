# Child Mind Institute Problematic Internet Use Prediction Using PCA & RF Classifier

This project aims to predict the Severity Impairment Index (`sii`) in children based on features pertaining to demographics, physical health, and psychological factors. A Random Forest (RF) Classifier model is derived from training data to classify severity on a scale of 0-3: 
- 0: No impairment
- 1: Mild impairment
- 2: Moderate impairment
- 3: Severe impairment

## Project Overview

The dataset comprises both training and test data. The primary focus is to clean and preprocess the data, extract meaningful features, and then employ machine learning algorithms (PCA and RF) to predict the Severity Impairment Index based on relevant features.

## Steps Involved

### 1. **Importing Required Libraries**
For data manipulation, analysis, and visualization, these libraries are utilized:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical calculations and manipulating arrays (e.g., filling missing values).
- `matplotlib.pyplot`: For creating visualizations (boxplots and rendering of other graphs).
- `seaborn`: For easier visualization and the use of heatmaps and boxplots.
- `sklearn.model_selection`: For splitting dataset into training and testing sets (`train_test_split`).
- `sklearn.preprocessing`: For feature scaling using `StandardScaler`.
- `sklearn.decomposition`: For dimensionality reduction with `PCA`.
- `sklearn.ensemble`: For building Random Forest Classifier model (`RandomForestClassifier`).
- `sklearn.metrics`: For evaluating model performance with metrics like `classification_report` and `confusion_matrix`.

### 2. **Data Loading and Initial Exploration**
Data is imported using `pandas.read_csv()`.

## **Data Cleaning and Preprocessing**
The dataset is loaded from CSV files, and these steps are applied:
- **Missing Values**: Columns with more than 50% missing values are excluded from the dataset. Remaining columns with missing values are filled with zeroes.
- **Target Variable Cleaning**: Rows with missing values in the target variable (`sii`) are dropped.
- **Categorical Variables**: Categorical columns (e.g., `'SDS-Season'`, which is season of participation in the Sleep Disturbance Scale) are encoded using numeric values.
- **Feature Selection**: Features that are present in both the training and testing datasets are used for modeling.
- **Dropping Unnecessary Columns**: The `id` column is removed as it doesn't contribute to the model's prediction.

## **Exploratory Data Analysis (EDA)**
EDA is performed to understand the relationship between various features and the target variable:
- **Boxplots** are created for categorical and numerical columns to analyze their relationship with `sii`.
- A **correlation matrix** is used to quantify the correlation between features, and then visualized via a **heatmap**.

## **Model Training**
- Dataset is split into **training (80%)** and **testing (20%)** sets.
- Features are **standardized** to have zero mean and unit variance using `StandardScaler`.
- **Principal Component Analysis (PCA)** is applied for dimensionality reduction, preserving 95% of the variance.
- A **Random Forest Classifier** is trained using PCA-transformed features.

## **Model Evaluation**
After training the model, the performance is evaluated using these metrics:
- **Classification Report**: Includes precision, recall, and F1-score metrics.
- **Confusion Matrix**: This helps visualize the model's prediction performance.
- **Accuracy**: Model's accuracy on test set is displayed.

## **Results**
After model evaluation, the trained Random Forest model is used to predict `sii` on the test set. Final predictions are saved to a CSV file, which includes the test set IDs and predicted `sii` values.

## **Usage**
- **Data Files**: Ensure that dataset files (`train.csv` and `test.csv`) are available in the appropriate directory.
- **Running the Script**: The script can be run and executed in a Python environment.
- **Predictions**: After running the script, the predicted `sii` values for the test set will be saved as `submission.csv`.

## Dataset Acknowledgement
This project uses data from the [Problematic Internet Usage competition](https://www.kaggle.com/competitions/child-mind-institute-detecting-problematic-internet-use) hosted by Kaggle. Due to competition rules, the dataset cannot be shared here.
