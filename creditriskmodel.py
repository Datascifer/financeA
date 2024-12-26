### This code performs comprehensive credit risk analysis on lending data, which evaluates borrower default probability by examining factors like income, credit history, and debt ratios to inform lending decisions. The script implements a complete data processing pipeline using pandas for manipulation, scikit-learn for preprocessing, and matplotlib for visualization. It begins by loading lending data and analyzing missing values, followed by mean imputation for data completeness. The process continues with categorical variable encoding and numerical feature scaling to prepare the data for modeling. Throughout execution, the code generates visual insights including distribution plots and correlation heatmaps to understand variable relationships, with all transformations documented through print statements. The final processed dataset is exported for subsequent analysis, providing a clean, standardized dataset ready for credit risk modeling. 

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load lending Club dataset
print("Original Data:")
data = pd.read_csv('USE_YOUR_DATASET.csv', low_memory=False)
data.head(10)

# Data Exploration for missing data
## Count and visualize missing values
missing_values = data.isnull().sum()
missing_percentages = (missing_values / len(data)) * 100

print("\nMissing Values Count and Percentage:")
missing_data = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentages
})
print(missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

# Data Exploration
## Drop columns with all missing values
data = data.dropna(axis=1, how='all')

## Data Exploration with Plots
print("2. Data Exploration")
print("\nBasic information about the dataset:")
print(data.info())
print(data.describe())

## Create histograms for numeric columns
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
n_cols = len(numeric_cols)
plt.figure(figsize=(15, 5*((n_cols+2)//3)))  # Adjust figure size based on number of columns

for i, col in enumerate(numeric_cols, 1):
    plt.subplot((n_cols+2)//3, 3, i)
    plt.hist(data[col].dropna(), bins=50)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('numeric_distributions.png')
plt.close()

## Create bar plots for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
n_cat_cols = len(categorical_cols)
plt.figure(figsize=(15, 5*((n_cat_cols+2)//3)))

for i, col in enumerate(categorical_cols, 1):
    plt.subplot((n_cat_cols+2)//3, 3, i)
    value_counts = data[col].value_counts().nlargest(10)  # Show top 10 categories
    plt.bar(range(len(value_counts)), value_counts.values)
    plt.title(f'Top 10 Categories in {col}')
    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('categorical_distributions.png')
plt.close()

## Create correlation heatmap for numeric columns
plt.figure(figsize=(12, 8))
correlation_matrix = data[numeric_cols].corr()
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print("\nPlots have been saved as 'numeric_distributions.png', 'categorical_distributions.png', and 'correlation_heatmap.png'")

# Handling Missing Values
## Handling missing values with mean imputation
print("\nHandling Missing Values:")
numeric_df = data[numeric_cols].copy()  # Create a copy for numeric data
imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer.fit_transform(numeric_df)
print("\nMissing Data replaced with mean:")
print(data.head())

# Encoding Categorical Variables
## Encoding categorical variables
print("\nEncoding Categorical Variables:")
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col].astype(str))
print("\n1-hot encoding categorical data:")
print(data.head())

# Scaling numerical features
print("\nScaling Numerical Features:")
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
print("\nScaling numerical features using StandardScaler:")
print(data.head())

# Save processed data
data.to_csv('processed_lending_data.csv', index=False)
print("\nProcessed data saved to 'processed_lending_data.csv'")
