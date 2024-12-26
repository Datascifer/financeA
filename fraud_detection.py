# This code implements fraud detection using decision trees and random forests. It loads credit card transaction data, processes it through decision tree and random forest classifiers, and evaluates model performance. The analysis includes SMOTE balancing for handling class imbalance, cross-validation for model stability, and feature importance analysis. Key components include model training, performance metrics comparison, and visualization of results through confusion matrices and feature importance plots, utilizing scikit-learn's machine learning capabilities alongside pandas for data manipulation and matplotlib for visualization.

# Decision Tree Analysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
url = 'https://raw.githubusercontent.com/marhcouto/fraud-detection/master/data/card_transdata.csv?raw=true'
data = pd.read_csv(url)

# Print the top 5 rows
print("\n--- First 5 rows of data ---")
print(data.head(5))

# Print summary stats
print("\n--- Summary Statistics ---")
print(data.describe())

# Event rate
print("\n--- Event Rate ---")
event_rate = data['fraud'].mean() * 100
print(f'Event Rate: {event_rate:.2f}%')

# Define the atrributes (X) and the label (y)
X = data.drop('fraud', axis=1)
y = data['fraud']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a decision tree classifier
model = DecisionTreeClassifier(max_depth=3) #max_depth is maximum number of levels in the tree

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print("\n--- Model Performance Metrics ---")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_rep)

# Visualize the decision tree
plt.figure(figsize=(25, 10))
plot_tree(model, 
          filled=True, 
          feature_names=['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',
                         'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order'],
          class_names=['Non-Fraud', 'Fraud'])
plt.show()

# Model building
## Import essential libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

## Initialize a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Using 100 trees

## Train the Random Forest model on the training data
rf_model.fit(X_train, y_train)

## Make predictions on the test data
rf_y_pred = rf_model.predict(X_test)

# Model Evaluation
## Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_confusion = confusion_matrix(y_test, rf_y_pred)
rf_classification_rep = classification_report(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)

print("\n--- Random Forest Model Performance Metrics ---")
print(f"Accuracy: {rf_accuracy:.2f}")
print("Confusion Matrix:")
print(rf_confusion)
print("Classification Report:")
print(rf_classification_rep)

# Comparative Analysis
## Compare metrics of Decision Tree and Random Forest
dt_precision = precision_score(y_test, y_pred)
dt_recall = recall_score(y_test, y_pred)

comparison_metrics = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Precision': [dt_precision, rf_precision],
    'Recall': [dt_recall, rf_recall],
    'Accuracy': [accuracy, rf_accuracy]
})

## Display the comparison
print("\n--- Model Comparison ---")
print(comparison_metrics)

## Visualize the comparison using a bar chart
comparison_metrics.set_index('Model', inplace=True)
comparison_metrics.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# Balancing the data
## Check class distribution in the original dataset
print("\n--- Class Distribution in the Original Dataset ---")
print(y.value_counts())

## Plot the class distribution
plt.figure(figsize=(6, 4))
y.value_counts().plot(kind='bar')
plt.title("Class Distribution in the Original Dataset")
plt.xlabel("Fraud (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

from imblearn.over_sampling import SMOTE

# SMOTE balancing
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Verify the balancing
print("\n--- Class Distribution After Balancing ---")
print(y_balanced.value_counts())

# Split the balanced dataset into training and testing sets
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

# Rebuild the Random Forest model
rf_model_balanced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_balanced.fit(X_train_balanced, y_train_balanced)

# Evaluate on the test set
rf_y_pred_balanced = rf_model_balanced.predict(X_test_balanced)

# Performance metrics
rf_balanced_accuracy = accuracy_score(y_test_balanced, rf_y_pred_balanced)
rf_balanced_precision = precision_score(y_test_balanced, rf_y_pred_balanced)
rf_balanced_recall = recall_score(y_test_balanced, rf_y_pred_balanced)
rf_balanced_confusion = confusion_matrix(y_test_balanced, rf_y_pred_balanced)

print("\n--- Balanced Random Forest Model Performance ---")
print(f"Accuracy: {rf_balanced_accuracy:.2f}")
print(f"Precision: {rf_balanced_precision:.2f}")
print(f"Recall: {rf_balanced_recall:.2f}")
print("Confusion Matrix:")
print(rf_balanced_confusion)

# Cross validation
from sklearn.model_selection import cross_val_score

## Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model_balanced, X_balanced, y_balanced, cv=5, scoring='accuracy')

print("\n--- Cross-Validation Results ---")
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.2f}")

# Feature Importance Analysis
## Get feature importances
feature_importances = rf_model_balanced.feature_importances_

## Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importances ---")
print(importance_df)

## Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()  # Highest importance on top
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
