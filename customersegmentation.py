# Customer Segmentation
# This code performs customer segmentation analysis using multiple clustering algorithms. It loads customer data including age, income, and spending scores, then applies K-means, Hierarchical, and DBSCAN clustering methods. The analysis includes data visualization, optimal cluster determination through elbow and silhouette methods, and dimensionality reduction via PCA. The code compares clustering results across methods, visualizes cluster distributions, and provides comprehensive cluster characteristics through centroids and summary statistics.

# Data Exploration and Analysis
## Load essential libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

## Load the dataset
file_path = 'mallcustomers.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Inspect the first few rows
print(data.head())

# Check for missing values and basic statistics
print(data.info())
print(data.describe())

# Distribution of numerical features
numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for feature in numerical_features:
    plt.figure()
    plt.hist(data[feature], bins=15, alpha=0.7)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Gender breakdown
gender_counts = data['Gender'].value_counts()
plt.figure()
gender_counts.plot(kind='bar', color=['skyblue', 'pink'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Scatter plot matrix
from pandas.plotting import scatter_matrix

scatter_matrix(data[numerical_features], alpha=0.8, figsize=(10, 8))
plt.suptitle('Scatter Matrix of Numerical Features')
plt.show()

# Correlation heatmap
correlation_matrix = data[numerical_features].corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), numerical_features, rotation=45)
plt.yticks(range(len(correlation_matrix)), numerical_features)
plt.title('Correlation Heatmap')
plt.show()

# Age vs Annual Income colored by Spending Score
plt.figure()
plt.scatter(data['Age'], data['Annual Income (k$)'], 
            c=data['Spending Score (1-100)'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Spending Score')
plt.title('Age vs Annual Income (Colored by Spending Score)')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.show()

# Boxplots to compare spending behavior by gender
plt.figure()
data.boxplot(column='Spending Score (1-100)', by='Gender', grid=False)
plt.title('Spending Score by Gender')
plt.suptitle('')  # Remove default title
plt.xlabel('Gender')
plt.ylabel('Spending Score')
plt.show()

# Enhanced Clustering with K-Means

## Select relevant features for clustering
selected_features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

## Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_features)

## Apply KMeans clustering
num_clusters = 5  # This can be adjusted based on experimentation
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

## Prepare for Hierarchical Clustering
Z = linkage(scaled_data, 'ward')

wcss = []
cluster_range = range(1, 11)

# Calculate WCSS for each cluster count
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.metrics import silhouette_score

silhouette_scores = []

# Calculate silhouette scores for each cluster count (from 2 to 10)
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))

# Plot the silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Analysis for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Second kmeans
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Optimal Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize clusters
plt.figure(figsize=(8, 6))
for cluster in range(optimal_clusters):
    cluster_data = data[data['Optimal Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'], 
                label=f'Cluster {cluster}', alpha=0.7)

# Plot cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0] * scaler.scale_[0] + scaler.mean_[0], 
            centroids[:, 1] * scaler.scale_[1] + scaler.mean_[1], 
            s=200, c='red', marker='X', label='Centroids')

plt.title(f'K-Means Clustering with {optimal_clusters} Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Advanced Implementation of Hierarchical Clustering

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=10, leaf_rotation=45., leaf_font_size=12., show_contracted=True)
plt.title('Dendrogram (Hierarchical Clustering)')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

## Hierarchical clusters
from scipy.cluster.hierarchy import fcluster

# Define number of clusters
num_clusters = 5
data['Hierarchical Cluster'] = fcluster(Z, t=num_clusters, criterion='maxclust') # Extract hierarchical clusters
print(data[['Annual Income (k$)', 'Spending Score (1-100)', 'Hierarchical Cluster']].head())

## Compare clusters (K-Means vs Hierarchical)
comparison = data.groupby(['Cluster', 'Hierarchical Cluster']).size().unstack(fill_value=0)
print("K-Means vs Hierarchical Clustering Comparison:")
print(comparison)

## Plot hierarchical clusters
plt.figure(figsize=(8, 6))
for cluster in range(1, num_clusters + 1):
    cluster_data = data[data['Hierarchical Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], 
                label=f'Cluster {cluster}', alpha=0.7)

plt.title('Hierarchical Clustering: Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

## Compare K-Means clusters with Hierarchical clusters
comparison = pd.crosstab(data['Cluster'], data['Hierarchical Cluster'])

print("Comparison of K-Means and Hierarchical Clustering:")
print(comparison)


# Comprehensive Cluster Analysis
## Rescale centroids to original scale
centroids_rescaled = centroids * scaler.scale_ + scaler.mean_

## Create a DataFrame for centroids
centroid_df = pd.DataFrame(centroids_rescaled, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
centroid_df.index = [f'Cluster {i}' for i in range(5)]

print("Centroid Analysis:")
print(centroid_df)

## Group data by clusters and calculate means
cluster_summary = data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
cluster_summary['Count'] = data['Cluster'].value_counts()

print("Cluster Characteristics:")
print(cluster_summary)

# Advanced Clustering Techniques

from sklearn.cluster import DBSCAN

## Apply DBSCAN to the scaled data
dbscan = DBSCAN(eps=0.5, min_samples=5)
data['DBSCAN Cluster'] = dbscan.fit_predict(scaled_data)

## Compare DBSCAN clusters with K-Means and Hierarchical clustering
print("Cluster Comparison (K-Means vs DBSCAN):")
print(data[['Cluster', 'DBSCAN Cluster']].value_counts())

## Visualize DBSCAN clusters
plt.figure(figsize=(8, 6))
unique_clusters = np.unique(data['DBSCAN Cluster'])
for cluster in unique_clusters:
    cluster_data = data[data['DBSCAN Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                label=f'Cluster {cluster}', alpha=0.7)

plt.title("DBSCAN Clustering: Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# Dimensionality Reduction Technique
from sklearn.decomposition import PCA

## Apply PCA to the scaled data (2 components for visualization)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

## Visualize the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7)
plt.title("PCA Projection of Data with K-Means Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()

## Analyze explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance by Components: {explained_variance}")
