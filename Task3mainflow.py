import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load the Dataset
df = pd.read_csv("C:/Users/Karthika Suresh/Downloads/Mall_Customers.csv")
print("Dataset Shape:", df.shape)
print("Dataset Info:")
print(df.info())
print("Missing Values:")
print(df.isnull().sum())
print("Duplicate Entries:", df.duplicated().sum())

# Step 2: Data Preprocessingencoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])  # Male -> 1, Female -> 0

# Standardize the numerical features (excluding CustomerID)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Step 3: Determine Optimal Clusters using Elbow Method
wcss = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Step 4: Apply K-Means Clustering with optimal clusters (e.g., k=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 5: Visualization using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segmentation (PCA Reduced)')
plt.show()

# Step 6: Generate Insights
cluster_summary = df.groupby('Cluster').mean()
print("Cluster Summary:\n", cluster_summary)

# Recommendations based on clusters
print("Marketing Recommendations:")
print("- High Spending Customers: Focus on loyalty programs.")
print("- Mid-Level Customers: Offer discounts to increase spending.")
print("- Low Spending Customers: Personalized promotions to enhance engagement.")
