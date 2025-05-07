import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
try:
    df = pd.read_csv('Students_games_and_success_Rate.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The CSV file 'Students_games_and_success_Rate.csv' was not found. Please make sure the file is in the correct directory.")
    exit()

# --- Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# 1. Data Cleaning: Handling Missing Values
print("\n--- 1. Data Cleaning: Handling Missing Values ---")
numerical_cols = df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("Missing values handled.")

# 2. Data Integration: Not applicable

# 3. Data Reduction: Attribute Subset Selection
print("\n--- 3. Data Reduction: Attribute Subset Selection ---")
selected_columns_cluster = ['math_score', 'reading_score', 'writing_score', 'gaming_hours']
df_cluster = df[selected_columns_cluster].copy()
print(f"Selected columns for clustering: {df_cluster.columns.tolist()}")

# 4. Data Transformation: Normalization
print("\n--- 4. Data Transformation: Normalization ---")
scaler_cluster = MinMaxScaler()
df_cluster_scaled = pd.DataFrame(scaler_cluster.fit_transform(df_cluster), columns=df_cluster.columns)
print("Numerical features normalized for clustering.")

# 5. Data Discretization: Not directly used for K-Means in this example

# --- Model Application: Cluster Analysis (K-Means) ---
print("\n--- 3. Model Application: Cluster Analysis (K-Means) ---")

# Choosing the number of clusters (Elbow Method)
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_cluster_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Based on the elbow method, let's choose a number of clusters (e.g., 3)
n_clusters = 3
kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans_model.fit_predict(df_cluster_scaled)

print(f"\nK-Means clustering applied with {n_clusters} clusters.")
print("\nCluster assignments for the first few students:")
print(df[['math_score', 'reading_score', 'writing_score', 'gaming_hours', 'cluster']].head())

# Analyze the clusters (optional)
print("\nCluster Analysis - Mean values of features per cluster:")
print(df.groupby('cluster')[['math_score', 'reading_score', 'writing_score', 'gaming_hours']].mean())

print("\nData Preprocessing and Cluster Analysis completed!")