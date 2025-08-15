# ============================================
# K-Means Clustering - Mall Customer Segmentation
# ============================================

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 2. Load Dataset
# Make sure 'Mall_Customers.csv' is in the same directory
df = pd.read_csv("Mall_Customers.csv")
print("First 5 rows of dataset:")
print(df.head())

# 3. Check basic info
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 4. Select features for clustering
# You can modify based on task; here we take 'Annual Income' & 'Spending Score'
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 5. Optional: PCA for Visualization (if you want to plot in 2D from more features)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
plt.scatter(pca_result[:,0], pca_result[:,1])
plt.title("PCA Projection of Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 6. Find Optimal K using Elbow Method
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# 7. Fit K-Means with chosen K (example: K=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 8. Visualize Clusters
plt.figure(figsize=(8,6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
            c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title(f'Customer Segments (K={optimal_k})')
plt.show()

# 9. Evaluate Clustering using Silhouette Score
score = silhouette_score(X, df['Cluster'])
print(f"\nSilhouette Score: {score:.2f}")

# 10. Save clustered data
df.to_csv("Mall_Customers_Clustered.csv", index=False)
print("\nClustered dataset saved as 'Mall_Customers_Clustered.csv'")
