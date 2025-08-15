# 🛍️ Mall Customer Segmentation using K-Means

## 📌 Overview
This project applies **K-Means Clustering** to segment mall customers based on their **Annual Income** and **Spending Score**.  
The goal is to identify distinct customer groups for targeted marketing and business strategies.

---

## 📂 Dataset
**File:** `Mall_Customers.csv`  
- `CustomerID` → Unique customer identifier  
- `Gender` → Male / Female  
- `Age` → Age of the customer  
- `Annual Income (k$)` → Annual income in thousand dollars  
- `Spending Score (1-100)` → Score assigned based on spending habits

---

## ⚙️ Requirements
Install the required Python packages before running the code:

```bash
pip install pandas matplotlib scikit-learn
🚀 Steps Performed in Code
Load Dataset

python
Copy
Edit
df = pd.read_csv("Mall_Customers.csv")
Data Exploration – Check first rows, dataset info, and statistics.

Feature Selection – Use Annual Income (k$) & Spending Score (1-100) for clustering.

Optional PCA – Reduce dimensions for visualization if using more than two features.

Elbow Method – Determine the optimal number of clusters K.

K-Means Clustering – Fit K-Means model with chosen K.

Visualization – Scatter plot of customer segments.

Evaluation – Calculate Silhouette Score to evaluate clustering quality.

Save Results – Export clustered dataset to CSV.

📊 Example Output
Elbow Method Graph:
Shows how inertia changes with different values of K to choose the best number of clusters.

Cluster Visualization:
Scatter plot where each color represents a different cluster.

🧮 Silhouette Score
The Silhouette Score measures how well data points fit within their assigned clusters.
A score closer to 1 means better-defined clusters.

💾 Saving Results
The clustered dataset is saved as:

Copy
Edit
Mall_Customers_Clustered.csv
with an added Cluster column indicating the assigned group.

📌 How to Run
bash
Copy
Edit
python KMEANS.py
