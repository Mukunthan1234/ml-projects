# -------------------------------
# Customer Segmentation using K-Means
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 1️⃣ Load dataset
df = pd.read_csv(r"C:\Users\Mukundhan\OneDrive\Desktop\datasets\CustomerDetails.csv")

# 2️⃣ Select features for clustering
X = df[["AccountBalance", "CIBIL_Score", "RV"]]

# 3️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Elbow Method (to find optimal k)
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method - Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.show()

# 5️⃣ Apply KMeans with optimal clusters (say 3)
model = KMeans(n_clusters=3, init="k-means++", random_state=42)
model.fit(X_scaled)

# 6️⃣ Add cluster labels to the original dataframe
df["Cluster"] = model.labels_

# 7️⃣ Analyze each cluster’s characteristics
cluster_summary = df.groupby("Cluster")[["AccountBalance", "CIBIL_Score", "RV"]].mean()
print("\nCluster Summary:\n")
print(cluster_summary)

# 8️⃣ Automatically assign readable labels based on mean values
# Identify which cluster has highest average values
cluster_rank = cluster_summary.mean(axis=1).rank().astype(int)

label_map = {}
for cluster_id, rank in cluster_rank.items():
    if rank == 3:
        label_map[cluster_id] = "High Value"
    elif rank == 2:
        label_map[cluster_id] = "Medium Value"
    else:
        label_map[cluster_id] = "Dormant"

df["CustomerSegment"] = df["Cluster"].map(label_map)

# 9️⃣ Show first few results
print("\nLabeled Customer Segments:\n")
print(df[["AccountBalance", "CIBIL_Score", "RV", "Cluster", "CustomerSegment"]].head())

# 🔟 Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1],
                hue=df["CustomerSegment"], palette="viridis", s=70)
plt.title("Customer Segmentation based on Account Balance & CIBIL Score")
plt.xlabel("Account Balance (scaled)")
plt.ylabel("CIBIL Score (scaled)")
plt.legend(title="Segment")
plt.show()

# 11️⃣ Save results
df.to_csv("clustered_customers.csv", index=False)

print("\nClustered dataset saved as clustered_customers.csv")


