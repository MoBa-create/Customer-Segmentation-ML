from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

np.random.seed(42)
n_samples = 1000

annual_income = np.random.normal(60, 25, n_samples).round(2)
spending_score = np.random.randint(1, 100, n_samples)

df = pd.DataFrame({
    "Annual_Income": annual_income,
    "Spending_Score": spending_score
})

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(x_scaled)

print("\n=== Cluster Summary (Mean Values) ===")
cluster_summary = df.groupby("Cluster").mean().round(2)
print(cluster_summary)

joblib.dump(kmeans, os.path.join(OUTPUTS_DIR, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "scaler.pkl"))
df.to_csv(os.path.join(OUTPUTS_DIR, "egmented_customers.csv"), index=False)

print("\nModel, Scaler, and Data saved successfully to 'outputs' folder!")