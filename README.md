# Customer Segmentation using K-Means Clustering (Unsupervised Learning)

An end-to-end Unsupervised Machine Learning pipeline built with **Scikit-Learn** and **K-Means Clustering** to segment customers into distinct behavioral groups based on annual income and spending score metrics.

---

## 📌 Key Highlights
* **Unsupervised Customer Data:** Synthesized 1,000 customer behavioral records (Annual Income & Spending Score).
* **Feature Scaling:** Applied `StandardScaler` to normalize feature magnitudes for distance-based clustering.
* **Optimal K Determination:** Utilized the **Elbow Method (WCSS / Inertia Analysis)** to select $k = 4$ clusters.
* **Business Insights:** Categorized customers into 4 distinct profiles:
  * **Cluster 0 (Careful):** High Income, Low Spending.
  * **Cluster 1 (Budget-Conscious):** Low Income, Low Spending.
  * **Cluster 2 (VIP / Target):** High Income, High Spending.
  * **Cluster 3 (Careless Spenders):** Low Income, High Spending.
* **Pipeline Serialization:** Saved both trained `KMeans` model and `StandardScaler` using `joblib`.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib
