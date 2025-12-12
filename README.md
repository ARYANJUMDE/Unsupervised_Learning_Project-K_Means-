📊 Customer Clustering using K-Means

This project performs K-Means clustering on a dataset containing two numeric features (Feature 1 and Feature 2).
It includes complete preprocessing, visualization, optimal cluster detection, silhouette scoring, and prediction for new points.


📁 Dataset

The project uses:

cluster_data.csv

The dataset must contain at least:

1.Feature 1

2.Feature 2

🛠️ Technologies Used

1.Python

2.NumPy

3.Pandas

4.Matplotlib

5.Seaborn

6.Scikit-Learn



📌 Key Steps
1️⃣ Data Loading & Exploration

The script loads and explores data using:

1.df.head()

2.df.info()

3.df.describe()

4.Missing value check

5.Duplicate removal


2️⃣ Data Visualization

A scatter plot is created:
sns.scatterplot(x=df["Feature 1"], y=df["Feature 2"])

3️⃣ Feature Scaling

Features are scaled using:
StandardScaler()

4️⃣ Finding Optimal k (Elbow Method)

The elbow method loops from k = 1 to 10 and plots inertia.
for k in range(1,11):
    kmeans = KMeans(n_clusters=k)

5️⃣ Model Training

With chosen k = 3:
kmeans = KMeans(n_clusters=3)

6️⃣ Assign Cluster Labels

A new column is added:
df["Cluster"] = kmeans.labels_




7️⃣ Scatter Plot with Cluster Coloring

Clusters are visualized using Seaborn.

8️⃣ Plotting Centroids

Centroids are shown using:
plt.scatter(centroids[:,0], centroids[:,1], marker='X')

9️⃣ Silhouette Score

Silhouette score evaluates clustering quality:
silhouette_score(X_scaled, df["Cluster"])

🔟 Predicting Cluster for New Data Points

Example predictions:
new_data = np.array([[2.5,3.5],[7.0,8.0],[1.0,0.5]])


📈 Visual Outputs

The script generates:

✔ Elbow Method graph
✔ Original scatterplot
✔ Clustered scatterplot
✔ Centroid plot

🧪 Sample Output

1.Silhouette Score printed in console

2.Cluster labels added to DataFrame

3.Predicted cluster for new data points printed


▶️ How to Run
Step 1 — Install libraries

pip install numpy pandas matplotlib seaborn scikit-learn

Step 2 — Place dataset

Ensure cluster_data.csv is in the same folder as your Python script.

Step 3 — Run script
python clustering.py

📦 Requirements File

You can add a requirements.txt:
numpy
pandas
matplotlib
seaborn
scikit-learn

🧠 Insights

1.K-Means successfully groups similar customers.

2.Elbow method helps identify optimal number of clusters.

3.Silhouette score helps measure clustering performance.

4.The model can predict clusters for new data points.
