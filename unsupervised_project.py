import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df=pd.read_csv("cluster_data.csv")
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)
plt.figure(figsize=(10,6))
sns.scatterplot(x=df["Feature 1"],y=df["Feature 2"])
plt.title("Scatter plot of Feature 1 vs Feature 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

X=df[["Feature 1","Feature 2"]] ### Selecting relevant features
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X) ### Feature Scaling
inertia=[] ### To store inertia values (It is the sum of squared distances to nearest cluster center)
K=range(1,11) ### Testing k values from 1 to 10
for k in K:
    kmeans=KMeans(n_clusters=k,random_state=42,n_init=10) 
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    
plt.figure(figsize=(10,6))
plt.plot(K,inertia,'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

optimal_k=3 ## Assuming from elbow plot we choose k=3 ie how many clusters we see in elbow plot
kmeans=KMeans(n_clusters=optimal_k,random_state=42,n_init=10)
kmeans.fit(X_scaled)
df['Cluster']=kmeans.labels_ ### Assigning cluster labels to original dataframe ie adding new column
plt.figure(figsize=(10,6))
sns.scatterplot(x=df["Feature 1"],y=df["Feature 2"],hue=df["Cluster"],palette="Set1")
plt.title("KMeans Clustering with optimal k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

centroids=kmeans.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],s=300,c='black',marker='X',label='Centroids')
plt.legend()
plt.show()


# Sil score evaluation >0.5 is good
sil_score=silhouette_score(X_scaled,df["Cluster"])
print("Silhouette Score for k =",sil_score)

# Predicting cluster for new data points
new_data=np.array([[2.5,3.5],[7.0,8.0],[1.0,0.5]])
new_data_scaled=StandardScaler().fit(X).transform(new_data)
predicted_clusters=kmeans.predict(new_data_scaled)
print("Predicted clusters for new data points:",predicted_clusters)


