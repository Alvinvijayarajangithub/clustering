from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Drop the target (species) column for clustering
print("Dataset before preprocessing:")
print(data.head())
print("\nDropped species column since clustering is unsupervised.")
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(data)

# Visualization
sns.scatterplot(data=data, x='sepal length (cm)', y='sepal width (cm)', hue='KMeans_Cluster', palette='viridis')
plt.title("KMeans Clustering on Iris Dataset")
plt.show()
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Apply Hierarchical Clustering
linkage_matrix = linkage(data.iloc[:, :4], method='ward')
dendrogram(linkage_matrix)
plt.title("Dendrogram for Iris Dataset")
plt.show()

# Agglomerative Clustering
agglom = AgglomerativeClustering(n_clusters=3)
data['Hierarchical_Cluster'] = agglom.fit_predict(data)

# Visualization
sns.scatterplot(data=data, x='sepal length (cm)', y='sepal width (cm)', hue='Hierarchical_Cluster', palette='viridis')
plt.title("Hierarchical Clustering on Iris Dataset")
plt.show()
