import pandas as pd
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

iris=datasets.load_iris()
df=pd.DataFrame(iris.data)
df.info()
df.describe()
 
df.head()

scaler=StandardScaler()
X_scaled_data=scaler.fit_transform(df)
print(X_scaled_data)  

model=PCA(n_components=2)
dec=model.fit_transform(X_scaled_data)
print(dec)


plt.scatter(dec[:,0],dec[:,1],c=iris.target)
 

#aglo clustring
import scipy.cluster.hierarchy as sc
plt.figure(figsize=(20,7))
plt.title("dendrogram")
#create dend
sc.dendrogram(sc.linkage(dec))
plt.title("dendrogram")
plt.xlable("sample Index")
plt.ylable("ecl dist")

cluster=AgglomerativeClustering(n_clusters=2)
cluster.fit(dec)

cluster.labels_
plt.scatter(dec[:,0],dec[:,1],c=cluster.labels_)
