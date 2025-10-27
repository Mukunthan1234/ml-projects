import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans
import numpy as np

X,y=make_blobs(n_samples=1000,centers=3,n_features=2)

plt.scatter(X[:,0],X[:,1],c=y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(X_train)
x_test_scaled=scaler.fit_transform(X_test)


wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(x_train_scaled)
    wcss.append(kmeans.inertia_)
print(wcss)

plt.plot(range(1,11),wcss)
plt.title("wcss graph")
plt.xlable("k-value")
plt.ylabel("wcss")
plt.show()

model=KMeans(n_clusters=3,init="k-means++")
model.fit(x_train_scaled)
y_pred=model.predict(x_test_scaled)
y_pred

plt.scatter(X_test[:,0],X_test[:,1],c=y_pred)
