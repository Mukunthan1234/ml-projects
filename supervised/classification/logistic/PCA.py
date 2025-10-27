import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("../../ml/rd/wine_quality_classification.csv")
df["quality_label"] = df["quality_label"].map({"low": 0, "medium": 1, "high": 2})

x = df[["fixed_acidity", "residual_sugar", "alcohol", "density"]]
y = df["quality_label"]


scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)

print("\nStandardized Data:")
print(scaled_data)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

print("\nPCA Result (2 components):")
print(pca_data)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

X=pca_data
Y=df["quality_label"]
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
prec = precision_score(Y_test, y_pred, average='weighted')
rec = recall_score(Y_test, y_pred, average='weighted')
f1 = f1_score(Y_test, y_pred, average='weighted')


print("Confusion Matrix:\n", cm)
print("Accuracy:", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall:", round(rec, 4))
print("F1 Score:", round(f1, 4))
