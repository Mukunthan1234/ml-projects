import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("../../ml/rd/wine_quality_classification.csv")


print(df.isnull().sum())


df["quality_label"] = df["quality_label"].map({"low": 0, "medium": 1, "high": 2})

# Select features and target
X = df[["fixed_acidity", "residual_sugar", "alcohol", "density"]]
y = df["quality_label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate metrics
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print results
print("Confusion Matrix:\n", cm)
print("Accuracy:", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall:", round(rec, 4))
print("F1 Score:", round(f1, 4))

# Predict for a new sample
new = [[10.3, 10.4, 15.6, 2.0005]]
new_pred = model.predict(new)
print("Predicted Class for New Sample:", new_pred)
