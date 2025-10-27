import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\ML\supervised\classification\decison tree\financial_risk_assessment.csv")


cols_to_drop = [
    "Education Level", "Marital Status", "City", "State", "Country",
    "Marital Status Change", "Payment History", "Loan Purpose", "Gender"
]
df = df.drop(columns=cols_to_drop)


for col in ["Income", "Credit Score", "Loan Amount", "Assets Value"]:
    df[col].fillna(df[col].mean(), inplace=True)

df["Number of Dependents"].fillna(df["Number of Dependents"].median(), inplace=True)
df["Previous Defaults"].fillna(df["Previous Defaults"].median(), inplace=True)

df["Risk Rating"] = df["Risk Rating"].map({"Low": 0, "Medium": 1, "High": 2})
df["Employment Status"] = df["Employment Status"].map({"Unemployed": 0, "Employed": 1, "Self-employed": 2})


print("Class distribution before training:")
print(df["Risk Rating"].value_counts())


X = df.drop(columns=["Risk Rating"])
y = df["Risk Rating"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nPredicted values:\n", y_pred)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
