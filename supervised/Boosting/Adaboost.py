import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"C:\Users\Mukundhan\OneDrive\Desktop\datasets\Travel.csv")

df.info()
df.describe()
df.isnull().sum()


#clearing null values 
df.Age.fillna(df.Age.median(), inplace=True)
df.DurationOfPitch.fillna(df.DurationOfPitch.mean(),inplace=True)
df.TypeofContact.fillna(df.TypeofContact.mode(),inplace=True)
df.NumberOfFollowups.fillna(df.NumberOfFollowups.mean(),inplace=True)
df.PreferredPropertyStar.fillna(df.PreferredPropertyStar.mean(),inplace=True)
df.NumberOfTrips.fillna(df.NumberOfTrips.mean(),inplace=True)
df.MonthlyIncome.fillna(df.MonthlyIncome.mean(),inplace=True)
df.NumberOfChildrenVisiting.fillna(df.NumberOfChildrenVisiting.mean(),inplace=True)



df.drop('CustomerID', inplace=True, axis=1)
df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(columns=['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)


X = df.drop(['ProdTaken'], axis=1)
y = df['ProdTaken']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

## get all the numeric features
num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
print('Num of Numerical Features :', len(num_features))

cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
print('Num of Categorical Features :', len(cat_features))

cat_features = X.select_dtypes(include="object").columns
num_features = X.select_dtypes(exclude="object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
         ("OneHotEncoder", oh_transformer, cat_features),
          ("StandardScaler", numeric_transformer, num_features)
    ]
)
print(preprocessor)
X_train=preprocessor.fit_transform(X_train)
pd.DataFrame(X_train)
X_test=preprocessor.transform(X_test)
pd.DataFrame(X_test)

model=AdaBoostClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

df.isnull().sum()
