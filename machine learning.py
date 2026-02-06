from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

file_path = "./cleaned_exoplanets_with_hab_augmented (1).csv"
data = pd.read_csv(file_path)

feature_cols = [c for c in data.columns if c not in ('hab', 'pl_name')]
X = data[feature_cols].copy()

if data['hab'].isna().any():
    print("Dropping rows with NaN target values in 'hab'.")
    data = data.dropna(subset=['hab'])
    X = X.loc[data.index]

y = data['hab'].astype(int)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

version = tuple(map(int, sklearn.__version__.split(".")[:2]))
if version >= (1, 2):
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', onehot)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(clf, "planet_habitability.joblib")

def debug_predict_row_from_dataset(index):
    row = X.iloc[[index]]
    true_label = y.iloc[index]
    pred = clf.predict(row)
    prob = clf.predict_proba(row)
    print(f"Row {index} true label: {true_label}, predicted: {pred[0]}, prob: {prob}")

def predict_new_planet_dict(planet_dict):
    new_df = pd.DataFrame([planet_dict])
    pred = clf.predict(new_df)
    prob = clf.predict_proba(new_df)
    print("Prediction for new planet:", "Habitable" if pred[0] == 1 else "Not Habitable")
    print("Probability:", prob)
    return pred, prob


new_planet = {
  "name": "LHS 1140 b",
  "pl_orbper": 24.737,       
  "pl_rade": 1.64,               
  "pl_bmasse": 5.6,              
  "pl_dens": 6.98,              
  "pl_eqt": 230,               
  "pl_orbsmax": 0.0957,         
  "pl_orbeccen": 0.096,    
  "st_teff": 3096,
  "st_mass": 0.18,
  "st_rad": 0.216,
  "st_lum": 0.0038,
  "sy_dist": 15.0,             
  "discoverymethod_Transit": 1
}

print("Testing New Planet")
predict_new_planet_dict(new_planet)
