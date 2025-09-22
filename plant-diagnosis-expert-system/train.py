import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import joblib, os

DATA_PATH = os.path.join("data", "diseases.csv")
OUT_PATH = os.path.join("backend", "model.pkl")

df = pd.read_csv(DATA_PATH)
X = df[["leaf_color","spots","wilt"]]
y = df["disease"]

# ordinal encode categorical features
encoder = OrdinalEncoder()
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
model = Pipeline([("enc", encoder), ("clf", clf)])
model.fit(X, y)
joblib.dump(model, OUT_PATH)
print("Model trained and saved to", OUT_PATH)
