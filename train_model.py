import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset/employee_attrition.csv")

# Select only 10 features
features = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "YearsAtCompany",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "OverTime",
    "JobLevel",
    "JobInvolvement"
]

X = df[features]

y = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})

# Ordered categories
categories = [
    ["Low", "Medium", "High", "Very High"],                    # JobSatisfaction
    ["Low", "Medium", "High", "Very High"],                    # EnvironmentSatisfaction
    ["Bad", "Better", "Good", "Best"],                         # WorkLifeBalance
    ["No", "Yes"],                                             # OverTime
    ["Entry Level","Junior Level","Mid Level","Senior Level","Executive Level"],   # JobLevel
    ["Low","Medium","High","Very High"]                        # JobInvolvement
]

categorical = [
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "OverTime",
    "JobLevel",
    "JobInvolvement"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(categories=categories),
            categorical
        )
    ],
    remainder="passthrough"
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Preprocess training and test data
X_train_processed = pipeline.named_steps["preprocessor"].fit_transform(X_train)
X_test_processed = pipeline.named_steps["preprocessor"].transform(X_test)

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed,
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# Train only the Random Forest model
pipeline.named_steps["model"].fit(
    X_train_balanced,
    y_train_balanced
)

# Predict
y_pred = pipeline.named_steps["model"].predict(X_test_processed)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save updated pipeline
joblib.dump(pipeline, "model/employee_attrition_model.pkl")

print("\nModel saved successfully!")



