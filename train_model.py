import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load datasets
clinical = pd.read_excel("cancer_data/Clinical.xlsx")
family_history = pd.read_excel("cancer_data/family_history.xlsx")
follow_up = pd.read_excel("cancer_data/follow_up.xlsx")
pathology_detail = pd.read_excel("cancer_data/pathology_detail.xlsx")

# Remove duplicate identifier columns from non-primary datasets
for df in [family_history, follow_up, pathology_detail]:
    df.drop(columns=["case_submitter_id", "project_id"], errors="ignore", inplace=True)

# Merge datasets by `case_id`
df = clinical.merge(family_history, on="case_id", how="left")
df = df.merge(follow_up, on="case_id", how="left")
df = df.merge(pathology_detail, on="case_id", how="left")

# Calculate survival_time in years
df['survival_time'] = df.apply(
    lambda row: row['days_to_death'] if row['vital_status'] == "Dead" else row['days_to_last_follow_up'], axis=1
)
df['survival_time'] = df['survival_time'].fillna(df['survival_time'].median())
df["survival_time"] = df["survival_time"] / 365

# Convert age to years
df["age_at_diagnosis"] = df["age_at_diagnosis"] / 365
df = df[df['protocol_identifier'] != 'Not Reported']

# Convert tumor_grade to numeric
tumor_grade_mapping = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "High Grade": 4, "GX": -1, "GB": -1, "Not Reported": -1}
df["tumor_grade"] = df["tumor_grade"].map(tumor_grade_mapping).fillna(2)

# Rename family history column
df.rename(columns={"relative_with_cancer_history": "family_cancer_history"}, inplace=True)

# Drop rows where target variable is missing
df = df.dropna(subset=["protocol_identifier"])

# Encode the target variable
df['protocol_identifier'] = df['protocol_identifier'].astype(str)
label_encoder = LabelEncoder()
df['protocol_identifier_encoded'] = label_encoder.fit_transform(df['protocol_identifier'])

# Select features
features = [
    "age_at_diagnosis", "tumor_grade", "family_cancer_history", "survival_time",
    "tumor_largest_dimension_diameter", "ajcc_clinical_stage", "metastasis_at_diagnosis", "prior_treatment"
]
df = df[features + ["protocol_identifier_encoded"]]

# Fill numeric values with median
numeric_cols = ["age_at_diagnosis", "tumor_grade", "family_cancer_history", "survival_time", "tumor_largest_dimension_diameter"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical values with 'Unknown'
df[['ajcc_clinical_stage', 'metastasis_at_diagnosis', 'prior_treatment']] = df[['ajcc_clinical_stage', 'metastasis_at_diagnosis', 'prior_treatment']].fillna('Unknown')

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['ajcc_clinical_stage', 'metastasis_at_diagnosis', 'prior_treatment'], drop_first=False)

# Separate features and target
X = df.drop(columns=["protocol_identifier_encoded"])
y = df["protocol_identifier_encoded"]

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Save feature names
feature_names = X.columns.tolist()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model and preprocessing tools
joblib.dump(clf, "model/cancer_treatment_protocol_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")

print("Model training complete. Files saved in 'model/' directory.")

