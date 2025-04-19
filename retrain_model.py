
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import seaborn as sns

# Load datasets
clinical = pd.read_excel("cancer_data/Clinical.xlsx")
family_history = pd.read_excel("cancer_data/family_history.xlsx")
follow_up = pd.read_excel("cancer_data/follow_up.xlsx")
pathology_detail = pd.read_excel("cancer_data/pathology_detail.xlsx")

# Merge datasets
for df in [family_history, follow_up, pathology_detail]:
    df.drop(columns=["case_submitter_id", "project_id"], errors="ignore", inplace=True)
df = clinical.merge(family_history, on="case_id", how="left")
df = df.merge(follow_up, on="case_id", how="left")
df = df.merge(pathology_detail, on="case_id", how="left")

# Preprocess
df['survival_time'] = df.apply(lambda row: row['days_to_death'] if row['vital_status'] == "Dead"
                               else row['days_to_last_follow_up'], axis=1)
df['survival_time'] = df['survival_time'].fillna(df['survival_time'].median()) / 365
df["age_at_diagnosis"] = df["age_at_diagnosis"] / 365
df = df[df['protocol_identifier'] != 'Not Reported']
df.rename(columns={"relative_with_cancer_history": "family_cancer_history"}, inplace=True)

tumor_grade_mapping = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "High Grade": 4, "GX": -1, "GB": -1, "Not Reported": -1}
df["tumor_grade"] = df["tumor_grade"].map(tumor_grade_mapping).fillna(2)

# Binarize outcome for classification: long-survivor vs. short-survivor
df = df.dropna(subset=["survival_time"])
df['long_survivor'] = (df['survival_time'] > 3).astype(int)

# Select features
numeric_cols = ["age_at_diagnosis", "tumor_grade", "family_cancer_history", "survival_time", "tumor_largest_dimension_diameter"]
categorical_cols = ['ajcc_pathologic_stage', 'prior_treatment']
features = numeric_cols + categorical_cols
available_features = [col for col in features if col in df.columns]
df = df[available_features + ["long_survivor"]]

# Imputation
numeric_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().sum() > 0]
imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Encode categorical features
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
for col in existing_cat_cols:
    df[col] = df[col].fillna('Unknown')
df = pd.get_dummies(df, columns=existing_cat_cols)

# Separate features and target
X = df.drop(columns=["long_survivor"])
y = df["long_survivor"]
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Classification
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Feature importance plot
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features_sorted = X.columns[indices]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features_sorted, rotation=90)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Clustering
X_cluster = df[numeric_cols].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster)
silhouette = silhouette_score(X_cluster, clusters)
print(f"✅ Clustering performed on {len(X_cluster)} records. Silhouette Score: {silhouette:.3f}")

# PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_cluster)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette="viridis")
plt.title("KMeans Clustering (PCA-reduced data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("kmeans_clusters.png")
plt.close()
