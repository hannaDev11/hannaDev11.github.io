
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# Select relevant features
df["age_at_diagnosis"] = df["age_at_diagnosis"] / 365
df["survival_time"] = df.apply(lambda row: row['days_to_death'] if row['vital_status'] == "Dead"
                               else row['days_to_last_follow_up'], axis=1)
df["survival_time"] = df["survival_time"] / 365
df.rename(columns={"relative_with_cancer_history": "family_cancer_history"}, inplace=True)
tumor_grade_mapping = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "High Grade": 4, "GX": np.nan, "GB": np.nan, "Not Reported": np.nan}
df["tumor_grade"] = df["tumor_grade"].map(tumor_grade_mapping)

# Features for clustering
cluster_features = [
    "age_at_diagnosis",
    "tumor_grade",
    "family_cancer_history",
    "tumor_largest_dimension_diameter",
    "survival_time",
    "ajcc_pathologic_stage",
    "metastasis_at_diagnosis",
    "prior_treatment"
]

df = df[cluster_features]
df = df.dropna(how="all", axis=1)

# Fill missing numerics and encode categoricals
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")
df = pd.get_dummies(df, columns=categorical_cols)

# Normalize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df["cluster"] = clusters

# Silhouette Score
score = silhouette_score(scaled_data, clusters)
print(f"📊 Silhouette Score: {score:.3f}")

# PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette="Set2")
plt.title("KMeans Clustering of Patient Data (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("pca_patient_clusters.png")
plt.close()
