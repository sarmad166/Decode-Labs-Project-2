import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("Data/vehicle_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# -------------------------------------------------
# FEATURES AND TARGET
# -------------------------------------------------

X = df.drop("maintenance_risk", axis=1)
y = df["maintenance_risk"]


# -------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# -------------------------------------------------

X = pd.get_dummies(
    X,
    columns=[
        "vehicle_type",
        "service_frequency"
    ],
    dtype=int
)


# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -------------------------------------------------
# FEATURE SCALING
# -------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------
# FIND BEST K USING 5-FOLD CROSS VALIDATION
# -------------------------------------------------

k_values = range(1, 22, 2)

cv_results = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    mean_accuracy = scores.mean()

    cv_results.append(mean_accuracy)

    print(
        f"K = {k} | "
        f"CV Accuracy = {mean_accuracy * 100:.2f}%"
    )


# -------------------------------------------------
# SELECT BEST K
# -------------------------------------------------

best_index = cv_results.index(
    max(cv_results)
)

best_k = list(k_values)[best_index]

best_cv_accuracy = cv_results[best_index]

print("\nBest K:", best_k)

print(
    "Best Cross-Validation Accuracy:",
    round(best_cv_accuracy * 100, 2),
    "%"
)


# -------------------------------------------------
# TRAIN FINAL MODEL
# -------------------------------------------------

knn = KNeighborsClassifier(
    n_neighbors=best_k
)

knn.fit(
    X_train_scaled,
    y_train
)


# -------------------------------------------------
# CREATE MODELS DIRECTORY
# -------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

joblib.dump(
    knn,
    "models/knn_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)


# -------------------------------------------------
# SAVE MODEL INFORMATION
# -------------------------------------------------

model_info = {
    "algorithm": "K-Nearest Neighbors",
    "best_k": best_k,
    "cv_folds": 5,
    "cv_accuracy": best_cv_accuracy,
    "dataset_size": len(df),
    "test_size": 0.20
}

joblib.dump(
    model_info,
    "models/model_info.pkl"
)


print("\n================================")
print("MODEL SAVING COMPLETED")
print("================================")

print("KNN model saved successfully!")
print("Scaler saved successfully!")
print("Feature columns saved successfully!")
print("Model information saved successfully!")