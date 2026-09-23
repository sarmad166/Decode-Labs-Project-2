import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("Data/vehicle_data.csv")

# Separate features and target
X = df.drop("maintenance_risk", axis=1)
y = df["maintenance_risk"]

# Encode categorical columns
X = pd.get_dummies(
    X,
    columns=["vehicle_type", "service_frequency"],
    dtype=int
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Find Best K
# -----------------------------

k_values = range(1, 22, 2)

accuracies = []

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)

# Best K
best_index = accuracies.index(max(accuracies))

best_k = list(k_values)[best_index]

print("Best K:", best_k)
print(
    "Best Accuracy:",
    round(accuracies[best_index] * 100, 2),
    "%"
)

# -----------------------------
# Final KNN Model
# -----------------------------

knn = KNeighborsClassifier(
    n_neighbors=best_k
)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

# -----------------------------
# Evaluation
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nFinal Model Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"]
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")

plt.tight_layout()
plt.show()