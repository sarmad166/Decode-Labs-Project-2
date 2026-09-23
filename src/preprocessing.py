import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Data/vehicle_data.csv")

print("Original dataset shape:", df.shape)

# Separate features and target
X = df.drop("maintenance_risk", axis=1)
y = df["maintenance_risk"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["vehicle_type", "service_frequency"],
    dtype=int
)

print("\nEncoded dataset shape:", X.shape)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nBefore Scaling:")
print(X_train.head())

# Create scaler
scaler = StandardScaler()

# Fit scaler ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply same scaler to test data
X_test_scaled = scaler.transform(X_test)

print("\nAfter Scaling:")
print(X_train_scaled[:5])

print("\nTraining data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

print("\nFeature scaling completed successfully!")