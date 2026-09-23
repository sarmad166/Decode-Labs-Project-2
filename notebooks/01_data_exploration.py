import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/vehicle_data.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nMaintenance Risk Distribution:")
print(df["maintenance_risk"].value_counts())

sns.countplot(data=df, x="maintenance_risk")

plt.title("Maintenance Risk Distribution")
plt.xlabel("Maintenance Risk")
plt.ylabel("Number of Vehicles")
plt.show()
print("\nStatistical Summary:")
print(df.describe())

print("\nVehicle Type Distribution:")
print(df["vehicle_type"].value_counts())

print("\nService Frequency Distribution:")
print(df["service_frequency"].value_counts())

print("\nAverage values by Maintenance Risk:")
print(
    df.groupby("maintenance_risk")[
        [
            "vehicle_age",
            "annual_mileage",
            "engine_temperature",
            "engine_vibration",
            "oil_age_km",
            "brake_wear_pct",
            "previous_repairs"
        ]
    ].mean().round(2)
)

# Risk vs Vehicle Age

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="maintenance_risk",
    y="vehicle_age"
)

plt.title("Vehicle Age vs Maintenance Risk")
plt.xlabel("Maintenance Risk")
plt.ylabel("Vehicle Age (Years)")

plt.show()

# Driving Behavior vs Maintenance Risk

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="maintenance_risk",
    y="hard_braking_events"
)

plt.title("Hard Braking Events vs Maintenance Risk")
plt.xlabel("Maintenance Risk")
plt.ylabel("Hard Braking Events")

plt.show()

# Engine Vibration vs Maintenance Risk

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="maintenance_risk",
    y="engine_vibration"
)

plt.title("Engine Vibration vs Maintenance Risk")
plt.xlabel("Maintenance Risk")
plt.ylabel("Engine Vibration")

plt.show()

# Correlation Heatmap

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.show()

# Annual Mileage vs Maintenance Risk

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="maintenance_risk",
    y="annual_mileage"
)

plt.title("Annual Mileage vs Maintenance Risk")
plt.xlabel("Maintenance Risk")
plt.ylabel("Annual Mileage (km)")

plt.tight_layout()
plt.show()