import numpy as np
import pandas as pd

np.random.seed(42)

NUM_RECORDS = 15000

vehicle_age = np.random.randint(1, 16, NUM_RECORDS)

vehicle_type = np.random.choice(
    ["Sedan", "SUV", "Hatchback", "Pickup"],
    NUM_RECORDS
)

annual_mileage = np.random.randint(5000, 40001, NUM_RECORDS)

city_driving_pct = np.random.randint(20, 91, NUM_RECORDS)

highway_driving_pct = 100 - city_driving_pct

daily_driving_hours = np.round(
    np.random.uniform(0.5, 5.0, NUM_RECORDS),
    1
)

# Driving behavior
hard_braking_events = np.round(
    np.random.uniform(0, 20, NUM_RECORDS) * (city_driving_pct / 100)
).astype(int)

rapid_acceleration_events = np.round(
    np.random.uniform(0, 18, NUM_RECORDS) * (city_driving_pct / 100)
).astype(int)


# Engine and vehicle health
engine_temperature = np.clip(
    np.round(np.random.normal(92, 7, NUM_RECORDS), 1),
    75,
    120
)

engine_vibration = np.round(
    np.random.uniform(0.5, 5.0, NUM_RECORDS),
    2
)

oil_age_km = np.random.randint(
    500,
    15001,
    NUM_RECORDS
)

brake_wear_pct = np.round(
    np.random.uniform(5, 95, NUM_RECORDS),
    1
)

tire_pressure_avg = np.round(
    np.random.normal(32, 2.5, NUM_RECORDS),
    1
)

battery_voltage = np.round(
    np.random.normal(12.8, 0.6, NUM_RECORDS),
    2
)

# Maintenance history
last_service_km = np.random.randint(
    500,
    30001,
    NUM_RECORDS
)

previous_repairs = np.random.randint(
    0,
    9,
    NUM_RECORDS
)

service_frequency = np.random.choice(
    ["Regular", "Occasional", "Rare"],
    NUM_RECORDS,
    p=[0.55, 0.30, 0.15]
)

# Calculate maintenance risk score

risk_score = (
    vehicle_age * 2
    + annual_mileage / 5000
    + city_driving_pct / 20
    + daily_driving_hours * 2
    + hard_braking_events * 0.8
    + rapid_acceleration_events * 0.7
    + np.maximum(engine_temperature - 90, 0) * 1.5
    + engine_vibration * 5
    + oil_age_km / 2000
    + brake_wear_pct * 0.15
    + np.abs(tire_pressure_avg - 32) * 4
    + np.maximum(13 - battery_voltage, 0) * 10
    + last_service_km / 3000
    + previous_repairs * 2
)

# Convert risk score into categories

low_threshold = np.percentile(risk_score, 40)
high_threshold = np.percentile(risk_score, 75)

maintenance_risk = np.where(
    risk_score < low_threshold,
    "Low",
    np.where(
        risk_score < high_threshold,
        "Medium",
        "High"
    )
)

# Create the dataset

df = pd.DataFrame({
    "vehicle_age": vehicle_age,
    "vehicle_type": vehicle_type,
    "annual_mileage": annual_mileage,
    "city_driving_pct": city_driving_pct,
    "highway_driving_pct": highway_driving_pct,
    "daily_driving_hours": daily_driving_hours,
    "hard_braking_events": hard_braking_events,
    "rapid_acceleration_events": rapid_acceleration_events,
    "engine_temperature": engine_temperature,
    "engine_vibration": engine_vibration,
    "oil_age_km": oil_age_km,
    "brake_wear_pct": brake_wear_pct,
    "tire_pressure_avg": tire_pressure_avg,
    "battery_voltage": battery_voltage,
    "last_service_km": last_service_km,
    "previous_repairs": previous_repairs,
    "service_frequency": service_frequency,
    "maintenance_risk": maintenance_risk
})


# Inspect the dataset

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nMaintenance risk distribution:")
print(df["maintenance_risk"].value_counts())

# Save dataset
df.to_csv("Data/vehicle_data.csv", index=False)

print("\nDataset saved successfully!")