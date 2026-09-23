from prediction import predict_vehicle_risk


# Example vehicle
vehicle = {
    "vehicle_age": 8,
    "vehicle_type": "SUV",
    "annual_mileage": 30000,
    "city_driving_pct": 70,
    "highway_driving_pct": 30,
    "daily_driving_hours": 3.5,
    "hard_braking_events": 12,
    "rapid_acceleration_events": 10,
    "engine_temperature": 100,
    "engine_vibration": 3.8,
    "oil_age_km": 9000,
    "brake_wear_pct": 70,
    "tire_pressure_avg": 29.5,
    "battery_voltage": 12.1,
    "last_service_km": 15000,
    "previous_repairs": 5,
    "service_frequency": "Occasional"
}


# Predict
prediction, confidence = predict_vehicle_risk(vehicle)


print("\n==============================")
print("   AUTOGUARD AI PREDICTION")
print("==============================")

print("\nMaintenance Risk:", prediction)

print("\nConfidence:")
for risk, probability in confidence.items():
    print(
        f"{risk}: {probability * 100:.2f}%"
    )

print("\nPrediction completed successfully!")