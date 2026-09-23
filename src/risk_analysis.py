def analyze_risk(vehicle_data):
    
    factors = []
    recommendations = []

    # Vehicle Age
    if vehicle_data["vehicle_age"] >= 10:
        factors.append({
            "factor": "Vehicle Age",
            "severity": "High",
            "message": "The vehicle is relatively old."
        })

        recommendations.append(
            "High Priority: Schedule a detailed inspection due to vehicle age."
        )

    elif vehicle_data["vehicle_age"] >= 7:
        factors.append({
            "factor": "Vehicle Age",
            "severity": "Medium",
            "message": "The vehicle has moderate age."
        })

        recommendations.append(
            "Monitor age-related components during regular servicing."
        )

    # Annual Mileage
    if vehicle_data["annual_mileage"] >= 30000:
        factors.append({
            "factor": "Annual Mileage",
            "severity": "High",
            "message": "Annual mileage is relatively high."
        })

        recommendations.append(
            "High Priority: Monitor service intervals closely because of high usage."
        )

    # City Driving
    if vehicle_data["city_driving_pct"] >= 75:
        factors.append({
            "factor": "City Driving",
            "severity": "Medium",
            "message": "A large percentage of driving occurs in the city."
        })

        recommendations.append(
            "Inspect brakes and engine components more frequently due to heavy city driving."
        )

    # Hard Braking
    if vehicle_data["hard_braking_events"] >= 10:
        factors.append({
            "factor": "Hard Braking",
            "severity": "High",
            "message": "Frequent hard braking events were detected."
        })

        recommendations.append(
            "High Priority: Inspect brake pads, discs and the braking system."
        )

    # Rapid Acceleration
    if vehicle_data["rapid_acceleration_events"] >= 10:
        factors.append({
            "factor": "Rapid Acceleration",
            "severity": "High",
            "message": "Frequent rapid acceleration events were detected."
        })

        recommendations.append(
            "Reduce aggressive acceleration to lower mechanical stress."
        )

    # Engine Temperature
    if vehicle_data["engine_temperature"] >= 100:
        factors.append({
            "factor": "Engine Temperature",
            "severity": "High",
            "message": "Engine temperature is elevated."
        })

        recommendations.append(
            "High Priority: Inspect coolant level, radiator and cooling system."
        )

    elif vehicle_data["engine_temperature"] >= 96:
        factors.append({
            "factor": "Engine Temperature",
            "severity": "Medium",
            "message": "Engine temperature is above the normal range."
        })

        recommendations.append(
            "Monitor engine temperature and cooling-system performance."
        )

    # Engine Vibration
    if vehicle_data["engine_vibration"] >= 3.5:
        factors.append({
            "factor": "Engine Vibration",
            "severity": "High",
            "message": "Engine vibration is relatively high."
        })

        recommendations.append(
            "High Priority: Inspect engine mounts and investigate unusual vibration."
        )

    elif vehicle_data["engine_vibration"] >= 2.8:
        factors.append({
            "factor": "Engine Vibration",
            "severity": "Medium",
            "message": "Moderate engine vibration detected."
        })

        recommendations.append(
            "Monitor vibration and inspect the engine during the next service."
        )

    # Oil Age
    if vehicle_data["oil_age_km"] >= 10000:
        factors.append({
            "factor": "Oil Age",
            "severity": "High",
            "message": "A long distance has been covered since the oil change."
        })

        recommendations.append(
            "High Priority: Replace engine oil and inspect the oil filter."
        )

    # Brake Wear
    if vehicle_data["brake_wear_pct"] >= 70:
        factors.append({
            "factor": "Brake Wear",
            "severity": "High",
            "message": "Brake wear percentage is relatively high."
        })

        recommendations.append(
            "High Priority: Inspect and replace worn brake components if necessary."
        )

    # Tire Pressure
    tire_pressure = vehicle_data["tire_pressure_avg"]

    if tire_pressure < 29 or tire_pressure > 35:
        factors.append({
            "factor": "Tire Pressure",
            "severity": "Medium",
            "message": "Average tire pressure is outside the preferred range."
        })

        recommendations.append(
            "Check and adjust tire pressure according to the manufacturer's specification."
        )

    # Battery
    if vehicle_data["battery_voltage"] < 12.2:
        factors.append({
            "factor": "Battery Voltage",
            "severity": "High",
            "message": "Battery voltage is relatively low."
        })

        recommendations.append(
            "High Priority: Check battery health and charging-system performance."
        )

    # Service Interval
    if vehicle_data["last_service_km"] >= 15000:
        factors.append({
            "factor": "Service Interval",
            "severity": "High",
            "message": "A long distance has been covered since the last service."
        })

        recommendations.append(
            "High Priority: Schedule a preventive maintenance service."
        )

    # Previous Repairs
    if vehicle_data["previous_repairs"] >= 5:
        factors.append({
            "factor": "Previous Repairs",
            "severity": "Medium",
            "message": "The vehicle has a relatively high number of previous repairs."
        })

        recommendations.append(
            "Review previous repair history for recurring mechanical problems."
        )

    # Service Frequency
    if vehicle_data["service_frequency"] == "Rare":
        factors.append({
            "factor": "Service Frequency",
            "severity": "High",
            "message": "The vehicle is serviced infrequently."
        })

        recommendations.append(
            "High Priority: Schedule a complete inspection and follow a regular preventive maintenance plan."
        )

    elif vehicle_data["service_frequency"] == "Occasional":
        factors.append({
            "factor": "Service Frequency",
            "severity": "Medium",
            "message": "Vehicle servicing is occasional."
        })

        recommendations.append(
            "Increase service consistency and avoid delaying scheduled maintenance."
        )

    # Default recommendation
    if len(recommendations) == 0:
        recommendations.append(
            "Continue regular preventive maintenance and monitor vehicle condition."
        )

    return factors, recommendations