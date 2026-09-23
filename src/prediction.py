import joblib
import pandas as pd


# Load saved files
model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


def predict_vehicle_risk(vehicle_data):
    """
    Predict maintenance risk for a single vehicle.
    """

    # Convert input dictionary into DataFrame
    input_df = pd.DataFrame([vehicle_data])

    # Encode categorical features
    input_df = pd.get_dummies(
        input_df,
        columns=["vehicle_type", "service_frequency"],
        dtype=int
    )

    # Make sure input has exactly the same columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(input_scaled)[0]

    # Get class names
    classes = model.classes_

    confidence = dict(
        zip(classes, probabilities)
    )

    return prediction, confidence