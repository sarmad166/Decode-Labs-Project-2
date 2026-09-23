import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.prediction import predict_vehicle_risk
from src.risk_analysis import analyze_risk


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AutoGuard AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #8b949e;
        margin-bottom: 30px;
    }

    .risk-high {
        padding: 25px;
        border-radius: 15px;
        background: rgba(255, 75, 75, 0.12);
        border: 1px solid #ff4b4b;
        text-align: center;
    }

    .risk-medium {
        padding: 25px;
        border-radius: 15px;
        background: rgba(255, 193, 7, 0.12);
        border: 1px solid #ffc107;
        text-align: center;
    }

    .risk-low {
        padding: 25px;
        border-radius: 15px;
        background: rgba(40, 167, 69, 0.12);
        border: 1px solid #28a745;
        text-align: center;
    }

    .factor-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #30363d;
        background: #161b22;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "Data/vehicle_data.csv"
)


# =========================================================
# PREPARE ML DATA
# =========================================================

X = df.drop(
    "maintenance_risk",
    axis=1
)

y = df["maintenance_risk"]


X = pd.get_dummies(
    X,
    columns=[
        "vehicle_type",
        "service_frequency"
    ],
    dtype=int
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# K VALUE TUNING
# 5-FOLD CROSS VALIDATION
# =========================================================

k_values = range(
    1,
    22,
    2
)

k_results = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    cv_scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    mean_accuracy = cv_scores.mean()

    k_results.append(
        mean_accuracy
    )


# =========================================================
# BEST K
# =========================================================

best_index = k_results.index(
    max(k_results)
)

best_k = list(k_values)[best_index]

best_cv_accuracy = k_results[best_index]


# =========================================================
# FINAL MODEL
# =========================================================

final_model = KNeighborsClassifier(
    n_neighbors=best_k
)

final_model.fit(
    X_train_scaled,
    y_train
)

final_prediction = final_model.predict(
    X_test_scaled
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🚗 AutoGuard AI"
    )

    st.caption(
        "Intelligent Vehicle Maintenance "
        "Risk Classification System"
    )

    st.divider()

    st.markdown(
        "### Navigation"
    )

    page = st.radio(
        "Go to",
        [
            "Dashboard",
            "Vehicle Prediction",
            "Data Insights",
            "Model Performance"
        ]
    )

    st.divider()

    st.markdown(
        "### 🤖 Machine Learning"
    )

    st.write(
        "Algorithm: **KNN Classifier**"
    )

    st.write(
        f"Optimal K: **{best_k}**"
    )

    st.write(
        "Dataset: **15,000 vehicles**"
    )

    st.write(
        "Risk Classes: **3**"
    )

    st.divider()

    st.caption(
        "Synthetic domain-inspired vehicle dataset "
        "created for this machine learning project."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚗 AutoGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Vehicle Maintenance Risk Classification System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header(
        "📊 Vehicle Risk Dashboard"
    )

    st.write(
        "Overview of the vehicle maintenance risk dataset."
    )

    st.divider()


    # -----------------------------------------------------
    # SUMMARY METRICS
    # -----------------------------------------------------

    total_vehicles = len(df)

    high_risk = (
        df["maintenance_risk"] == "High"
    ).sum()

    medium_risk = (
        df["maintenance_risk"] == "Medium"
    ).sum()

    low_risk = (
        df["maintenance_risk"] == "Low"
    ).sum()


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Vehicles",
            f"{total_vehicles:,}"
        )


    with col2:

        st.metric(
            "🔴 High Risk",
            f"{high_risk:,}"
        )


    with col3:

        st.metric(
            "🟡 Medium Risk",
            f"{medium_risk:,}"
        )


    with col4:

        st.metric(
            "🟢 Low Risk",
            f"{low_risk:,}"
        )


    st.divider()


    # -----------------------------------------------------
    # CHARTS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        risk_counts = (
            df["maintenance_risk"]
            .value_counts()
            .reindex(
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )
            .reset_index()
        )

        risk_counts.columns = [
            "Risk",
            "Vehicles"
        ]


        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Vehicles",
            text="Vehicles",
            title="Maintenance Risk Distribution"
        )

        fig.update_layout(
            xaxis_title="Risk Level",
            yaxis_title="Number of Vehicles"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        sample_df = df.sample(
            min(
                3000,
                len(df)
            ),
            random_state=42
        )


        fig = px.scatter(
            sample_df,
            x="annual_mileage",
            y="engine_vibration",
            color="maintenance_risk",
            hover_data=[
                "vehicle_age",
                "vehicle_type",
                "oil_age_km"
            ],
            title="Mileage vs Engine Vibration"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.divider()


    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    st.subheader(
        "🚘 Dataset Overview"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Average Vehicle Age",
            f"{df['vehicle_age'].mean():.1f} years"
        )


    with col2:

        st.metric(
            "Average Annual Mileage",
            f"{df['annual_mileage'].mean():,.0f} km"
        )


    with col3:

        st.metric(
            "Average Engine Vibration",
            f"{df['engine_vibration'].mean():.2f}"
        )


# =========================================================
# VEHICLE PREDICTION
# =========================================================

elif page == "Vehicle Prediction":

    st.header(
        "🔍 Analyze Your Vehicle"
    )

    st.write(
        "Enter vehicle usage, driving behavior, "
        "vehicle health and maintenance information."
    )

    st.divider()


    # =====================================================
    # VEHICLE INFORMATION
    # =====================================================

    st.subheader(
        "🚘 Vehicle Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        vehicle_age = st.number_input(
            "Vehicle Age (Years)",
            1,
            15,
            5
        )


        vehicle_type = st.selectbox(
            "Vehicle Type",
            [
                "Sedan",
                "SUV",
                "Hatchback",
                "Pickup"
            ]
        )


        annual_mileage = st.number_input(
            "Annual Mileage (km)",
            5000,
            40000,
            15000,
            step=1000
        )


    with col2:

        city_driving_pct = st.slider(
            "City Driving (%)",
            20,
            90,
            50
        )


        st.caption(
            f"Highway Driving: "
            f"{100 - city_driving_pct}%"
        )


        daily_driving_hours = st.number_input(
            "Daily Driving Hours",
            0.5,
            5.0,
            2.0,
            step=0.5
        )


        hard_braking_events = st.number_input(
            "Hard Braking Events",
            0,
            20,
            5
        )


    with col3:

        rapid_acceleration_events = st.number_input(
            "Rapid Acceleration Events",
            0,
            18,
            5
        )


        engine_temperature = st.number_input(
            "Engine Temperature (°C)",
            75.0,
            120.0,
            92.0,
            step=0.5
        )


        engine_vibration = st.number_input(
            "Engine Vibration",
            0.5,
            5.0,
            2.0,
            step=0.1
        )


    # =====================================================
    # VEHICLE HEALTH
    # =====================================================

    st.subheader(
        "🔧 Vehicle Health"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        oil_age_km = st.number_input(
            "Oil Age (km)",
            500,
            15000,
            5000,
            step=500
        )


        brake_wear_pct = st.slider(
            "Brake Wear (%)",
            5.0,
            95.0,
            30.0
        )


    with col2:

        tire_pressure_avg = st.number_input(
            "Average Tire Pressure",
            20.0,
            40.0,
            32.0,
            step=0.1
        )


        battery_voltage = st.number_input(
            "Battery Voltage",
            10.0,
            14.5,
            12.8,
            step=0.1
        )


    with col3:

        last_service_km = st.number_input(
            "Distance Since Last Service (km)",
            500,
            30000,
            5000,
            step=500
        )


        previous_repairs = st.number_input(
            "Previous Repairs",
            0,
            8,
            1
        )


    service_frequency = st.selectbox(
        "Service Frequency",
        [
            "Regular",
            "Occasional",
            "Rare"
        ]
    )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    st.divider()


    if st.button(
        "🚀 Analyze Vehicle Risk",
        use_container_width=True,
        type="primary"
    ):

        # -------------------------------------------------
        # HIGHWAY DRIVING
        # -------------------------------------------------

        highway_driving_pct = (
            100 - city_driving_pct
        )


        # -------------------------------------------------
        # VEHICLE DATA
        # -------------------------------------------------

        vehicle_data = {

            "vehicle_age":
                vehicle_age,

            "vehicle_type":
                vehicle_type,

            "annual_mileage":
                annual_mileage,

            "city_driving_pct":
                city_driving_pct,

            "highway_driving_pct":
                highway_driving_pct,

            "daily_driving_hours":
                daily_driving_hours,

            "hard_braking_events":
                hard_braking_events,

            "rapid_acceleration_events":
                rapid_acceleration_events,

            "engine_temperature":
                engine_temperature,

            "engine_vibration":
                engine_vibration,

            "oil_age_km":
                oil_age_km,

            "brake_wear_pct":
                brake_wear_pct,

            "tire_pressure_avg":
                tire_pressure_avg,

            "battery_voltage":
                battery_voltage,

            "last_service_km":
                last_service_km,

            "previous_repairs":
                previous_repairs,

            "service_frequency":
                service_frequency
        }


        # -------------------------------------------------
        # ML PREDICTION
        # -------------------------------------------------

        prediction, confidence = (
            predict_vehicle_risk(
                vehicle_data
            )
        )


        # -------------------------------------------------
        # RISK ANALYSIS
        # -------------------------------------------------

        factors, recommendations = (
            analyze_risk(
                vehicle_data
            )
        )


        # =================================================
        # PREDICTION RESULT
        # =================================================

        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )


        # -------------------------------------------------
        # RISK RESULT
        # -------------------------------------------------

        if prediction == "High":

            st.markdown(
                """
                <div class="risk-high">
                    <h2>🔴 HIGH MAINTENANCE RISK</h2>
                    <p>
                    The KNN model classified this
                    vehicle as High Risk.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


        elif prediction == "Medium":

            st.markdown(
                """
                <div class="risk-medium">
                    <h2>🟡 MEDIUM MAINTENANCE RISK</h2>
                    <p>
                    The KNN model classified this
                    vehicle as Medium Risk.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                """
                <div class="risk-low">
                    <h2>🟢 LOW MAINTENANCE RISK</h2>
                    <p>
                    The KNN model classified this
                    vehicle as Low Risk.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


        st.write("")


        # =================================================
        # MODEL CONFIDENCE
        # =================================================

        st.subheader(
            "🤖 Model Confidence"
        )


        conf_col1, conf_col2, conf_col3 = (
            st.columns(3)
        )


        for col, risk in zip(
            [
                conf_col1,
                conf_col2,
                conf_col3
            ],
            [
                "Low",
                "Medium",
                "High"
            ]
        ):

            with col:

                probability = confidence.get(
                    risk,
                    0
                )


                st.metric(
                    risk,
                    f"{probability * 100:.2f}%"
                )


                st.progress(
                    float(probability)
                )


        st.divider()


        # =================================================
        # SMART RISK ANALYSIS
        # =================================================

        st.subheader(
            "🧠 Smart Risk Analysis"
        )


        high_factors = sum(
            1
            for factor in factors
            if factor["severity"] == "High"
        )


        medium_factors = sum(
            1
            for factor in factors
            if factor["severity"] == "Medium"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Risk Level",
                prediction
            )


        with col2:

            st.metric(
                "🔴 High-Risk Factors",
                high_factors
            )


        with col3:

            st.metric(
                "🟡 Medium-Risk Factors",
                medium_factors
            )


        st.markdown(
            "### ⚠️ Detected Risk Factors"
        )


        if factors:

            st.write(
                f"The system identified "
                f"**{len(factors)} potential risk indicators**."
            )


            for factor in factors:

                if factor["severity"] == "High":

                    st.error(
                        f"🔴 **{factor['factor']}** — "
                        f"{factor['message']}"
                    )

                else:

                    st.warning(
                        f"🟡 **{factor['factor']}** — "
                        f"{factor['message']}"
                    )

        else:

            st.success(
                "🟢 No major risk indicators were "
                "detected from the entered vehicle information."
            )


        # =================================================
        # MAINTENANCE RECOMMENDATIONS
        # =================================================

        st.subheader(
            "🔧 Maintenance Recommendations"
        )


        if recommendations:

            for recommendation in recommendations:

                st.info(
                    f"💡 {recommendation}"
                )

        else:

            st.success(
                "Continue following a regular "
                "preventive maintenance schedule."
            )


        # =================================================
        # VEHICLE ANALYSIS SUMMARY
        # =================================================

        st.subheader(
            "📋 Vehicle Analysis Summary"
        )


        summary = pd.DataFrame(
            {
                "Parameter": [

                    "Vehicle Age",

                    "Vehicle Type",

                    "Annual Mileage",

                    "City Driving",

                    "Highway Driving",

                    "Daily Driving",

                    "Hard Braking",

                    "Rapid Acceleration",

                    "Engine Temperature",

                    "Engine Vibration",

                    "Oil Age",

                    "Brake Wear",

                    "Tire Pressure",

                    "Battery Voltage",

                    "Distance Since Service",

                    "Previous Repairs",

                    "Service Frequency"
                ],

                "Value": [

                    f"{vehicle_age} years",

                    vehicle_type,

                    f"{annual_mileage:,} km",

                    f"{city_driving_pct}%",

                    f"{highway_driving_pct}%",

                    f"{daily_driving_hours} hours/day",

                    str(hard_braking_events),

                    str(rapid_acceleration_events),

                    f"{engine_temperature} °C",

                    str(engine_vibration),

                    f"{oil_age_km:,} km",

                    f"{brake_wear_pct}%",

                    f"{tire_pressure_avg}",

                    f"{battery_voltage} V",

                    f"{last_service_km:,} km",

                    str(previous_repairs),

                    service_frequency
                ]
            }
        )


        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# DATA INSIGHTS
# =========================================================

elif page == "Data Insights":

    st.header(
        "📈 Data Insights"
    )

    st.write(
        "Explore relationships between vehicle "
        "characteristics and maintenance risk."
    )

    st.divider()


    # -----------------------------------------------------
    # VEHICLE AGE
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="vehicle_age",
            title="Vehicle Age vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # ANNUAL MILEAGE
    # -----------------------------------------------------

    with col2:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="annual_mileage",
            title="Annual Mileage vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # HARD BRAKING
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="hard_braking_events",
            title="Hard Braking vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # RAPID ACCELERATION
    # -----------------------------------------------------

    with col2:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="rapid_acceleration_events",
            title="Rapid Acceleration vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # ENGINE VIBRATION
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="engine_vibration",
            title="Engine Vibration vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # ENGINE TEMPERATURE
    # -----------------------------------------------------

    with col2:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="engine_temperature",
            title="Engine Temperature vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # OIL AGE
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="oil_age_km",
            title="Oil Age vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # BRAKE WEAR
    # -----------------------------------------------------

    with col2:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="brake_wear_pct",
            title="Brake Wear vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # CITY DRIVING
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="city_driving_pct",
            title="City Driving vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # PREVIOUS REPAIRS
    # -----------------------------------------------------

    with col2:

        fig = px.box(
            df,
            x="maintenance_risk",
            y="previous_repairs",
            title="Previous Repairs vs Maintenance Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.divider()


    # -----------------------------------------------------
    # DATASET PREVIEW
    # -----------------------------------------------------

    st.subheader(
        "📋 Dataset Preview"
    )


    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header(
        "🤖 Model Performance"
    )

    st.write(
        "Evaluation of the KNN classification model "
        "using a held-out test dataset and "
        "5-fold cross-validation."
    )

    st.divider()


    # =====================================================
    # PERFORMANCE METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        final_prediction
    )


    precision = precision_score(
        y_test,
        final_prediction,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        final_prediction,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        final_prediction,
        average="weighted",
        zero_division=0
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Test Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )


    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )


    st.divider()


    # =====================================================
    # CROSS-VALIDATION K TUNING
    # =====================================================

    st.subheader(
        "📊 K Value Tuning"
    )


    k_df = pd.DataFrame(
        {
            "K": list(k_values),

            "CV Accuracy": [
                value * 100
                for value in k_results
            ]
        }
    )


    fig = px.line(
        k_df,
        x="K",
        y="CV Accuracy",
        markers=True,
        title="5-Fold Cross-Validation Accuracy for Different K Values"
    )


    fig.add_vline(
        x=best_k,
        line_dash="dash",
        annotation_text=f"Best K = {best_k}"
    )


    fig.update_layout(
        xaxis_title="Number of Neighbors (K)",
        yaxis_title="Cross-Validation Accuracy (%)"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.success(
        f"Optimal K selected using 5-fold cross-validation: "
        f"**K = {best_k}** with mean CV accuracy of "
        f"**{best_cv_accuracy * 100:.2f}%**."
    )


    st.divider()


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(
        "🎯 Confusion Matrix"
    )


    cm = confusion_matrix(
        y_test,
        final_prediction,
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )


    fig = px.imshow(
        cm,
        text_auto=True,
        x=[
            "Low",
            "Medium",
            "High"
        ],
        y=[
            "Low",
            "Medium",
            "High"
        ],
        labels={
            "x": "Predicted Risk",
            "y": "Actual Risk"
        },
        title="KNN Confusion Matrix"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.divider()


    # =====================================================
    # MODEL CONFIGURATION
    # =====================================================

    st.subheader(
        "⚙️ Model Configuration"
    )


    model_info = pd.DataFrame(
        {
            "Parameter": [

                "Algorithm",

                "Optimal K",

                "Cross-Validation",

                "Training Records",

                "Testing Records",

                "Input Features",

                "Feature Scaling",

                "Classification Type"
            ],

            "Value": [

                "K-Nearest Neighbors",

                str(best_k),

                "5-Fold Cross-Validation",

                f"{len(X_train):,}",

                f"{len(X_test):,}",

                str(X.shape[1]),

                "StandardScaler",

                "Multi-Class"
            ]
        }
    )


    st.table(
        model_info
    )


    st.divider()


    # =====================================================
    # DATASET DISCLAIMER
    # =====================================================

    st.info(
        "The dataset used in this project is synthetic "
        "and domain-inspired. Model performance should "
        "not be interpreted as real-world vehicle failure "
        "prediction accuracy."
    )