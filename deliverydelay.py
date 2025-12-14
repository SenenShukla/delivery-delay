import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("delivery_delay (1).csv")
    return df

df = load_data()

# ----------------------------
# Train Model
# ----------------------------
X = df.drop("Delivery_Delay", axis=1)
y = df["Delivery_Delay"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚚 Delivery Delay Prediction App")
st.markdown("Predict whether a delivery will be **Delayed or On-Time**")

st.sidebar.header("📦 Delivery Details")

def user_input():
    data = {
        "Delivery_Distance": st.sidebar.slider("Delivery Distance (km)", 1.0, 50.0, 20.0),
        "Traffic_Congestion": st.sidebar.slider("Traffic Congestion (1–5)", 1, 5, 3),
        "Weather_Condition": st.sidebar.slider("Weather Condition (1–5)", 1, 5, 3),
        "Delivery_Slot": st.sidebar.slider("Delivery Slot (1–4)", 1, 4, 2),
        "Driver_Experience": st.sidebar.slider("Driver Experience (Years)", 1, 20, 5),
        "Num_Stops": st.sidebar.slider("Number of Stops", 1, 10, 4),
        "Vehicle_Age": st.sidebar.slider("Vehicle Age (Years)", 0, 15, 5),
        "Road_Condition_Score": st.sidebar.slider("Road Condition Score (1–5)", 1, 5, 3),
        "Package_Weight": st.sidebar.slider("Package Weight (kg)", 1.0, 50.0, 10.0),
        "Fuel_Efficiency": st.sidebar.slider("Fuel Efficiency (km/l)", 5.0, 25.0, 15.0),
        "Warehouse_Processing_Time": st.sidebar.slider("Warehouse Processing Time (min)", 10, 120, 40)
    }
    return pd.DataFrame([data])

input_df = user_input()

# ----------------------------
# Prediction
# ----------------------------
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# ----------------------------
# Output
# ----------------------------
st.subheader("📊 Prediction Result")

if prediction == 1:
    st.error(f"🚨 Delivery is likely to be DELAYED")
else:
    st.success(f"✅ Delivery is likely to be ON TIME")

st.metric("Probability of Delay", f"{probability*100:.2f}%")

st.subheader("📌 Input Summary")
st.write(input_df)

# ----------------------------
# Business Insight
# ----------------------------
st.subheader("💡 Business Insight")
st.write("""
- High **traffic congestion** and **warehouse processing time** significantly increase delay risk.
- Improving warehouse efficiency and route planning can reduce delays.
- Assign experienced drivers to high-risk deliveries.
""")
