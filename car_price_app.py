# car_price_app.py
# Streamlit app: Car Price Prediction — upload/generate dataset, train RandomForest, visualize, predict

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# -----------------------
# Config & paths
# -----------------------
MODEL_DIR = "models"
MODEL_LATEST = os.path.join(MODEL_DIR, "car_price_model_latest.joblib")

st.title("🚗 Car Price Prediction App — Train, Visualize & Predict")
st.markdown(
    "Upload your car dataset or use the built-in sample data. Train a RandomForest model, "
    "inspect metrics and visualizations, then predict prices for new cars."
)

# -----------------------
# Sample dataset generator
# -----------------------
def generate_car_sample(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)

    makes = ["Toyota", "Honda", "Hyundai", "Ford", "BMW", "Mercedes", "Kia", "Mahindra"]
    models = {
        "Toyota": ["Corolla", "Camry", "Innova"],
        "Honda": ["Civic", "City", "CRV"],
        "Hyundai": ["i20", "Verna", "Creta"],
        "Ford": ["Figo", "Ecosport"],
        "BMW": ["3 Series", "5 Series"],
        "Mercedes": ["C Class", "E Class"],
        "Kia": ["Seltos", "Sonet"],
        "Mahindra": ["XUV700", "Scorpio"]
    }
    fuel_types = ["Petrol", "Diesel", "CNG", "Hybrid", "Electric"]
    transmissions = ["Manual", "Automatic"]
    colors = ["White", "Black", "Silver", "Red", "Blue", "Grey"]
    locations = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Pune", "Hyderabad"]

    rows = []
    for _ in range(n):
        make = rng.choice(makes)
        model = rng.choice(models[make])
        year = int(rng.randint(2005, 2023))
        age = 2025 - year  # approximate age
        mileage = round(max(5, rng.normal(loc=50 - age*2, scale=20)), 1)  # thousand km
        fuel = rng.choice(fuel_types, p=[0.5, 0.3, 0.05, 0.1, 0.05])
        transmission = rng.choice(transmissions, p=[0.7, 0.3])
        engine = round(max(0.8, rng.normal(loc=1.6, scale=0.6)), 1)  # liters
        seats = rng.choice([2,4,5,7,8], p=[0.01,0.05,0.8,0.1,0.04])
        location = rng.choice(locations)
        color = rng.choice(colors)

        # Base price logic (lakhs or thousands — be consistent; here in thousands INR)
        base = 800 + (2025 - year) * (-20)  # newer cars higher base
        # make/model premium
        make_premium = {
            "Toyota": 50, "Honda": 45, "Hyundai": 20, "Ford": 15,
            "BMW": 200, "Mercedes": 250, "Kia": 18, "Mahindra": 25
        }
        base += make_premium.get(make, 0)
        # engine / transmission / fuel adjustments
        base += (engine - 1.0) * 30
        if transmission == "Automatic":
            base += 20
        if fuel == "Electric":
            base += 150
        if fuel == "Hybrid":
            base += 80

        # mileage reduces price
        base -= mileage * 0.8

        # location effect
        location_map = {"Mumbai": 10, "Bengaluru": 8, "Delhi": 6, "Chennai": 4, "Pune": 5, "Hyderabad": 3}
        base += location_map.get(location, 0)

        # noise
        price = max(50, base + rng.normal(scale=30))

        rows.append({
            "Make": make,
            "Model": model,
            "Year": year,
            "Mileage": round(mileage, 1),
            "FuelType": fuel,
            "Transmission": transmission,
            "Engine": engine,
            "Seats": seats,
            "Location": location,
            "Color": color,
            "Price": round(price, 2)  # in thousands INR
        })

    return pd.DataFrame(rows)

# -----------------------
# Sidebar: controls
# -----------------------
st.sidebar.header("Dataset & Model Controls")
use_sample = st.sidebar.checkbox("Use sample dataset", value=True)
uploaded_file = None
if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain Price column)", type=["csv"])

n_estimators = st.sidebar.slider("RandomForest n_estimators", min_value=20, max_value=500, value=100, step=10)
train_btn = st.sidebar.button("Train model")
load_btn = st.sidebar.button("Load saved model")
save_btn = st.sidebar.button("Save current model")

# -----------------------
# Load dataset
# -----------------------
if use_sample:
    df = generate_car_sample(n=1000)
    st.sidebar.success("Sample car dataset loaded")
else:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Uploaded dataset loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")
            df = None
    else:
        st.sidebar.info("Upload a CSV or use the sample dataset")
        df = None

if df is None:
    st.warning("Please upload or enable the sample dataset to continue.")
    st.stop()

# Preview
st.subheader("Dataset preview")
st.dataframe(df.head(8))
st.markdown(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")

# -----------------------
# Basic EDA visuals
# -----------------------
st.subheader("Exploratory Visuals")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Price vs Year (scatter)**")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.scatterplot(data=df, x="Year", y="Price", hue="FuelType", alpha=0.7, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.markdown("**Price distribution**")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.histplot(df["Price"], kde=True, bins=30, ax=ax2)
    st.pyplot(fig2)

# Average price line by make (new visual)
st.markdown("**Average Price by Year for Top Makes**")
top_makes = df["Make"].value_counts().nlargest(6).index.tolist()
avg_line = df[df["Make"].isin(top_makes)].groupby(["Year", "Make"])["Price"].mean().reset_index()
fig_line, axl = plt.subplots(figsize=(8, 4))
sns.lineplot(data=avg_line, x="Year", y="Price", hue="Make", ax=axl)
axl.set_title("Average Price by Year for Top Makes")
st.pyplot(fig_line)

# -----------------------
# Pipeline & training utils
# -----------------------
categorical_cols = ["Make", "Model", "FuelType", "Transmission", "Location", "Color"]
numeric_cols = ["Year", "Mileage", "Engine", "Seats"]

def build_pipeline(n_estimators_param=100):
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
    num_pipe = Pipeline([("scaler", StandardScaler())])
    pre = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)])
    model = RandomForestRegressor(n_estimators=n_estimators_param, random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe

def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# Train / Load logic
# -----------------------
model = None
metrics = None
train_results = None

if load_btn:
    if os.path.exists(MODEL_LATEST):
        model = joblib.load(MODEL_LATEST)
        st.sidebar.success("Loaded latest saved model")
    else:
        st.sidebar.error("No saved model found")

if train_btn:
    required = set(numeric_cols + categorical_cols + ["Price"])
    if not required.issubset(set(df.columns)):
        st.sidebar.error("Dataset missing required columns: " + ", ".join(sorted(list(required - set(df.columns)))))
    else:
        st.sidebar.info("Training model — please wait...")
        X = df[numeric_cols + categorical_cols]
        y = df["Price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipe = build_pipeline(n_estimators_param=n_estimators)
        with st.spinner("Fitting RandomForest..."):
            pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        metrics = {"MAE": mae, "MSE": mse, "R2": r2}
        model = pipe
        train_results = (X_test, y_test, preds)

        st.success("Training complete")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("MSE", f"{mse:.2f}")
        st.metric("R²", f"{r2:.3f}")

        # Feature importances: get OHE feature names
        try:
            pre = model.named_steps["pre"]
            ohe = pre.named_transformers_["cat"].named_steps["ohe"]
            cat_features = ohe.get_feature_names_out(categorical_cols)
            feat_names = numeric_cols + list(cat_features)
            importances = model.named_steps["model"].feature_importances_
            fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(20)
            st.subheader("Top feature importances")
            st.table(fi.reset_index(drop=True))
        except Exception:
            st.info("Could not compute feature importances.")

        # Save model
        ensure_model_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(MODEL_DIR, f"car_price_model_{ts}.joblib")
        joblib.dump(model, path)
        joblib.dump(model, MODEL_LATEST)
        st.sidebar.success(f"Model saved: {path}")

if save_btn:
    if model is not None:
        ensure_model_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(MODEL_DIR, f"car_price_model_{ts}.joblib")
        joblib.dump(model, path)
        joblib.dump(model, MODEL_LATEST)
        st.sidebar.success(f"Model saved: {path}")
    else:
        st.sidebar.error("No trained model to save")

# -----------------------
# Whole-dataset predictions (for visuals)
# -----------------------
preds_all = None
if model is not None:
    try:
        X_all = df[numeric_cols + categorical_cols]
        preds_all = model.predict(X_all)
    except Exception as e:
        st.warning(f"Could not predict on full dataset: {e}")

# -----------------------
# User prediction UI
# -----------------------
st.header("Predict Car Price")
c1, c2, c3 = st.columns(3)

with c1:
    make_in = st.selectbox("Make", sorted(df["Make"].unique()))
    # model options limited to make
    model_options = df[df["Make"] == make_in]["Model"].unique().tolist()
    model_in = st.selectbox("Model", model_options)
    year_in = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
with c2:
    mileage_in = st.number_input("Mileage (thousands km)", min_value=0.0, max_value=500.0, value=40.0, step=0.1)
    fuel_in = st.selectbox("Fuel Type", sorted(df["FuelType"].unique()))
    trans_in = st.selectbox("Transmission", sorted(df["Transmission"].unique()))
with c3:
    engine_in = st.number_input("Engine (L)", min_value=0.6, max_value=6.0, value=1.6, step=0.1)
    seats_in = st.selectbox("Seats", sorted(df["Seats"].unique()))
    loc_in = st.selectbox("Location", sorted(df["Location"].unique()))

predict_btn = st.button("Predict Price")

if predict_btn:
    if model is None:
        st.error("No model ready. Train or load a model first.")
    else:
        inp = pd.DataFrame([{
            "Make": make_in,
            "Model": model_in,
            "Year": year_in,
            "Mileage": mileage_in,
            "FuelType": fuel_in,
            "Transmission": trans_in,
            "Engine": engine_in,
            "Seats": seats_in,
            "Location": loc_in,
            "Color": df["Color"].mode()[0]  # pick a default color if not provided
        }])
        try:
            price_pred = model.predict(inp)[0]
            st.success(f"Estimated Price: ₹ {price_pred:,.2f} (units as dataset)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -----------------------
# Prediction visuals
# -----------------------
st.markdown("---")
st.subheader("Prediction Visuals (whole dataset)")

if preds_all is not None:
    figp, axp = plt.subplots(figsize=(6, 4))
    axp.scatter(df["Price"], preds_all, alpha=0.6)
    axp.plot([df["Price"].min(), df["Price"].max()], [df["Price"].min(), df["Price"].max()], "r--")
    axp.set_xlabel("Actual Price")
    axp.set_ylabel("Predicted Price")
    st.pyplot(figp)

    # residuals
    residuals = df["Price"] - preds_all
    figr, axr = plt.subplots(figsize=(6, 3))
    sns.histplot(residuals, kde=True, ax=axr)
    axr.set_title("Residuals (Actual - Predicted)")
    st.pyplot(figr)
else:
    st.info("Train or load a model to view prediction visuals.")

# -----------------------
# Download sample dataset
# -----------------------
st.markdown("---")
st.subheader("Download sample data")
sample = generate_car_sample(n=300)
csv = sample.to_csv(index=False).encode("utf-8")
st.download_button("Download sample CSV", data=csv, file_name="car_sample.csv", mime="text/csv")

st.caption("Built with scikit-learn & Streamlit — modify pipeline/features to suit your data.")
