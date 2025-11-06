import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from xgboost import XGBRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

# --- Compatibilidad NumPy 2.0 ---
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'bool_'):
    np.bool_ = np.bool_

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")
st.set_page_config(page_title="Industrial Forecasting", page_icon="📊", layout="wide")

# ============================================================
# 📥 Load data
# ============================================================
@st.cache_data
def load_data():
    cement_demand = pd.read_csv('data/cement_demand.csv', parse_dates=['date'])
    cement_real = pd.read_csv('data/cement_demand_real.csv', parse_dates=['date'])
    plant_energy = pd.read_csv('data/plant_energy.csv', parse_dates=['date'])
    plant_real = pd.read_csv('data/plant_energy_real.csv', parse_dates=['date'])
    plant_sensors = pd.read_csv('data/plant_sensors.csv', parse_dates=['timestamp'])
    plant_sensors_real = pd.read_csv('data/plant_sensors_real.csv', parse_dates=['timestamp'])
    return cement_demand, cement_real, plant_energy, plant_real, plant_sensors, plant_sensors_real


cement_demand, cement_real, plant_energy, plant_real, plant_sensors, plant_sensors_real = load_data()

# ============================================================
# 🧱 Streamlit layout
# ============================================================
st.set_page_config(page_title="Holcim BI Predictive Demo", layout="wide")
st.title("🏗️ Holcim BI Predictive Models Showcase")
st.markdown("Demostración de tres modelos de predicción y análisis industrial usando datos sintéticos representativos de operaciones cementeras.")

tabs = st.tabs(["📈 Cement Demand (Prophet)", "⚙️ Plant Energy (XGBoost)", "🧠 Sensor Anomalies (Autoencoder)"])

# ============================================================
# 1️⃣ Prophet - Cement Demand
# ============================================================
with tabs[0]:
    st.subheader("📈 Cement Demand Forecast")

    df_train = cement_demand.groupby('date')['demand_tons'].sum().reset_index().rename(columns={'date':'ds','demand_tons':'y'})
    df_test = cement_real.groupby('date')['demand_tons'].sum().reset_index().rename(columns={'date':'ds','demand_tons':'y'})

    model = Prophet(seasonality_mode='additive', yearly_seasonality=True)
    model.fit(df_train)

    future = model.make_future_dataframe(periods=len(df_test), freq='MS')
    forecast = model.predict(future)
    pred_future = forecast.tail(len(df_test))[['ds','yhat']].reset_index(drop=True)
    merged = pd.merge(df_test, pred_future, on='ds')

    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    r2 = r2_score(merged['y'], merged['yhat'])

    st.markdown(f"**MAE:** {mae:.2f} | **RMSE:** {rmse:.2f} | **R²:** {r2:.3f}")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_train['ds'], df_train['y'], label='Train')
    ax.plot(df_test['ds'], df_test['y'], label='Real Future', color='green')
    ax.plot(pred_future['ds'], pred_future['yhat'], label='Forecast', color='orange')
    ax.set_title('Cement Demand Forecast vs Real')
    ax.legend()
    st.pyplot(fig)


# ============================================================
# 2️⃣ XGBoost - Plant Energy Regression
# ============================================================
with tabs[1]:
    st.subheader("⚙️ Plant Energy Prediction")

    feature_cols = ['production_tons','kiln_temperature','humidity_pct','maintenance']
    target_col = 'energy_mwh'

    X_train = plant_energy[feature_cols]
    y_train = plant_energy[target_col]
    X_test = plant_real[feature_cols]
    y_test = plant_real[target_col]

    X_train = pd.concat([X_train, pd.get_dummies(plant_energy['fuel_type'], prefix='fuel')], axis=1)
    X_test = pd.concat([X_test, pd.get_dummies(plant_real['fuel_type'], prefix='fuel')], axis=1)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"**MAE:** {mae:.2f} | **RMSE:** {rmse:.2f} | **R²:** {r2:.3f}")

    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Real Energy (MWh)")
    ax.set_ylabel("Predicted Energy (MWh)")
    ax.set_title("XGBoost: Real vs Predicted Energy")
    st.pyplot(fig)
    image = Image.open("data/image2.png")
    st.image(image, use_container_width=True)

# ============================================================
# 3️⃣ Autoencoder - Sensor Anomaly Detection
# ============================================================


with tabs[2]:
    st.subheader("🧠 Sensor Anomaly Detection")

    sensor_features = ['sensor_temp','sensor_vibration','sensor_pressure','sensor_co2','sensor_humidity','output_efficiency']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(plant_sensors[sensor_features])
    X_test = scaler.transform(plant_sensors_real[sensor_features])

    input_dim = X_train.shape[1]
    encoding_dim = 3
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    autoencoder.fit(X_train, X_train, epochs=25, batch_size=32, shuffle=True, verbose=0)

    reconstructions = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

    train_recon = np.mean(np.power(X_train - autoencoder.predict(X_train), 2), axis=1)
    threshold = np.percentile(train_recon, 95)

    plant_sensors_real['reconstruction_error'] = mse
    plant_sensors_real['pred_anomaly'] = (mse > threshold).astype(int)

    cm = confusion_matrix(plant_sensors_real['anomaly'], plant_sensors_real['pred_anomaly'])

    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix - Autoencoder")
    st.pyplot(fig)
    # --- Interpretación automática de la matriz de confusión ---
    image = Image.open("data/image.png")
    st.image(image, caption="Tabla explicativa de la Logica", use_container_width=True)
    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    st.markdown(f"""
    ### 📊 Interpretación del Modelo Autoencoder

    - **Exactitud global:** {accuracy*100:.2f}%
    - **Precisión (detección de anomalías):** {precision*100:.2f}%
    - **Recall / Sensibilidad:** {recall*100:.2f}%
    - **Falsos positivos:** {false_positive_rate*100:.2f}%
    - **Falsos negativos:** {false_negative_rate*100:.2f}%

    **🧩 Interpretación:**
    El modelo identificó correctamente el **{recall*100:.1f}%** de las anomalías reales y mantuvo una tasa de falsos positivos del **{false_positive_rate*100:.1f}%**.  
    Un bajo número de falsos negativos indica buena sensibilidad para detectar fallas, mientras que un bajo número de falsos positivos refleja que el modelo no está sobredetectando eventos normales como anómalos.
    """)



    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.histplot(plant_sensors_real, x='reconstruction_error', hue='anomaly', bins=40, kde=True, ax=ax2)
    ax2.set_title("Reconstruction Error Distribution")
    st.pyplot(fig2)
