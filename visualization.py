import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi untuk menghitung dan menampilkan metrik evaluasi model
def show_model_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.subheader("Model Evaluation Metrics")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R²): {r2:.2f}")

# Fungsi untuk visualisasi prediksi vs nilai aktual
def plot_predictions_vs_actual(y_true, y_pred):
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    fig = px.scatter(df, x='Actual', y='Predicted', 
                     title='Predictions vs Actual Values',
                     labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
                     trendline='ols')  # Tambahkan garis regresi
    st.plotly_chart(fig)

# Fungsi untuk visualisasi residuals
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    df = pd.DataFrame({'Predicted': y_pred, 'Residuals': residuals})
    fig = px.scatter(df, x='Predicted', y='Residuals', 
                     title='Residuals vs Predicted Values',
                     labels={'Predicted': 'Predicted Values', 'Residuals': 'Residuals'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")  # Garis horizontal pada y=0
    st.plotly_chart(fig)

# Fungsi utama untuk menampilkan visualisasi
def show_visualizations():
    # Data contoh (ganti dengan data aktual Anda)
    y_true = pd.Series([80, 85, 78, 90, 88])  # Nilai sebenarnya
    y_pred = pd.Series([82, 84, 79, 89, 87])  # Prediksi model

    show_model_metrics(y_true, y_pred)
    plot_predictions_vs_actual(y_true, y_pred)
    plot_residuals(y_true, y_pred)
