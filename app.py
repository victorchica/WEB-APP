import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Configuración de página estilo profesional
st.set_page_config(page_title="Inferencia Bayesiana Pro", layout="wide")

st.title("🏛️ Estimador de Parámetros: Inferencia Bayesiana")
st.markdown("""
Esta aplicación realiza una **actualización Bayesiana** para una variable aleatoria de Bernoulli. 
Utilizamos una distribución **Beta** como *prior* (conjugada) para estimar el parámetro $\\theta$ (probabilidad de éxito).
""")

# --- BARRA LATERAL (INPUTS) ---
st.sidebar.header("Configuración de Parámetros")

# Prior (Lo que creemos antes de los datos)
st.sidebar.subheader("1. Creencias Previas (Prior)")
alpha_prior = st.sidebar.slider("Alpha (Éxitos previos)", 0.1, 20.0, 2.0)
beta_prior = st.sidebar.slider("Beta (Fracasos previos)", 0.1, 20.0, 2.0)

# Datos Observados (La realidad)
st.sidebar.subheader("2. Datos Observados (Likelihood)")
exitos = st.sidebar.number_input("Número de Éxitos", min_value=0, value=7)
total = st.sidebar.number_input("Total de Ensayos", min_value=1, value=10)

# --- CÁLCULO MATEMÁTICO ---
# La magia de la distribución conjugada: 
# Posterior ~ Beta(alpha + exitos, beta + (total - exitos))
alpha_post = alpha_prior + exitos
beta_post = beta_prior + (total - exitos)

x = np.linspace(0, 1, 500)
y_prior = stats.beta.pdf(x, alpha_prior, beta_prior)
y_posterior = stats.beta.pdf(x, alpha_post, beta_post)

# --- VISUALIZACIÓN ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, y_prior, label=f"Prior: Beta({alpha_prior}, {beta_prior})", linestyle="--", color="#7f8c8d")
ax.plot(x, y_posterior, label=f"Posterior: Beta({alpha_post}, {beta_post})", color="#2980b9", linewidth=3)
ax.fill_between(x, 0, y_posterior, color="#2980b9", alpha=0.2)

# Estética del gráfico
ax.set_title("Actualización de la Distribución de Probabilidad", fontsize=15)
ax.set_xlabel("Valor del Parámetro θ", fontsize=12)
ax.set_ylabel("Densidad de Probabilidad", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# --- MÉTRICAS FINALES ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Media Posterior (E)", round(alpha_post / (alpha_post + beta_post), 3))
with col2:
    st.metric("Moda (MAP)", round((alpha_post - 1) / (alpha_post + beta_post - 2) if alpha_post > 1 else 0, 3))
with col3:
    st.metric("Desviación Estándar", round(stats.beta.std(alpha_post, beta_post), 4))