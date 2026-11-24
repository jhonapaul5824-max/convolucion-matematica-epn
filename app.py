import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configuración de página y firma de autoría
st.set_page_config(page_title="Laboratorio de Convolución - EPN", layout="wide")

st.title("🧮 Operaciones con Funciones y Convolución")
st.caption("Materia: Matemática Avanzada | Escuela Politécnica Nacional (EPN)")

# Firma visible de autoría
st.sidebar.subheader("👤 Datos del Autor")
st.sidebar.write("**Estudiante:** Jhonatan Ramírez")
st.sidebar.write("**Universidad:** EPN")
st.sidebar.write("**Código de Autoría:** EPN-MATH-ADV-2026-JR")

st.markdown("""
> **Autor del Código:** Jhonatan Ramírez  
> *Aplicación interactiva para el cálculo de operaciones básicas y convolución continua de señales estándar.*
""")

# Parámetros de funciones
st.sidebar.header("⚙️ Configuración de Señales")
t_min, t_max = st.sidebar.slider("Rango de tiempo (t)", -10.0, 10.0, (-5.0, 5.0), step=0.5)
t = np.linspace(t_min, t_max, 1000)
dt = t[1] - t[0]

# Lista de señales solicitadas
opciones_senales = [
    "Pulso Rectangular / Compuerta",
    "Pulso Triangular",
    "Exponencial Decreciente",
    "Escalón Unitario u(t)",
    "Impulso Unitario δ(t)",
    "Rampa r(t)",
    "Función Signo sgn(t)"
]

tipo_f = st.sidebar.selectbox("Selecciona f(t):", opciones_senales)
tipo_g = st.sidebar.selectbox("Selecciona g(t):", opciones_senales)

def generar_funcion(tipo, t):
    if tipo == "Pulso Rectangular / Compuerta":
        return np.where((t >= -1) & (t <= 1), 1.0, 0.0)
    elif tipo == "Pulso Triangular":
        return np.maximum(0.0, 1.0 - np.abs(t))
    elif tipo == "Exponencial Decreciente":
        return np.where(t >= 0, np.exp(-t), 0.0)
    elif tipo == "Escalón Unitario u(t)":
        return np.where(t >= 0, 1.0, 0.0)
    elif tipo == "Impulso Unitario δ(t)":
        # Aproximación numérica del impulso unitario con área = 1
        ancho = 0.05
        return np.where((t >= -ancho/2) & (t <= ancho/2), 1.0 / ancho, 0.0)
    elif tipo == "Rampa r(t)":
        return np.where(t >= 0, t, 0.0)
    elif tipo == "Función Signun sgn(t)":
        return np.sign(t)
    return np.zeros_like(t)

f_t = generar_funcion(tipo_f, t)
g_t = generar_funcion(tipo_g, t)

tab1, tab2, tab3 = st.tabs(["📊 Operaciones Básicas", "🔄 Convolución (f * g)", "💻 Código Fuente del Autor"])

with tab1:
    st.subheader("Suma, Resta y Multiplicación de Funciones")
    operacion = st.radio("Selecciona la operación:", ["Suma: f(t) + g(t)", "Resta: f(t) - g(t)", "Producto: f(t) * g(t)"], horizontal=True)
    
    if operacion == "Suma: f(t) + g(t)":
        res = f_t + g_t
        lbl = "f(t) + g(t)"
    elif operacion == "Resta: f(t) - g(t)":
        res = f_t - g_t
        lbl = "f(t) - g(t)"
    else:
        res = f_t * g_t
        lbl = "f(t) · g(t)"

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t, f_t, label=f"f(t) = {tipo_f}", color="blue")
    ax[0].grid(True); ax[0].legend(loc="upper right")
    
    ax[1].plot(t, g_t, label=f"g(t) = {tipo_g}", color="green")
    ax[1].grid(True); ax[1].legend(loc="upper right")
    
    ax[2].plot(t, res, label=lbl, color="red")
    ax[2].grid(True); ax[2].legend(loc="upper right")
    
    st.pyplot(fig)

with tab2:
    st.subheader("Convolución Continua: (f * g)(t)")
    st.latex(r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau")
    
    conv_res = np.convolve(f_t, g_t, mode='same') * dt
    
    fig_conv, ax_conv = plt.subplots(figsize=(10, 4))
    ax_conv.plot(t, conv_res, label="(f * g)(t)", color="purple", lw=2)
    ax_conv.set_title(f"Convolución entre {tipo_f} y {tipo_g}")
    ax_conv.grid(True)
    ax_conv.legend(loc="upper right")
    st.pyplot(fig_conv)

with tab3:
    st.subheader("Verificación de Autoría")
    st.info("Código desarrollado por Jhonatan Ramírez para la materia de Matemática Avanzada (EPN).")
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
