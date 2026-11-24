import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configuración de página y firma de autoría
st.set_page_config(page_title="Laboratorio de Convolución - EPN", layout="wide")

st.title("🧮 Operaciones con Funciones y Convolución")
st.caption("Materia: Matemática Avanzada | Escuela Politécnica Nacional (EPN)")

# Firma visible de autoría para comprobación del docente
st.sidebar.subheader("👤 Datos del Autor")
st.sidebar.write("**Estudiante:** Jhonatan Ramírez")
st.sidebar.write("**Universidad:** EPN")
st.sidebar.write("**Código de Autoría:** EPN-MATH-ADV-2026-JR")

st.markdown("""
> **Autor del Código:** Jhonatan Ramírez  
> *Aplicación interactiva para el cálculo de suma, resta, multiplicación y convolución de funciones.*
""")

# Parámetros de funciones
st.sidebar.header("⚙️ Configuración")
t_min, t_max = st.sidebar.slider("Rango de tiempo (t)", -10.0, 10.0, (-5.0, 5.0), step=0.5)
t = np.linspace(t_min, t_max, 1000)
dt = t[1] - t[0]

tipo_f = st.sidebar.selectbox("Selecciona f(t):", ["Pulso Rectangular", "Exponencial Decreciente", "Seno", "Triangular"])
tipo_g = st.sidebar.selectbox("Selecciona g(t):", ["Pulso Rectangular", "Exponencial Decreciente", "Cos", "Triangular"])

def generar_funcion(tipo, t):
    if tipo == "Pulso Rectangular":
        return np.where((t >= -1) & (t <= 1), 1.0, 0.0)
    elif tipo == "Exponencial Decreciente":
        return np.where(t >= 0, np.exp(-t), 0.0)
    elif tipo == "Seno":
        return np.sin(2 * np.pi * 0.5 * t)
    elif tipo == "Cos":
        return np.cos(2 * np.pi * 0.5 * t)
    elif tipo == "Triangular":
        return np.maximum(0, 1 - np.abs(t))

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
    ax[0].plot(t, f_t, label="f(t)", color="blue")
    ax[0].grid(True); ax[0].legend()
    
    ax[1].plot(t, g_t, label="g(t)", color="green")
    ax[1].grid(True); ax[1].legend()
    
    ax[2].plot(t, res, label=lbl, color="red")
    ax[2].grid(True); ax[2].legend()
    
    st.pyplot(fig)

with tab2:
    st.subheader("Convolución Continua: (f * g)(t)")
    st.latex(r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau")
    
    conv_res = np.convolve(f_t, g_t, mode='same') * dt
    
    fig_conv, ax_conv = plt.subplots(figsize=(10, 4))
    ax_conv.plot(t, conv_res, label="(f * g)(t)", color="purple", lw=2)
    ax_conv.set_title(f"Convolución entre {tipo_f} y {tipo_g}")
    ax_conv.grid(True)
    ax_conv.legend()
    st.pyplot(fig_conv)

with tab3:
    st.subheader("Verificación de Autoría")
    st.info("Código desarrollado por Jhonatan Ramírez para la materia de Matemática Avanzada (EPN).")
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")