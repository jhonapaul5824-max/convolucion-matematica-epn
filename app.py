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
> *Aplicación interactiva con funciones por intervalos, señales positivas/negativas, parámetros de amplitud, desplazamiento temporal y animación paso a paso de la convolución.*
""")

opciones_senales = [
    "Pulso Rectangular / Compuerta",
    "Pulso Triangular",
    "Exponencial Decreciente",
    "Exponencial Creciente (t ≤ t0)",
    "Escalón Unitario u(t)",
    "Impulso Unitario δ(t)",
    "Rampa r(t)",
    "Función Signo sgn(t)",
    "Seno sin(t)",
    "Coseno cos(t)",
    "Función Sinc sinc(t)",
    "Pulso Gaussiano",
    "Parábola / Cuadrática t²",
    "Constante"
]

senales_unitarias = [
    "Escalón Unitario u(t)",
    "Impulso Unitario δ(t)",
    "Función Signo sgn(t)"
]

# Configuración global de tiempo
st.sidebar.header("⚙️ Rango de Tiempo")
t_min, t_max = st.sidebar.slider("Eje t", -10.0, 10.0, (-6.0, 6.0), step=0.5)
t = np.linspace(t_min, t_max, 1000)
dt = t[1] - t[0]


def generar_funcion(tipo, eje_t, amp, t0):
    tau = eje_t - t0
    if tipo == "Pulso Rectangular / Compuerta":
        return amp * np.where((tau >= -1) & (tau <= 1), 1.0, 0.0)
    elif tipo == "Pulso Triangular":
        return amp * np.maximum(0.0, 1.0 - np.abs(tau))
    elif tipo == "Exponencial Decreciente":
        return amp * np.where(tau >= 0, np.exp(-tau), 0.0)
    elif tipo == "Exponencial Creciente (t ≤ t0)":
        return amp * np.where(tau <= 0, np.exp(tau), 0.0)
    elif tipo == "Escalón Unitario u(t)":
        return amp * np.where(tau >= 0, 1.0, 0.0)
    elif tipo == "Impulso Unitario δ(t)":
        ancho = 0.05
        return amp * np.where((tau >= -ancho/2) & (tau <= ancho/2), 1.0 / ancho, 0.0)
    elif tipo == "Rampa r(t)":
        return amp * np.where(tau >= 0, tau, 0.0)
    elif tipo == "Función Signo sgn(t)":
        return amp * np.sign(tau)
    elif tipo == "Seno sin(t)":
        return amp * np.sin(tau)
    elif tipo == "Coseno cos(t)":
        return amp * np.cos(tau)
    elif tipo == "Función Sinc sinc(t)":
        return amp * np.sinc(tau)
    elif tipo == "Pulso Gaussiano":
        return amp * np.exp(-(tau ** 2))
    elif tipo == "Parábola / Cuadrática t²":
        return amp * (tau ** 2)
    elif tipo == "Constante":
        return amp * np.ones_like(eje_t)
    return np.zeros_like(eje_t)


def fmt_sub(val):
    """Formatea la expresión t - val de forma matemática limpia sin dobles signos."""
    if val < 0:
        return f"t + {abs(val):.1f}"
    elif val > 0:
        return f"t - {val:.1f}"
    else:
        return "t"


def fmt_amp(amp):
    """Formatea el multiplicador de amplitud soportando valores negativos."""
    if abs(amp - 1.0) < 1e-5:
        return ""
    elif abs(amp + 1.0) < 1e-5:
        return "-"
    else:
        return f"{amp:.1f} \\cdot "


def expresion_tramo_latex(tipo, amp, t0):
    """Genera una representación LaTeX corta para un tramo de f(t)."""
    s = fmt_sub(t0)
    a = fmt_amp(amp)
    if tipo == "Pulso Rectangular / Compuerta":
        return f"{a}\\Pi({s})"
    elif tipo == "Pulso Triangular":
        return f"{a}\\Lambda({s})"
    elif tipo == "Exponencial Decreciente":
        return f"{a}e^{{-({s})}}"
    elif tipo == "Exponencial Creciente (t ≤ t0)":
        return f"{a}e^{{{s}}}"
    elif tipo == "Escalón Unitario u(t)":
        return f"{a}u({s})"
    elif tipo == "Impulso Unitario δ(t)":
        return f"{a}\\delta({s})"
    elif tipo == "Rampa r(t)":
        return f"{a}r({s})"
    elif tipo == "Función Signo sgn(t)":
        return f"{a}\\text{{sgn}}({s})"
    elif tipo == "Seno sin(t)":
        return f"{a}\\sin({s})"
    elif tipo == "Coseno cos(t)":
        return f"{a}\\cos({s})"
    elif tipo == "Función Sinc sinc(t)":
        return f"{a}\\text{{sinc}}({s})"
    elif tipo == "Pulso Gaussiano":
        return f"{a}e^{{-({s})^2}}"
    elif tipo == "Parábola / Cuadrática t²":
        return f"{a}({s})^2"
    elif tipo == "Constante":
        return f"{amp:.1f}"
    return "0"


def selector_amplitud(tipo, label, key_prefix, container=st.sidebar):
    """Permite amplitud negativa (-5 a 5) o invertir signo (-1/+1) en señales unitarias."""
    if tipo in senales_unitarias:
        negativa = container.checkbox(f"Invertir signo (-) {label}", value=False, key=f"{key_prefix}_neg")
        return -1.0 if negativa else 1.0
    else:
        return container.slider(f"Amplitud {label}", -5.0, 5.0, 1.0, step=0.5, key=f"{key_prefix}_amp")


# --- PARÁMETROS PARA f(t) ---
st.sidebar.header("🔹 Configuración de f(t) [Señal Fija]")
modo_f = st.sidebar.radio("Modo de f(t):", ["Función Única", "Por Intervalos (hasta 4 tramos)"])

tramos_f = []
if modo_f == "Función Única":
    tipo_f = st.sidebar.selectbox("Función f(t):", opciones_senales, index=0)
    amp_f = selector_amplitud(tipo_f, "A_f", "f_unica")
    t0_f = st.sidebar.slider("Desplazamiento t0 para f(t - t0)", -4.0, 4.0, 0.0, step=0.5)
    f_t = generar_funcion(tipo_f, t, amp_f, t0_f)
    label_f = f"f(t): {tipo_f} (A={amp_f}, t0={t0_f})"
else:
    num_tramos = st.sidebar.number_input("Número de intervalos (máx. 4):", min_value=1, max_value=4, value=2, step=1)
    f_t = np.zeros_like(t)
    
    for i in range(int(num_tramos)):
        ini_def = round(max(t_min, -2.0 + i * 1.5), 1)
        fin_def = round(min(t_max, ini_def + 1.5), 1)
        with st.sidebar.expander(f"Tramo {i + 1}", expanded=(i == 0)):
            tipo_i = st.selectbox(f"Señal tramo {i + 1}:", opciones_senales, index=i % len(opciones_senales), key=f"tipo_f_{i}")
            amp_i = selector_amplitud(tipo_i, f"A_{i + 1}", f"f
