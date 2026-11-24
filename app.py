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
> *Aplicación interactiva con parámetros de amplitud, desplazamiento temporal y cálculo de convolución analítica.*
""")

opciones_senales = [
    "Pulso Rectangular / Compuerta",
    "Pulso Triangular",
    "Exponencial Decreciente",
    "Escalón Unitario u(t)",
    "Impulso Unitario δ(t)",
    "Rampa r(t)",
    "Función Signo sgn(t)"
]

# Configuración global de tiempo
st.sidebar.header("⚙️ Rango de Tiempo")
t_min, t_max = st.sidebar.slider("Eje t", -10.0, 10.0, (-6.0, 6.0), step=0.5)
t = np.linspace(t_min, t_max, 1000)
dt = t[1] - t[0]

# --- PARÁMETROS PARA f(t) ---
st.sidebar.header("🔹 Configuración de f(t)")
tipo_f = st.sidebar.selectbox("Función f(t):", opciones_senales, index=0)
es_unitaria_f = tipo_f in ["Escalón Unitario u(t)", "Impulso Unitario δ(t)", "Función Signo sgn(t)"]

amp_f = 1.0 if es_unitaria_f else st.sidebar.slider("Amplitud A_f", 0.5, 5.0, 1.0, step=0.5)
t0_f = st.sidebar.slider("Desplazamiento t0 para f(t - t0)", -4.0, 4.0, 0.0, step=0.5)

# --- PARÁMETROS PARA g(t) ---
st.sidebar.header("🟢 Configuración de g(t)")
tipo_g = st.sidebar.selectbox("Función g(t):", opciones_senales, index=0)
es_unitaria_g = tipo_g in ["Escalón Unitario u(t)", "Impulso Unitario δ(t)", "Función Signo sgn(t)"]

amp_g = 1.0 if es_unitaria_g else st.sidebar.slider("Amplitud A_g", 0.5, 5.0, 1.0, step=0.5)
t0_g = st.sidebar.slider("Desplazamiento t0 para g(t - t0)", -4.0, 4.0, 0.0, step=0.5)


def generar_funcion(tipo, t, amp, t0):
    tau = t - t0
    if tipo == "Pulso Rectangular / Compuerta":
        return amp * np.where((tau >= -1) & (tau <= 1), 1.0, 0.0)
    elif tipo == "Pulso Triangular":
        return amp * np.maximum(0.0, 1.0 - np.abs(tau))
    elif tipo == "Exponencial Decreciente":
        return amp * np.where(tau >= 0, np.exp(-tau), 0.0)
    elif tipo == "Escalón Unitario u(t)":
        return np.where(tau >= 0, 1.0, 0.0)
    elif tipo == "Impulso Unitario δ(t)":
        ancho = 0.05
        return np.where((tau >= -ancho/2) & (tau <= ancho/2), 1.0 / ancho, 0.0)
    elif tipo == "Rampa r(t)":
        return amp * np.where(tau >= 0, tau, 0.0)
    elif tipo == "Función Signo sgn(t)":
        return np.sign(tau)
    return np.zeros_like(t)


def obtener_intervalos_analiticos(f_name, g_name, A1, A2, shift1, shift2):
    shift_total = shift1 + shift2
    A_total = A1 * A2
    s_tot = f"{shift_total:.1f}"
    
    if f_name == "Pulso Rectangular / Compuerta" and g_name == "Pulso Rectangular / Compuerta":
        t_low = f"{-2 + shift_total:.1f}"
        t_mid = s_tot
        t_high = f"{2 + shift_total:.1f}"
        
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + t_low + r" \\ "
            f"{A_total:.1f}" r" \cdot (t - " + t_low + r"), & " + t_low + r" \le t < " + t_mid + r" \\ "
            f"{A_total:.1f}" r" \cdot (" + t_high + r" - t), & " + t_mid + r" \le t \le " + t_high + r" \\ "
            r"0, & t > " + t_high + r" \end{cases}"
        )
        return eq, f"Resultado: Triángulo de amplitud {A_total*2:.1f} centrado en t = {s_tot} sobre el intervalo [{t_low}, {t_high}]."

    elif f_name == "Exponencial Decreciente" and g_name == "Escalón Unitario u(t)":
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + s_tot + r" \\ "
            f"{A1:.1f}" r" \cdot (1 - e^{-(t - " + s_tot + r")}), & t \ge " + s_tot + r" \end{cases}"
        )
        return eq, f"Resultado: Crecimiento exponencial acotado por A = {A1:.1f} iniciado en t = {s_tot}."

    elif "Impulso Unitario" in f_name or "Impulso Unitario" in g_name:
        eq = r"(f * g)(t) = f_{\text{desplazada}}(t - " + s_tot + r")"
        return eq, f"Propiedad de tamizado: El impulso desplaza la salida a t = {s_tot}."

    else:
        eq = r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau"
        return eq, f"Evaluación numérica continua con A_f = {A1:.1f}, A_g = {A2:.1f} y desplazamiento neto en t = {s_tot}."


f_t = generar_funcion(tipo_f, t, amp_f, t0_f)
g_t = generar_funcion(tipo_g, t, amp_g, t0_g)

tab1, tab2, tab3 = st.tabs(["📊 Operaciones Básicas", "🔄 Convolución e Intervalos", "💻 Código Fuente del Autor"])

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
    ax[0].plot(t, f_t, label=f"f(t): {tipo_f} (A={amp_f}, t0={t0_f})", color="blue")
    ax[0].grid(True); ax[0].legend(loc="upper right")
    
    ax[1].plot(t, g_t, label=f"g(t): {tipo_g} (A={amp_g}, t0={t0_g})", color="green")
    ax[1].grid(True); ax[1].legend(loc="upper right")
    
    ax[2].plot(t, res, label=lbl, color="red")
    ax[2].grid(True); ax[2].legend(loc="upper right")
    
    st.pyplot(fig)

with tab2:
    st.subheader("Convolución Continua con Parámetros A y t0")
    st.latex(r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau")
    
    latex_eq, nota = obtener_intervalos_analiticos(tipo_f, tipo_g, amp_f, amp_g, t0_f, t0_g)
    
    st.markdown("### 📝 Expresión Analítica Ajustada por Intervalos")
    st.latex(latex_eq)
    st.caption(f"📌 **Nota de análisis:** {nota}")
    
    st.markdown("---")
    st.markdown("### 📈 Gráfica del Resultado Convolucionado")
    
    conv_res = np.convolve(f_t, g_t, mode='same') * dt
    
    fig_conv, ax_conv = plt.subplots(figsize=(10, 4))
    ax_conv.plot(t, conv_res, label="(f * g)(t)", color="purple", lw=2)
    ax_conv.set_title(f"Convolución con desplazamientos (t0_f={t0_f}, t0_g={t0_g})")
    ax_conv.set_xlabel("t")
    ax_conv.set_ylabel("(f * g)(t)")
    ax_conv.grid(True)
    ax_conv.legend(loc="upper right")
    st.pyplot(fig_conv)

with tab3:
    st.subheader("Verificación de Autoría")
    st.info("Código desarrollado por Jhonatan Ramírez para la materia de Matemática Avanzada (EPN).")
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
