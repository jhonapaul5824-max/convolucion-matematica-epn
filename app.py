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
> *Aplicación interactiva con parámetros de amplitud, desplazamiento temporal y cálculo analítico exacto por intervalos.*
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


def fmt_sub(val):
    """Formatea la expresión t - val de forma matemática limpia sin dobles signos."""
    if val < 0:
        return f"t + {abs(val):.1f}"
    elif val > 0:
        return f"t - {val:.1f}"
    else:
        return "t"


def fmt_amp(amp):
    """Omite el multiplicador de amplitud si es 1.0."""
    return "" if abs(amp - 1.0) < 1e-5 else f"{amp:.1f} \cdot "


def obtener_intervalos_analiticos(f_name, g_name, A1, A2, shift1, shift2):
    shift_total = shift1 + shift2
    A_total = A1 * A2
    s_tot = f"{shift_total:.1f}"
    str_amp_tot = fmt_amp(A_total)
    str_amp_1 = fmt_amp(A1)
    
    # 1. Pulso Rectangular * Pulso Rectangular
    if "Compuerta" in f_name and "Compuerta" in g_name:
        t_low_num = -2 + shift_total
        t_mid_num = shift_total
        t_high_num = 2 + shift_total
        
        t_low = f"{t_low_num:.1f}"
        t_mid = f"{t_mid_num:.1f}"
        t_high = f"{t_high_num:.1f}"
        
        expr_inc = fmt_sub(t_low_num)
        
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + t_low + r" \\ "
            + str_amp_tot + r"(" + expr_inc + r"), & " + t_low + r" \le t < " + t_mid + r" \\ "
            + str_amp_tot + r"(" + t_high + r" - t), & " + t_mid + r" \le t \le " + t_high + r" \\ "
            r"0, & t > " + t_high + r" \end{cases}"
        )
        return eq, f"Triángulo de amplitud {A_total*2:.1f} centrado en t = {s_tot} sobre el intervalo [{t_low}, {t_high}]."

    # 2. Exponencial * Escalón
    elif ("Exponencial" in f_name and "Escalón" in g_name) or ("Escalón" in f_name and "Exponencial" in g_name):
        exp_t = fmt_sub(shift_total)
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + s_tot + r" \\ "
            + str_amp_1 + r"(1 - e^{-(" + exp_t + r")}), & t \ge " + s_tot + r" \end{cases}"
        )
        return eq, f"Crecimiento exponencial acotado iniciado en t = {s_tot}."

    # 3. Pulso Rectangular * Exponencial
    elif ("Compuerta" in f_name and "Exponencial" in g_name) or ("Exponencial" in f_name and "Compuerta" in g_name):
        t1_num = -1 + shift_total
        t2_num = 1 + shift_total
        t1 = f"{t1_num:.1f}"
        t2 = f"{t2_num:.1f}"
        
        exp1 = fmt_sub(t1_num)
        exp2 = fmt_sub(t2_num)
        
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + t1 + r" \\ "
            + str_amp_tot + r"(1 - e^{-(" + exp1 + r")}), & " + t1 + r" \le t < " + t2 + r" \\ "
            + str_amp_tot + r"e^{-(" + exp2 + r")}(1 - e^{-2}), & t \ge " + t2 + r" \end{cases}"
        )
        return eq, f"Respuesta de pulso acotado entre t = {t1} y t = {t2}."

    # 4. Escalón * Escalón
    elif "Escalón" in f_name and "Escalón" in g_name:
        rampa_expr = fmt_sub(shift_total)
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + s_tot + r" \\ "
            + rampa_expr + r", & t \ge " + s_tot + r" \end{cases}"
        )
        return eq, f"Rampa resultante iniciada en t = {s_tot}."

    # 5. Pulso Rectangular * Escalón
    elif ("Compuerta" in f_name and "Escalón" in g_name) or ("Escalón" in f_name and "Compuerta" in g_name):
        t1_num = -1 + shift_total
        t2_num = 1 + shift_total
        t1 = f"{t1_num:.1f}"
        t2 = f"{t2_num:.1f}"
        expr_lin = fmt_sub(t1_num)
        
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + t1 + r" \\ "
            + str_amp_1 + r"(" + expr_lin + r"), & " + t1 + r" \le t < " + t2 + r" \\ "
            f"{2 * A1:.1f}, & t \ge " + t2 + r" \end{cases}"
        )
        return eq, f"Crecimiento lineal entre {t1} y {t2}, constante después."

    # 6. Impulso Unitario
    elif "Impulso" in f_name or "Impulso" in g_name:
        otra = g_name if "Impulso" in f_name else f_name
        exp_imp = fmt_sub(shift_total)
        eq = r"(f * g)(t) = \text{" + otra.split()[0] + r"}(" + exp_imp + r")"
        return eq, f"Propiedad de tamizado: La función se desplaza totalmente a t = {s_tot}."

    # 7. Caso General
    else:
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + s_tot + r" \\ "
            r"\int_{-\infty}^{t} f(\tau) g(t - \tau) d\tau, & t \ge " + s_tot + r" \end{cases}"
        )
        return eq, f"Evaluación de la integral por tramos con desplazamiento neto en t = {s_tot}."


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
