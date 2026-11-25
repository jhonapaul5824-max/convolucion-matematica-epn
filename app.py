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
> *Aplicación interactiva con funciones por intervalos, parámetros de amplitud, desplazamiento temporal y animación paso a paso de la convolución.*
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


def generar_funcion(tipo, eje_t, amp, t0):
    tau = eje_t - t0
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
    """Omite el multiplicador de amplitud si es 1.0."""
    return "" if abs(amp - 1.0) < 1e-5 else f"{amp:.1f} \\cdot "


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
    elif tipo == "Escalón Unitario u(t)":
        return f"u({s})"
    elif tipo == "Impulso Unitario δ(t)":
        return f"\\delta({s})"
    elif tipo == "Rampa r(t)":
        return f"{a}r({s})"
    elif tipo == "Función Signo sgn(t)":
        return f"\\text{{sgn}}({s})"
    return "0"


# --- PARÁMETROS PARA f(t) ---
st.sidebar.header("🔹 Configuración de f(t) [Señal Fija]")
modo_f = st.sidebar.radio("Modo de f(t):", ["Función Única", "Por Intervalos (hasta 4 tramos)"])

tramos_f = []
if modo_f == "Función Única":
    tipo_f = st.sidebar.selectbox("Función f(t):", opciones_senales, index=0)
    es_unitaria_f = tipo_f in ["Escalón Unitario u(t)", "Impulso Unitario δ(t)", "Función Signo sgn(t)"]
    amp_f = 1.0 if es_unitaria_f else st.sidebar.slider("Amplitud A_f", 0.5, 5.0, 1.0, step=0.5)
    t0_f = st.sidebar.slider("Desplazamiento t0 para f(t - t0)", -4.0, 4.0, 0.0, step=0.5)
    f_t = generar_funcion(tipo_f, t, amp_f, t0_f)
    label_f = f"f(t): {tipo_f} (A={amp_f}, t0={t0_f})"
else:
    num_tramos = st.sidebar.number_input("Número de intervalos (máx. 4):", min_value=1, max_value=4, value=2, step=1)
    f_t = np.zeros_like(t)
    
    # Intervalos iniciales sugeridos para evitar solapamientos por defecto
    ancho_def = (t_max - t_min) / (num_tramos + 2)
    
    for i in range(int(num_tramos)):
        ini_def = round(max(t_min, -2.0 + i * 1.5), 1)
        fin_def = round(min(t_max, ini_def + 1.5), 1)
        with st.sidebar.expander(f"Tramo {i + 1}", expanded=(i == 0)):
            tipo_i = st.selectbox(f"Señal tramo {i + 1}:", opciones_senales, index=i % len(opciones_senales), key=f"tipo_f_{i}")
            es_unit_i = tipo_i in ["Escalón Unitario u(t)", "Impulso Unitario δ(t)", "Función Signo sgn(t)"]
            amp_i = 1.0 if es_unit_i else st.slider(f"Amplitud A_{i + 1}", 0.5, 5.0, 1.0, step=0.5, key=f"amp_f_{i}")
            t0_i = st.slider(f"Desplazamiento t0_{i + 1}", -4.0, 4.0, 0.0, step=0.5, key=f"t0_f_{i}")
            rango_i = st.slider(
                f"Intervalo [a, b] tramo {i + 1}",
                float(t_min), float(t_max),
                (float(ini_def), float(fin_def)),
                step=0.5,
                key=f"rango_f_{i}"
            )
            a_i, b_i = rango_i
            mask = (t >= a_i) & (t <= b_i)
            senal_tramo = generar_funcion(tipo_i, t, amp_i, t0_i)
            f_t = np.where(mask, senal_tramo, f_t)
            tramos_f.append((tipo_i, amp_i, t0_i, a_i, b_i))
            
    label_f = f"f(t): Por intervalos ({int(num_tramos)} tramos)"

# --- PARÁMETROS PARA g(t) ---
st.sidebar.header("🟢 Configuración de g(t) [Señal Móvil]")
tipo_g = st.sidebar.selectbox("Función g(t):", opciones_senales, index=0)
es_unitaria_g = tipo_g in ["Escalón Unitario u(t)", "Impulso Unitario δ(t)", "Función Signo sgn(t)"]

amp_g = 1.0 if es_unitaria_g else st.sidebar.slider("Amplitud A_g", 0.5, 5.0, 1.0, step=0.5)
t0_g = st.sidebar.slider("Desplazamiento t0 para g(t - t0)", -4.0, 4.0, 0.0, step=0.5)
g_t = generar_funcion(tipo_g, t, amp_g, t0_g)


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
        t_low, t_mid, t_high = f"{t_low_num:.1f}", f"{t_mid_num:.1f}", f"{t_high_num:.1f}"
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
        t1_num, t2_num = -1 + shift_total, 1 + shift_total
        t1, t2 = f"{t1_num:.1f}", f"{t2_num:.1f}"
        exp1, exp2 = fmt_sub(t1_num), fmt_sub(t2_num)
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
        t1_num, t2_num = -1 + shift_total, 1 + shift_total
        t1, t2 = f"{t1_num:.1f}", f"{t2_num:.1f}"
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
    ax[0].plot(t, f_t, label=label_f, color="blue")
    ax[0].grid(True); ax[0].legend(loc="upper right")
    
    ax[1].plot(t, g_t, label=f"g(t): {tipo_g} (A={amp_g}, t0={t0_g})", color="green")
    ax[1].grid(True); ax[1].legend(loc="upper right")
    
    ax[2].plot(t, res, label=lbl, color="red")
    ax[2].grid(True); ax[2].legend(loc="upper right")
    
    st.pyplot(fig)

with tab2:
    st.subheader("Convolución Continua y Desplazamiento Interactivo")
    st.latex(r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \, g(t - \tau) \, d\tau")
    
    # Sección analítica
    st.markdown("### 📝 Expresión Analítica")
    if modo_f == "Función Única":
        latex_eq, nota = obtener_intervalos_analiticos(tipo_f, tipo_g, amp_f, amp_g, t0_f, t0_g)
        st.latex(latex_eq)
        st.caption(f"📌 **Nota de análisis:** {nota}")
    else:
        filas = []
        for (tipo_i, amp_i, t0_i, a_i, b_i) in tramos_f:
            expr_i = expresion_tramo_latex(tipo_i, amp_i, t0_i)
            filas.append(f"{expr_i}, & {a_i:.1f} \\le t \\le {b_i:.1f}")
        filas.append(r"0, & \text{en otro caso}")
        latex_f_tramos = r"f(t) = \begin{cases} " + r" \\ ".join(filas) + r" \end{cases}"
        st.latex(latex_f_tramos)
        st.caption("📌 **Nota de análisis:** La convolución se calcula numéricamente integrando el producto de la señal por intervalos $f(\\tau)$ con la señal móvil $g(t - \\tau)$.")

    st.markdown("---")
    st.markdown("### 🎬 Señal Fija $f(\\tau)$ vs. Señal Móvil $g(t - \\tau)$")
    
    # Control interactivo para mover la señal g(t - tau)
    t_actual = st.slider(
        "Desplaza el instante t para mover la señal g(t - τ):",
        float(t_min), float(t_max), 0.0, step=0.1
    )
    
    # Señal móvil g(t_actual - tau) evaluada sobre el eje tau = t
    g_movil = generar_funcion(tipo_g, t_actual - t, amp_g, t0_g)
    producto_inst = f_t * g_movil
    area_actual = np.trapezoid(producto_inst, t) if hasattr(np, "trapezoid") else np.trapz(producto_inst, t)
    
    # Convolución completa alineada exactamente con el eje t
    t_conv_full = np.linspace(2 * t_min, 2 * t_max, 2 * len(t) - 1)
    conv_full = np.convolve(f_t, g_t, mode='full') * dt
    conv_res = np.interp(t, t_conv_full, conv_full)
    
    fig_conv, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Gráfica superior: Señal fija, señal móvil y área de traslape
    ax1.plot(t, f_t, label="Señal fija f(τ)", color="blue", lw=2)
    ax1.plot(t, g_movil, label=f"Señal móvil g({t_actual:.1f} - τ)", color="green", lw=2, linestyle="--")
    ax1.fill_between(t, 0, producto_inst, color="orange", alpha=0.35, label="Área f(τ)·g(t - τ)")
    ax1.axvline(t_actual, color="red", linestyle=":", alpha=0.7)
    ax1.set_title(f"Traslape en t = {t_actual:.1f}  |  Integral actual ≈ {area_actual:.2f}")
    ax1.set_ylabel("Amplitud")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    
    # Gráfica inferior: Resultado de la convolución con el punto actual marcado
    ax2.plot(t, conv_res, label="(f * g)(t)", color="purple", lw=2)
    ax2.axvline(t_actual, color="red", linestyle=":", alpha=0.7)
    ax2.plot(t_actual, area_actual, "ro", markersize=8, label=f"(f * g)({t_actual:.1f}) = {area_actual:.2f}")
    ax2.set_title("Resultado de la Convolución (f * g)(t)")
    ax2.set_xlabel("t / τ")
    ax2.set_ylabel("(f * g)(t)")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    st.pyplot(fig_conv)

with tab3:
    st.subheader("Verificación de Autoría")
    st.info("Código desarrollado por Jhonatan Ramírez para la materia de Matemática Avanzada (EPN).")
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
