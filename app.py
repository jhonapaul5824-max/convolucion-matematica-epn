import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Laboratorio de Convolución · EPN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores profesional
COLOR_NAVY = "#1E3A8A"
COLOR_F = "#2563EB"          # Azul cobalto para f(t)
COLOR_G = "#059669"          # Verde esmeralda para g(t)
COLOR_OPS = "#DC2626"        # Carmesí para operaciones básicas
COLOR_CONV = "#7C3AED"       # Violeta para (f * g)(t)
COLOR_AREA = "#F59E0B"       # Ámbar para área de traslape
COLOR_CURSOR = "#E11D48"     # Rosa intenso para línea de instante t

# Estilos globales (fuentes modernas y componentes limpios)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
    background-color: #FFFFFF;
}

/* Ajuste de márgenes principales */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

/* Tipografía de encabezados */
h1, h2, h3, h4 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #0F172A;
    letter-spacing: -0.02em;
    font-weight: 700 !important;
}

p, label, .stRadio label, .stCheckbox label, .stSelectbox label, .stSlider label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #334155;
}

/* Barra lateral limpia y bien espaciada */
section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}

/* Encabezado principal */
.hero-header {
    background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
    border: 1px solid #E2E8F0;
    border-left: 4px solid #1E3A8A;
    border-radius: 10px;
    padding: 1.4rem 1.75rem;
    margin-bottom: 1.5rem;
}

.hero-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1E3A8A;
    margin-bottom: 0.35rem;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.025em;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 0.94rem;
    color: #475569;
    margin: 0;
    line-height: 1.5;
}

/* Tarjeta de autor en el sidebar */
.author-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.95rem 1.05rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

.author-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 0.35rem;
}

.author-name {
    font-size: 0.98rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.15rem;
}

.author-meta {
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 0.55rem;
}

.author-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    color: #1E3A8A;
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
}

/* Títulos de sección del sidebar */
.sidebar-section {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #1E3A8A;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0.35rem;
    margin-top: 1.15rem;
    margin-bottom: 0.75rem;
}

.sidebar-section.signal-f { color: #1D4ED8; border-bottom-color: #BFDBFE; }
.sidebar-section.signal-g { color: #047857; border-bottom-color: #A7F3D0; }

/* Subtítulos dentro de pestañas */
.section-heading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0.25rem 0 0.75rem 0;
    letter-spacing: -0.015em;
}

/* Tarjeta para notas de análisis */
.note-box {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #2563EB;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.88rem;
    color: #334155;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}

.note-box strong {
    color: #0F172A;
}

/* Pestañas refinadas */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1.25rem;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #64748B;
    padding: 0.6rem 1.1rem;
    border-radius: 8px 8px 0 0;
}

.stTabs [aria-selected="true"] {
    color: #1E3A8A !important;
    border-bottom-color: #1E3A8A !important;
}

/* Código fuente con fuente monospace moderna */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

hr {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.25rem 0;
}
</style>
""", unsafe_allow_html=True)

# Configuración estética global de Matplotlib
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#334155",
    "axes.titlecolor": "#0F172A",
    "axes.titleweight": "semibold",
    "axes.titlesize": 11,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "legend.frameon": True,
    "legend.facecolor": "#FFFFFF",
    "legend.edgecolor": "#E2E8F0",
    "legend.framealpha": 0.92,
    "legend.fontsize": 9,
})

# Encabezado principal
st.markdown("""
<div class="hero-header">
    <div class="hero-kicker">Escuela Politécnica Nacional · Matemática Avanzada</div>
    <h1 class="hero-title">Laboratorio de Operaciones con Funciones y Convolución</h1>
    <p class="hero-subtitle">
        Análisis interactivo de señales continuas y por intervalos para <strong>f(t)</strong> y <strong>g(t)</strong>,
        con evaluación analítica y generación progresiva de la integral de convolución.
    </p>
</div>
""", unsafe_allow_html=True)

# Tarjeta de autoría en la barra lateral
st.sidebar.markdown("""
<div class="author-card">
    <div class="author-label">Autoría del Proyecto</div>
    <div class="author-name">Jhonatan Ramírez</div>
    <div class="author-meta">Escuela Politécnica Nacional (EPN)</div>
    <span class="author-badge">EPN-MATH-ADV-2026-JR</span>
</div>
""", unsafe_allow_html=True)

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
st.sidebar.markdown('<div class="sidebar-section">Rango de Tiempo</div>', unsafe_allow_html=True)
t_min, t_max = st.sidebar.slider("Dominio del eje t", -10.0, 10.0, (-6.0, 6.0), step=0.5)
mostrar_marcas_t = st.sidebar.checkbox("Mostrar marcas de inicio/fin en el eje t", value=True)
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
        return amp * np.where((tau >= -ancho / 2) & (tau <= ancho / 2), 1.0 / ancho, 0.0)
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


def evaluar_tramos(tramos, eje_t):
    """Evalúa una función definida por intervalos sobre cualquier eje temporal (t o t - tau)."""
    y = np.zeros_like(eje_t)
    for (tipo_i, amp_i, t0_i, a_i, b_i) in tramos:
        mask = (eje_t >= a_i) & (eje_t <= b_i)
        y = np.where(mask, generar_funcion(tipo_i, eje_t, amp_i, t0_i), y)
    return y


def configurar_eje_t(ax, etiqueta_x="t"):
    """Configura un estilo limpio y técnico para los ejes de Matplotlib."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which="major", linestyle="-", color="#E2E8F0", linewidth=0.8)
    ax.grid(True, which="minor", linestyle=":", color="#F1F5F9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="both", labelbottom=True)
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel(etiqueta_x, fontsize=9.5, fontweight="medium")


def obtener_puntos_criticos(cfg):
    """Obtiene los valores de t donde inicia o termina cada tramo o pulso."""
    puntos = set()
    if cfg["modo"] == "Por Intervalos (hasta 4 tramos)":
        for (_, _, _, a_i, b_i) in cfg["tramos"]:
            puntos.add(a_i)
            puntos.add(b_i)
    else:
        tipo = cfg["tipo"]
        t0 = cfg["t0"]
        if tipo in ["Pulso Rectangular / Compuerta", "Pulso Triangular"]:
            puntos.update([t0 - 1.0, t0 + 1.0])
        elif tipo in [
            "Exponencial Decreciente",
            "Exponencial Creciente (t ≤ t0)",
            "Escalón Unitario u(t)",
            "Impulso Unitario δ(t)",
            "Rampa r(t)",
            "Función Signo sgn(t)"
        ]:
            puntos.add(t0)
    return sorted(p for p in puntos if t_min <= p <= t_max)


def dibujar_marcas_t(ax, puntos, color, desplazamiento_y=0.05):
    """Dibuja líneas verticales suaves y etiquetas con el valor exacto de t en los bordes."""
    if not mostrar_marcas_t or not puntos:
        return
    y_min, y_max = ax.get_ylim()
    y_pos = y_min + desplazamiento_y * (y_max - y_min)
    for p in puntos:
        ax.axvline(p, color=color, linestyle="--", lw=1.0, alpha=0.4)
        ax.text(
            p, y_pos, f" {p:g} ",
            color=color, fontsize=8, fontweight="semibold",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor=color, lw=0.8, alpha=0.85)
        )


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
    """Genera una representación LaTeX corta para un tramo."""
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


def latex_funcion_tramos(nombre_fn, tramos):
    """Construye la expresión LaTeX por partes para f(t) o g(t)."""
    filas = []
    for (tipo_i, amp_i, t0_i, a_i, b_i) in tramos:
        expr_i = expresion_tramo_latex(tipo_i, amp_i, t0_i)
        filas.append(f"{expr_i}, & {a_i:.1f} \\le t \\le {b_i:.1f}")
    filas.append(r"0, & \text{en otro caso}")
    return f"{nombre_fn}(t) = \\begin{{cases}} " + r" \\ ".join(filas) + r" \end{cases}"


def selector_amplitud(tipo, label, key_prefix, container=st.sidebar):
    """Permite amplitud negativa (-5 a 5) o invertir signo (-1/+1) en señales unitarias."""
    if tipo in senales_unitarias:
        negativa = container.checkbox(f"Invertir signo (−) {label}", value=False, key=f"{key_prefix}_neg")
        return -1.0 if negativa else 1.0
    else:
        return container.slider(f"Amplitud {label}", -5.0, 5.0, 1.0, step=0.5, key=f"{key_prefix}_amp")


def configurar_senal_sidebar(nombre, clase_css, prefijo):
    """Crea los controles en la barra lateral para una señal (única o hasta 4 tramos)."""
    rol = "Señal Fija" if nombre == "f" else "Señal Móvil"
    st.sidebar.markdown(
        f'<div class="sidebar-section {clase_css}">Configuración de {nombre}(t) · {rol}</div>',
        unsafe_allow_html=True
    )
    modo = st.sidebar.radio(
        f"Modo de definición para {nombre}(t):",
        ["Función Única", "Por Intervalos (hasta 4 tramos)"],
        key=f"modo_{prefijo}"
    )

    tramos = []
    if modo == "Función Única":
        tipo = st.sidebar.selectbox(f"Tipo de señal {nombre}(t):", opciones_senales, index=0, key=f"tipo_{prefijo}")
        amp = selector_amplitud(tipo, f"A_{nombre}", f"{prefijo}_unica")
        t0 = st.sidebar.slider(f"Desplazamiento t₀ en {nombre}(t − t₀)", -4.0, 4.0, 0.0, step=0.5, key=f"t0_{prefijo}")
        y_t = generar_funcion(tipo, t, amp, t0)
        label = f"{nombre}(t): {tipo} (A={amp:g}, t₀={t0:g})"
        return {
            "modo": modo, "tipo": tipo, "amp": amp, "t0": t0,
            "tramos": tramos, "y": y_t, "label": label
        }
    else:
        num_tramos = st.sidebar.number_input(
            f"Número de intervalos para {nombre}(t) (máx. 4):",
            min_value=1, max_value=4, value=2, step=1,
            key=f"num_tramos_{prefijo}"
        )
        for i in range(int(num_tramos)):
            ini_def = round(max(t_min, -2.0 + i * 1.5), 1)
            fin_def = round(min(t_max, ini_def + 1.5), 1)
            with st.sidebar.expander(f"{nombre}(t) — Tramo {i + 1}", expanded=(i == 0)):
                tipo_i = st.selectbox(
                    f"Señal tramo {i + 1}:",
                    opciones_senales,
                    index=i % len(opciones_senales),
                    key=f"tipo_{prefijo}_{i}"
                )
                amp_i = selector_amplitud(tipo_i, f"A_{nombre}{i + 1}", f"{prefijo}_tramo_{i}", container=st)
                t0_i = st.slider(
                    f"Desplazamiento t₀ ({i + 1})",
                    -4.0, 4.0, 0.0, step=0.5,
                    key=f"t0_{prefijo}_{i}"
                )
                rango_i = st.slider(
                    f"Intervalo [a, b] tramo {i + 1}",
                    float(t_min), float(t_max),
                    (float(ini_def), float(fin_def)),
                    step=0.5,
                    key=f"rango_{prefijo}_{i}"
                )
                tramos.append((tipo_i, amp_i, t0_i, rango_i[0], rango_i[1]))

        y_t = evaluar_tramos(tramos, t)
        label = f"{nombre}(t): Definida por intervalos ({int(num_tramos)} tramos)"
        return {
            "modo": modo, "tipo": None, "amp": None, "t0": None,
            "tramos": tramos, "y": y_t, "label": label
        }


# Configurar f(t) [Señal Fija] y g(t) [Señal Móvil]
cfg_f = configurar_senal_sidebar("f", "signal-f", "f")
cfg_g = configurar_senal_sidebar("g", "signal-g", "g")

f_t = cfg_f["y"]
g_t = cfg_g["y"]


def evaluar_g_movil(t_instante):
    """Evalúa g(t_instante - tau) tanto en modo único como por intervalos."""
    eje_desplazado = t_instante - t
    if cfg_g["modo"] == "Función Única":
        return generar_funcion(cfg_g["tipo"], eje_desplazado, cfg_g["amp"], cfg_g["t0"])
    else:
        return evaluar_tramos(cfg_g["tramos"], eje_desplazado)


def obtener_intervalos_analiticos(f_name, g_name, A1, A2, shift1, shift2):
    shift_total = shift1 + shift2
    A_total = A1 * A2
    s_tot = f"{shift_total:.1f}"
    str_amp_tot = fmt_amp(A_total)

    # 1. Pulso Rectangular * Pulso Rectangular
    if "Compuerta" in f_name and "Compuerta" in g_name:
        t_low_num, t_mid_num, t_high_num = -2 + shift_total, shift_total, 2 + shift_total
        t_low, t_mid, t_high = f"{t_low_num:.1f}", f"{t_mid_num:.1f}", f"{t_high_num:.1f}"
        expr_inc = fmt_sub(t_low_num)
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + t_low + r" \\ "
            + str_amp_tot + r"(" + expr_inc + r"), & " + t_low + r" \le t < " + t_mid + r" \\ "
            + str_amp_tot + r"(" + t_high + r" - t), & " + t_mid + r" \le t \le " + t_high + r" \\ "
            r"0, & t > " + t_high + r" \end{cases}"
        )
        return eq, f"Triángulo de amplitud pico {A_total * 2:.1f} centrado en t = {s_tot} sobre el intervalo [{t_low}, {t_high}]."

    # 2. Exponencial Decreciente * Escalón
    elif ("Exponencial Decreciente" in f_name and "Escalón" in g_name) or ("Escalón" in f_name and "Exponencial Decreciente" in g_name):
        exp_t = fmt_sub(shift_total)
        eq = (
            r"(f * g)(t) = \begin{cases} "
            r"0, & t < " + s_tot + r" \\ "
            + str_amp_tot + r"(1 - e^{-(" + exp_t + r")}), & t \ge " + s_tot + r" \end{cases}"
        )
        return eq, f"Respuesta exponencial acotada iniciada en t = {s_tot}."

    # 3. Pulso Rectangular * Exponencial Decreciente
    elif ("Compuerta" in f_name and "Exponencial Decreciente" in g_name) or ("Exponencial Decreciente" in f_name and "Compuerta" in g_name):
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
            + str_amp_tot + r"(" + rampa_expr + r"), & t \ge " + s_tot + r" \end{cases}"
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
            + str_amp_tot + r"(" + expr_lin + r"), & " + t1 + r" \le t < " + t2 + r" \\ "
            f"{2 * A_total:.1f}, & t \ge " + t2 + r" \end{cases}"
        )
        return eq, f"Cambio lineal entre {t1} y {t2}, manteniéndose constante en {2 * A_total:.1f} posteriormente."

    # 6. Impulso Unitario
    elif "Impulso" in f_name or "Impulso" in g_name:
        otra = g_name if "Impulso" in f_name else f_name
        exp_imp = fmt_sub(shift_total)
        eq = r"(f * g)(t) = " + str_amp_tot + r"\text{" + otra.split()[0] + r"}(" + exp_imp + r")"
        return eq, f"Propiedad de tamizado: la función se desplaza íntegramente a t = {s_tot}."

    # 7. Caso General
    else:
        eq = r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \, g(t - \tau) \, d\tau"
        return eq, f"Evaluación numérica de la integral de convolución con desplazamiento neto en t = {s_tot}."


def generar_figura_convolucion(t_actual, conv_res):
    """Genera la figura estática de dos paneles para el modo de control manual."""
    g_movil = evaluar_g_movil(t_actual)
    producto_inst = f_t * g_movil
    area_actual = np.trapezoid(producto_inst, t) if hasattr(np, "trapezoid") else np.trapz(producto_inst, t)

    fig_conv, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), dpi=120)
    fig_conv.patch.set_facecolor("#FFFFFF")

    # Gráfica superior: Señal fija, señal móvil y área de traslape
    ax1.plot(t, f_t, label="Señal fija f(τ)", color=COLOR_F, lw=2.2)
    ax1.plot(t, g_movil, label=f"Señal móvil g({t_actual:.1f} − τ)", color=COLOR_G, lw=2.2, linestyle="--")
    ax1.fill_between(t, 0, producto_inst, color=COLOR_AREA, alpha=0.28, label="Área f(τ)·g(t − τ)")
    ax1.axhline(0, color="#64748B", lw=0.9, alpha=0.6)
    ax1.axvline(t_actual, color=COLOR_CURSOR, linestyle=":", lw=1.4, alpha=0.8)
    ax1.set_title(f"Traslape en t = {t_actual:.1f}   |   Integral actual ≈ {area_actual:.2f}", pad=10)
    ax1.set_ylabel("Amplitud")
    configurar_eje_t(ax1, etiqueta_x="τ")
    dibujar_marcas_t(ax1, obtener_puntos_criticos(cfg_f), color=COLOR_F, desplazamiento_y=0.05)
    ax1.legend(loc="upper right")

    # Gráfica inferior: Convolución progresiva
    mask_progreso = t <= t_actual
    ax2.plot(t[mask_progreso], conv_res[mask_progreso], label="(f * g)(t)", color=COLOR_CONV, lw=2.5)
    ax2.axhline(0, color="#64748B", lw=0.9, alpha=0.6)
    ax2.axvline(t_actual, color=COLOR_CURSOR, linestyle=":", lw=1.4, alpha=0.8)
    ax2.plot(
        t_actual, area_actual, "o",
        color=COLOR_CURSOR, markersize=7.5,
        markeredgecolor="#FFFFFF", markeredgewidth=1.2,
        label=f"(f * g)({t_actual:.1f}) = {area_actual:.2f}"
    )

    y_min_c, y_max_c = float(np.min(conv_res)), float(np.max(conv_res))
    margen_c = max(0.5, 0.15 * (y_max_c - y_min_c))
    ax2.set_ylim(y_min_c - margen_c, y_max_c + margen_c)

    ax2.set_title("Resultado de la Convolución (f * g)(t)", pad=10)
    ax2.set_ylabel("(f * g)(t)")
    configurar_eje_t(ax2, etiqueta_x="t")
    ax2.legend(loc="upper right")

    plt.tight_layout(pad=1.8)
    return fig_conv


def generar_animacion_plotly(valores_t, conv_res, fps=20):
    """Genera una animación fluida en el navegador usando cuadros nativos de Plotly."""
    tiene_impulso = (
        (cfg_f["tipo"] and "Impulso" in cfg_f["tipo"])
        or (cfg_g["tipo"] and "Impulso" in cfg_g["tipo"])
        or any("Impulso" in tramo[0] for tramo in cfg_f["tramos"] + cfg_g["tramos"])
    )
    salto = 1 if tiene_impulso else 2
    t_vis = t[::salto]
    f_vis = f_t[::salto]

    frames_data = []
    y1_min, y1_max = float(np.min(f_t)), float(np.max(f_t))

    for t_val in valores_t:
        g_movil = evaluar_g_movil(t_val)
        producto_inst = f_t * g_movil
        area_actual = float(
            np.trapezoid(producto_inst, t) if hasattr(np, "trapezoid") else np.trapz(producto_inst, t)
        )

        y1_min = min(y1_min, float(np.min(g_movil)), float(np.min(producto_inst)))
        y1_max = max(y1_max, float(np.max(g_movil)), float(np.max(producto_inst)))

        mask_progreso = t <= t_val
        frames_data.append((
            float(t_val),
            g_movil[::salto],
            producto_inst[::salto],
            t[mask_progreso][::salto],
            conv_res[mask_progreso][::salto],
            area_actual,
        ))

    margen_1 = max(0.5, 0.15 * (y1_max - y1_min))
    y1_lim = [y1_min - margen_1, y1_max + margen_1]

    y2_min, y2_max = float(np.min(conv_res)), float(np.max(conv_res))
    margen_2 = max(0.5, 0.15 * (y2_max - y2_min))
    y2_lim = [y2_min - margen_2, y2_max + margen_2]

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.16,
        subplot_titles=(
            "Traslape: Señal fija f(τ) vs. Señal móvil g(t − τ)",
            "Resultado de la Convolución (f * g)(t)"
        )
    )

    # Estilizar subtítulos antes de añadir anotaciones de líneas verticales
    fig.update_annotations(font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color="#0F172A"))

    t0_val, g0, prod0, t_prog0, conv_prog0, area0 = frames_data[0]

    # Traza 0 (estática): f(τ)
    fig.add_trace(go.Scatter(
        x=t_vis, y=f_vis, mode="lines",
        name="Señal fija f(τ)", line=dict(color=COLOR_F, width=2.5)
    ), row=1, col=1)

    # Traza 1 (dinámica): g(t - τ)
    fig.add_trace(go.Scatter(
        x=t_vis, y=g0, mode="lines",
        name="Señal móvil g(t − τ)", line=dict(color=COLOR_G, width=2.5, dash="dash")
    ), row=1, col=1)

    # Traza 2 (dinámica): Área de traslape
    fig.add_trace(go.Scatter(
        x=t_vis, y=prod0, mode="lines",
        name="Área f(τ)·g(t − τ)", fill="tozeroy",
        fillcolor="rgba(245, 158, 11, 0.25)",
        line=dict(color="rgba(217, 119, 6, 0.65)", width=1.2)
    ), row=1, col=1)

    # Traza 3 (dinámica): Línea vertical en t_actual (panel superior)
    fig.add_trace(go.Scatter(
        x=[t0_val, t0_val], y=y1_lim, mode="lines",
        line=dict(color=COLOR_CURSOR, width=1.5, dash="dot"), showlegend=False
    ), row=1, col=1)

    # Traza 4 (dinámica): Convolución progresiva
    fig.add_trace(go.Scatter(
        x=t_prog0, y=conv_prog0, mode="lines",
        name="(f * g)(t)", line=dict(color=COLOR_CONV, width=3)
    ), row=2, col=1)

    # Traza 5 (dinámica): Punto indicador con valor en vivo
    fig.add_trace(go.Scatter(
        x=[t0_val], y=[area0], mode="markers+text",
        name="Valor actual",
        marker=dict(color=COLOR_CURSOR, size=9, line=dict(color="#FFFFFF", width=1.5)),
        text=[f"  t = {t0_val:.1f} · Área = {area0:.2f}"],
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#0F172A"),
        textposition="top right"
    ), row=2, col=1)

    # Traza 6 (dinámica): Línea vertical en t_actual (panel inferior)
    fig.add_trace(go.Scatter(
        x=[t0_val, t0_val], y=y2_lim, mode="lines",
        line=dict(color=COLOR_CURSOR, width=1.5, dash="dot"), showlegend=False
    ), row=2, col=1)

    # Marcas de puntos críticos de f(τ)
    if mostrar_marcas_t:
        for p in obtener_puntos_criticos(cfg_f):
            fig.add_vline(
                x=p, line_dash="dash", line_color="rgba(37, 99, 235, 0.35)",
                line_width=1, row=1, col=1,
                annotation_text=f"{p:g}",
                annotation_position="bottom right",
                annotation_font=dict(family="JetBrains Mono, monospace", size=10, color=COLOR_F)
            )

    duracion_ms = int(1000 / fps)
    frames = []
    slider_steps = []

    for idx, (t_val, g_v, prod_v, t_prog, conv_prog, area_v) in enumerate(frames_data):
        nombre_frame = str(idx)
        frames.append(go.Frame(
            name=nombre_frame,
            traces=[1, 2, 3, 4, 5, 6],
            data=[
                go.Scatter(x=t_vis, y=g_v),
                go.Scatter(x=t_vis, y=prod_v),
                go.Scatter(x=[t_val, t_val], y=y1_lim),
                go.Scatter(x=t_prog, y=conv_prog),
                go.Scatter(x=[t_val], y=[area_v], text=[f"  t = {t_val:.1f} · Área = {area_v:.2f}"]),
                go.Scatter(x=[t_val, t_val], y=y2_lim),
            ]
        ))
        slider_steps.append({
            "method": "animate",
            "label": f"{t_val:.1f}",
            "args": [
                [nombre_frame],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
            ]
        })

    fig.frames = frames

    fig.update_layout(
        height=740,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#334155"),
        margin=dict(l=55, r=30, t=75, b=95),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.06,
            xanchor="center", x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11.5)
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": -0.09,
            "xanchor": "left", "yanchor": "top",
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "bgcolor": "#F8FAFC",
            "bordercolor": "#CBD5E1",
            "font": dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#0F172A"),
            "buttons": [
                {
                    "label": "▶ Reproducir",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": duracion_ms, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }
                    ]
                },
                {
                    "label": "❚❚ Pausar",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                }
            ]
        }],
        sliders=[{
            "active": 0,
            "x": 0.23, "len": 0.77,
            "y": -0.09, "yanchor": "top",
            "bgcolor": "#F1F5F9",
            "bordercolor": "#CBD5E1",
            "activebgcolor": COLOR_NAVY,
            "font": dict(family="JetBrains Mono, monospace", size=10, color="#475569"),
            "currentvalue": {
                "prefix": "Instante t = ",
                "visible": True,
                "xanchor": "right",
                "font": dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#0F172A")
            },
            "pad": {"b": 10, "t": 10},
            "steps": slider_steps
        }]
    )

    estilo_eje_x = dict(
        range=[t_min, t_max], dtick=1.0,
        showgrid=True, gridcolor="#E2E8F0",
        linecolor="#CBD5E1", zeroline=True, zerolinecolor="#94A3B8"
    )
    estilo_eje_y = dict(
        showgrid=True, gridcolor="#E2E8F0",
        linecolor="#CBD5E1", zeroline=True, zerolinecolor="#94A3B8"
    )

    fig.update_xaxes(title_text="τ", row=1, col=1, **estilo_eje_x)
    fig.update_xaxes(title_text="t", row=2, col=1, **estilo_eje_x)
    fig.update_yaxes(range=y1_lim, title_text="Amplitud", row=1, col=1, **estilo_eje_y)
    fig.update_yaxes(range=y2_lim, title_text="(f * g)(t)", row=2, col=1, **estilo_eje_y)

    return fig


tab1, tab2, tab3 = st.tabs(["Operaciones Básicas", "Convolución e Intervalos", "Código Fuente"])

with tab1:
    st.markdown('<div class="section-heading">Suma, Resta y Multiplicación de Señales</div>', unsafe_allow_html=True)
    operacion = st.radio(
        "Selecciona la operación a evaluar:",
        ["Suma: f(t) + g(t)", "Resta: f(t) − g(t)", "Producto: f(t) · g(t)"],
        horizontal=True
    )

    if operacion == "Suma: f(t) + g(t)":
        res = f_t + g_t
        lbl = "f(t) + g(t)"
    elif operacion == "Resta: f(t) − g(t)":
        res = f_t - g_t
        lbl = "f(t) − g(t)"
    else:
        res = f_t * g_t
        lbl = "f(t) · g(t)"

    fig, ax = plt.subplots(3, 1, figsize=(10, 8.6), dpi=120)
    fig.patch.set_facecolor("#FFFFFF")

    ax[0].plot(t, f_t, label=cfg_f["label"], color=COLOR_F, lw=2.2)
    ax[0].axhline(0, color="#64748B", lw=0.9, alpha=0.6)
    ax[0].set_ylabel("f(t)")
    configurar_eje_t(ax[0], etiqueta_x="t")
    dibujar_marcas_t(ax[0], obtener_puntos_criticos(cfg_f), color=COLOR_F)
    ax[0].legend(loc="upper right")

    ax[1].plot(t, g_t, label=cfg_g["label"], color=COLOR_G, lw=2.2)
    ax[1].axhline(0, color="#64748B", lw=0.9, alpha=0.6)
    ax[1].set_ylabel("g(t)")
    configurar_eje_t(ax[1], etiqueta_x="t")
    dibujar_marcas_t(ax[1], obtener_puntos_criticos(cfg_g), color=COLOR_G)
    ax[1].legend(loc="upper right")

    ax[2].plot(t, res, label=lbl, color=COLOR_OPS, lw=2.3)
    ax[2].axhline(0, color="#64748B", lw=0.9, alpha=0.6)
    ax[2].set_ylabel("Resultado")
    configurar_eje_t(ax[2], etiqueta_x="t")
    ax[2].legend(loc="upper right")

    plt.tight_layout(pad=1.6)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.markdown('<div class="section-heading">Integral de Convolución Continua</div>', unsafe_allow_html=True)
    st.latex(r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \, g(t - \tau) \, d\tau")

    # Sección analítica
    st.markdown('<div class="section-heading">Expresión Analítica</div>', unsafe_allow_html=True)
    if cfg_f["modo"] == "Función Única" and cfg_g["modo"] == "Función Única":
        latex_eq, nota = obtener_intervalos_analiticos(
            cfg_f["tipo"], cfg_g["tipo"], cfg_f["amp"], cfg_g["amp"], cfg_f["t0"], cfg_g["t0"]
        )
        st.latex(latex_eq)
        st.markdown(f'<div class="note-box"><strong>Nota de análisis:</strong> {nota}</div>', unsafe_allow_html=True)
    else:
        if cfg_f["modo"] != "Función Única":
            st.latex(latex_funcion_tramos("f", cfg_f["tramos"]))
        if cfg_g["modo"] != "Función Única":
            st.latex(latex_funcion_tramos("g", cfg_g["tramos"]))
        st.markdown(
            '<div class="note-box"><strong>Nota de análisis:</strong> La convolución se evalúa numéricamente integrando el producto de la señal fija <em>f(τ)</em> con la señal móvil <em>g(t − τ)</em>.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="section-heading">Señal Fija f(τ) vs. Señal Móvil g(t − τ)</div>', unsafe_allow_html=True)

    # Convolución completa alineada exactamente con el eje t
    t_conv_full = np.linspace(2 * t_min, 2 * t_max, 2 * len(t) - 1)
    conv_full = np.convolve(f_t, g_t, mode="full") * dt
    conv_res = np.interp(t, t_conv_full, conv_full)

    # Selector de modo de desplazamiento
    modo_anim = st.radio(
        "Modo de desplazamiento de la señal móvil:",
        ["Control Manual", "Animación Automática"],
        horizontal=True
    )

    if modo_anim == "Control Manual":
        t_actual = st.slider(
            "Desplaza el instante t para mover la señal g(t − τ):",
            float(t_min), float(t_max), 0.0, step=0.1
        )
        fig_conv = generar_figura_convolucion(t_actual, conv_res)
        st.pyplot(fig_conv)
        plt.close(fig_conv)
    else:
        col1, col2 = st.columns(2)
        with col1:
            paso_anim = st.slider("Resolución de paso Δt (menor = más suave):", 0.05, 0.50, 0.15, step=0.05)
        with col2:
            fps = st.slider("Velocidad (cuadros por segundo · FPS):", 5, 40, 20, step=5)

        n_frames = max(2, int(round((t_max - t_min) / paso_anim)) + 1)
        valores_t = np.linspace(t_min, t_max, n_frames)

        fig_anim = generar_animacion_plotly(valores_t, conv_res, fps=fps)
        st.plotly_chart(fig_anim, use_container_width=True)

with tab3:
    st.markdown('<div class="section-heading">Verificación de Autoría y Código Fuente</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note-box"><strong>Registro de autoría:</strong> Código desarrollado por <strong>Jhonatan Ramírez</strong> para la asignatura de Matemática Avanzada en la Escuela Politécnica Nacional (EPN).</div>',
        unsafe_allow_html=True
    )
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
