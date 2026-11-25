import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
> *Aplicación interactiva con funciones por intervalos para $f(t)$ y $g(t)$, señales positivas/negativas y generación progresiva de la convolución.*
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
    """Muestra valores de t cada 1 unidad y cuadrícula fina cada 0.5 en cada subgráfica."""
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which="major", linestyle="-", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", which="both", labelbottom=True)
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel(etiqueta_x)


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


def dibujar_marcas_t(ax, puntos, color, desplazamiento_y=0.04):
    """Dibuja líneas verticales suaves y etiquetas con el valor exacto de t en los bordes."""
    if not mostrar_marcas_t or not puntos:
        return
    y_min, y_max = ax.get_ylim()
    y_pos = y_min + desplazamiento_y * (y_max - y_min)
    for p in puntos:
        ax.axvline(p, color=color, linestyle="--", lw=1.0, alpha=0.45)
        ax.text(
            p, y_pos, f" {p:g}",
            color=color, fontsize=8.5, fontweight="bold",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.75)
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
        negativa = container.checkbox(f"Invertir signo (-) {label}", value=False, key=f"{key_prefix}_neg")
        return -1.0 if negativa else 1.0
    else:
        return container.slider(f"Amplitud {label}", -5.0, 5.0, 1.0, step=0.5, key=f"{key_prefix}_amp")


def configurar_senal_sidebar(nombre, icono, prefijo):
    """Crea los controles en la barra lateral para una señal (única o hasta 4 tramos)."""
    st.sidebar.header(f"{icono} Configuración de {nombre}(t)")
    modo = st.sidebar.radio(
        f"Modo de {nombre}(t):",
        ["Función Única", "Por Intervalos (hasta 4 tramos)"],
        key=f"modo_{prefijo}"
    )

    tramos = []
    if modo == "Función Única":
        tipo = st.sidebar.selectbox(f"Función {nombre}(t):", opciones_senales, index=0, key=f"tipo_{prefijo}")
        amp = selector_amplitud(tipo, f"A_{nombre}", f"{prefijo}_unica")
        t0 = st.sidebar.slider(f"Desplazamiento t0 para {nombre}(t - t0)", -4.0, 4.0, 0.0, step=0.5, key=f"t0_{prefijo}")
        y_t = generar_funcion(tipo, t, amp, t0)
        label = f"{nombre}(t): {tipo} (A={amp}, t0={t0})"
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
            with st.sidebar.expander(f"{nombre}(t) - Tramo {i + 1}", expanded=(i == 0)):
                tipo_i = st.selectbox(
                    f"Señal tramo {i + 1}:",
                    opciones_senales,
                    index=i % len(opciones_senales),
                    key=f"tipo_{prefijo}_{i}"
                )
                amp_i = selector_amplitud(tipo_i, f"A_{nombre}{i + 1}", f"{prefijo}_tramo_{i}", container=st)
                t0_i = st.slider(
                    f"Desplazamiento t0_{i + 1}",
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
        label = f"{nombre}(t): Por intervalos ({int(num_tramos)} tramos)"
        return {
            "modo": modo, "tipo": None, "amp": None, "t0": None,
            "tramos": tramos, "y": y_t, "label": label
        }


# Configurar f(t) [Señal Fija] y g(t) [Señal Móvil]
cfg_f = configurar_senal_sidebar("f", "🔹", "f")
cfg_g = configurar_senal_sidebar("g", "🟢", "g")

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
        return eq, f"Cambio lineal entre {t1} y {t2}, constante en {2 * A_total:.1f} después."

    # 6. Impulso Unitario
    elif "Impulso" in f_name or "Impulso" in g_name:
        otra = g_name if "Impulso" in f_name else f_name
        exp_imp = fmt_sub(shift_total)
        eq = r"(f * g)(t) = " + str_amp_tot + r"\text{" + otra.split()[0] + r"}(" + exp_imp + r")"
        return eq, f"Propiedad de tamizado: La función se desplaza totalmente a t = {s_tot}."

    # 7. Caso General
    else:
        eq = r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \, g(t - \tau) \, d\tau"
        return eq, f"Evaluación numérica de la integral de convolución con desplazamiento neto en t = {s_tot}."


def generar_figura_convolucion(t_actual, conv_res):
    """Genera la figura estática de dos paneles para el modo de control manual."""
    g_movil = evaluar_g_movil(t_actual)
    producto_inst = f_t * g_movil
    area_actual = np.trapezoid(producto_inst, t) if hasattr(np, "trapezoid") else np.trapz(producto_inst, t)

    fig_conv, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5))

    # Gráfica superior: Señal fija, señal móvil y área de traslape
    ax1.plot(t, f_t, label="Señal fija f(τ)", color="blue", lw=2)
    ax1.plot(t, g_movil, label=f"Señal móvil g({t_actual:.1f} - τ)", color="green", lw=2, linestyle="--")
    ax1.fill_between(t, 0, producto_inst, color="orange", alpha=0.35, label="Área f(τ)·g(t - τ)")
    ax1.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax1.axvline(t_actual, color="red", linestyle=":", alpha=0.7)
    ax1.set_title(f"Traslape en t = {t_actual:.1f}  |  Integral actual ≈ {area_actual:.2f}")
    ax1.set_ylabel("Amplitud")
    configurar_eje_t(ax1, etiqueta_x="τ")
    dibujar_marcas_t(ax1, obtener_puntos_criticos(cfg_f), color="blue", desplazamiento_y=0.04)
    ax1.legend(loc="upper right")

    # Gráfica inferior: Convolución generándose progresivamente hasta t_actual
    mask_progreso = t <= t_actual
    ax2.plot(t[mask_progreso], conv_res[mask_progreso], label="(f * g)(t)", color="purple", lw=2.5)
    ax2.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax2.axvline(t_actual, color="red", linestyle=":", alpha=0.7)
    ax2.plot(t_actual, area_actual, "ro", markersize=8, label=f"(f * g)({t_actual:.1f}) = {area_actual:.2f}")

    # Fijar límites en Y para que la gráfica no salte al irse dibujando
    y_min_c, y_max_c = float(np.min(conv_res)), float(np.max(conv_res))
    margen_c = max(0.5, 0.15 * (y_max_c - y_min_c))
    ax2.set_ylim(y_min_c - margen_c, y_max_c + margen_c)

    ax2.set_title("Resultado de la Convolución (f * g)(t) [Generación en tiempo real]")
    ax2.set_ylabel("(f * g)(t)")
    configurar_eje_t(ax2, etiqueta_x="t")
    ax2.legend(loc="upper right")

    plt.tight_layout()
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

    # Límites fijos en Y para evitar saltos de escala
    margen_1 = max(0.5, 0.15 * (y1_max - y1_min))
    y1_lim = [y1_min - margen_1, y1_max + margen_1]

    y2_min, y2_max = float(np.min(conv_res)), float(np.max(conv_res))
    margen_2 = max(0.5, 0.15 * (y2_max - y2_min))
    y2_lim = [y2_min - margen_2, y2_max + margen_2]

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.14,
        subplot_titles=(
            "Traslape: Señal fija f(τ) vs. Señal móvil g(t - τ)",
            "Resultado de la Convolución (f * g)(t) [Generación fluida]"
        )
    )

    t0_val, g0, prod0, t_prog0, conv_prog0, area0 = frames_data[0]

    # Traza 0 (estática): f(τ)
    fig.add_trace(go.Scatter(
        x=t_vis, y=f_vis, mode="lines",
        name="Señal fija f(τ)", line=dict(color="blue", width=2.5)
    ), row=1, col=1)

    # Traza 1 (dinámica): g(t - τ)
    fig.add_trace(go.Scatter(
        x=t_vis, y=g0, mode="lines",
        name="Señal móvil g(t - τ)", line=dict(color="green", width=2.5, dash="dash")
    ), row=1, col=1)

    # Traza 2 (dinámica): Área de traslape
    fig.add_trace(go.Scatter(
        x=t_vis, y=prod0, mode="lines",
        name="Área f(τ)·g(t - τ)", fill="tozeroy",
        fillcolor="rgba(255, 165, 0, 0.35)",
        line=dict(color="rgba(255, 165, 0, 0.6)", width=1)
    ), row=1, col=1)

    # Traza 3 (dinámica): Línea vertical en t_actual (panel superior)
    fig.add_trace(go.Scatter(
        x=[t0_val, t0_val], y=y1_lim, mode="lines",
        line=dict(color="red", width=1.5, dash="dot"), showlegend=False
    ), row=1, col=1)

    # Traza 4 (dinámica): Convolución progresiva
    fig.add_trace(go.Scatter(
        x=t_prog0, y=conv_prog0, mode="lines",
        name="(f * g)(t)", line=dict(color="purple", width=3)
    ), row=2, col=1)

    # Traza 5 (dinámica): Punto rojo con valor en vivo
    fig.add_trace(go.Scatter(
        x=[t0_val], y=[area0], mode="markers+text",
        name="Valor actual", marker=dict(color="red", size=10),
        text=[f"  t={t0_val:.1f}, Área={area0:.2f}"],
        textposition="top right"
    ), row=2, col=1)

    # Traza 6 (dinámica): Línea vertical en t_actual (panel inferior)
    fig.add_trace(go.Scatter(
        x=[t0_val, t0_val], y=y2_lim, mode="lines",
        line=dict(color="red", width=1.5, dash="dot"), showlegend=False
    ), row=2, col=1)

    # Marcas de puntos críticos de f(τ)
    if mostrar_marcas_t:
        for p in obtener_puntos_criticos(cfg_f):
            fig.add_vline(
                x=p, line_dash="dash", line_color="rgba(0, 0, 255, 0.4)",
                line_width=1, row=1, col=1,
                annotation_text=f"{p:g}", annotation_position="bottom right"
            )

    # Construcción de cuadros y pasos del slider
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
                go.Scatter(x=[t_val], y=[area_v], text=[f"  t={t_val:.1f}, Área={area_v:.2f}"]),
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
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": -0.08,
            "xanchor": "left", "yanchor": "top",
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "buttons": [
                {
                    "label": "▶️ Reproducir",
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
                    "label": "⏸️ Pausar",
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
            "x": 0.24, "len": 0.76,
            "y": -0.08, "yanchor": "top",
            "currentvalue": {"prefix": "Instante t = ", "visible": True, "xanchor": "right"},
            "pad": {"b": 10, "t": 10},
            "steps": slider_steps
        }]
    )

    fig.update_xaxes(range=[t_min, t_max], dtick=1.0, showgrid=True, title_text="τ", row=1, col=1)
    fig.update_xaxes(range=[t_min, t_max], dtick=1.0, showgrid=True, title_text="t", row=2, col=1)
    fig.update_yaxes(range=y1_lim, zeroline=True, zerolinecolor="rgba(0,0,0,0.4)", title_text="Amplitud", row=1, col=1)
    fig.update_yaxes(range=y2_lim, zeroline=True, zerolinecolor="rgba(0,0,0,0.4)", title_text="(f * g)(t)", row=2, col=1)

    return fig


tab1, tab2, tab3 = st.tabs(["📊 Operaciones Básicas", "🔄 Convolución e Intervalos", "💻 Código Fuente del Autor"])

with tab1:
    st.subheader("Suma, Resta y Multiplicación de Funciones")
    operacion = st.radio(
        "Selecciona la operación:",
        ["Suma: f(t) + g(t)", "Resta: f(t) - g(t)", "Producto: f(t) * g(t)"],
        horizontal=True
    )

    if operacion == "Suma: f(t) + g(t)":
        res = f_t + g_t
        lbl = "f(t) + g(t)"
    elif operacion == "Resta: f(t) - g(t)":
        res = f_t - g_t
        lbl = "f(t) - g(t)"
    else:
        res = f_t * g_t
        lbl = "f(t) · g(t)"

    fig, ax = plt.subplots(3, 1, figsize=(10, 9))

    ax[0].plot(t, f_t, label=cfg_f["label"], color="blue", lw=2)
    ax[0].axhline(0, color="black", lw=0.8, alpha=0.5)
    configurar_eje_t(ax[0], etiqueta_x="t")
    dibujar_marcas_t(ax[0], obtener_puntos_criticos(cfg_f), color="blue")
    ax[0].legend(loc="upper right")

    ax[1].plot(t, g_t, label=cfg_g["label"], color="green", lw=2)
    ax[1].axhline(0, color="black", lw=0.8, alpha=0.5)
    configurar_eje_t(ax[1], etiqueta_x="t")
    dibujar_marcas_t(ax[1], obtener_puntos_criticos(cfg_g), color="green")
    ax[1].legend(loc="upper right")

    ax[2].plot(t, res, label=lbl, color="red", lw=2)
    ax[2].axhline(0, color="black", lw=0.8, alpha=0.5)
    configurar_eje_t(ax[2], etiqueta_x="t")
    ax[2].legend(loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Convolución Continua y Desplazamiento Interactivo")
    st.latex(r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \, g(t - \tau) \, d\tau")

    # Sección analítica
    st.markdown("### 📝 Expresión Analítica")
    if cfg_f["modo"] == "Función Única" and cfg_g["modo"] == "Función Única":
        latex_eq, nota = obtener_intervalos_analiticos(
            cfg_f["tipo"], cfg_g["tipo"], cfg_f["amp"], cfg_g["amp"], cfg_f["t0"], cfg_g["t0"]
        )
        st.latex(latex_eq)
        st.caption(f"📌 **Nota de análisis:** {nota}")
    else:
        if cfg_f["modo"] != "Función Única":
            st.latex(latex_funcion_tramos("f", cfg_f["tramos"]))
        if cfg_g["modo"] != "Función Única":
            st.latex(latex_funcion_tramos("g", cfg_g["tramos"]))
        st.caption("📌 **Nota de análisis:** La convolución se calcula numéricamente integrando el producto de la señal fija $f(\\tau)$ con la señal móvil $g(t - \\tau)$.")

    st.markdown("---")
    st.markdown("### 🎬 Señal Fija $f(\\tau)$ vs. Señal Móvil $g(t - \\tau)$")

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
            "Desplaza el instante t para mover la señal g(t - τ):",
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
            fps = st.slider("Velocidad (cuadros por segundo - FPS):", 5, 40, 20, step=5)

        n_frames = max(2, int(round((t_max - t_min) / paso_anim)) + 1)
        valores_t = np.linspace(t_min, t_max, n_frames)

        fig_anim = generar_animacion_plotly(valores_t, conv_res, fps=fps)
        st.plotly_chart(fig_anim, use_container_width=True)

with tab3:
    st.subheader("Verificación de Autoría")
    st.info("Código desarrollado por Jhonatan Ramírez para la materia de Matemática Avanzada (EPN).")
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
