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
> *Aplicación interactiva para el cálculo de operaciones básicas, convolución continua y análisis analítico por intervalos.*
""")

# Parámetros de funciones
st.sidebar.header("⚙️ Configuración de Señales")
t_min, t_max = st.sidebar.slider("Rango de tiempo (t)", -10.0, 10.0, (-5.0, 5.0), step=0.5)
t = np.linspace(t_min, t_max, 1000)
dt = t[1] - t[0]

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
        ancho = 0.05
        return np.where((t >= -ancho/2) & (t <= ancho/2), 1.0 / ancho, 0.0)
    elif tipo == "Rampa r(t)":
        return np.where(t >= 0, t, 0.0)
    elif tipo == "Función Signo sgn(t)":
        return np.sign(t)
    return np.zeros_like(t)

def obtener_intervalos_analiticos(f_name, g_name):
    """Devuelve la solución analítica formal expresada a trozos por intervalos para casos estándar."""
    if f_name == "Pulso Rectangular / Compuerta" and g_name == "Pulso Rectangular / Compuerta":
        return r"""
        (f * g)(t) = 
        \begin{cases} 
        0, & t < -2 \\ 
        t + 2, & -2 \le t < 0 \\ 
        2 - t, & 0 \le t \le 2 \\ 
        0, & t > 2 
        \end{cases}
        """, "Resultado: Triángulo centrado en t=0 con soporte en [-2, 2]."

    elif f_name == "Exponencial Decreciente" and g_name == "Escalón Unitario u(t)":
        return r"""
        (f * g)(t) = 
        \begin{cases} 
        0, & t < 0 \\ 
        1 - e^{-t}, & t \ge 0 
        \end{cases}
        """, "Resultado: Crecimiento exponencial acotado para t ≥ 0."

    elif f_name == "Pulso Rectangular / Compuerta" and g_name == "Exponencial Decreciente":
        return r"""
        (f * g)(t) = 
        \begin{cases} 
        0, & t < -1 \\ 
        1 - e^{-(t+1)}, & -1 \le t < 1 \\ 
        e^{-(t-1)} - e^{-(t+1)}, & t \ge 1 
        \end{cases}
        """, "Resultado: Respuesta de un filtro de primer orden ante un pulso."

    elif "Impulso Unitario" in f_name or "Impulso Unitario" in g_name:
        otra = g_name if "Impulso Unitario" in f_name else f_name
        return rf"""
        (f * g)(t) = {otra}(t) \quad \forall t \in \mathbb{{R}}
        """, "Propiedad de tamizado: La convolución con un impulso δ(t) devuelve la misma función."

    else:
        return r"""
        (f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
        """, "Representación evaluada numéricamente de forma continua sobre el rango t definido."

f_t = generar_funcion(tipo_f, t)
g_t = generar_funcion(tipo_g, t)

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
    ax[0].plot(t, f_t, label=f"f(t) = {tipo_f}", color="blue")
    ax[0].grid(True); ax[0].legend(loc="upper right")
    
    ax[1].plot(t, g_t, label=f"g(t) = {tipo_g}", color="green")
    ax[1].grid(True); ax[1].legend(loc="upper right")
    
    ax[2].plot(t, res, label=lbl, color="red")
    ax[2].grid(True); ax[2].legend(loc="upper right")
    
    st.pyplot(fig)

with tab2:
    st.subheader("Convolución Continua y Definición por Intervalos")
    st.latex(r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau")
    
    # Mostrar desarrollo por intervalos analíticos
    latex_eq, nota = obtener_intervalos_analiticos(tipo_f, tipo_g)
    
    st.markdown("### 📝 Expresión Analítica Resultante por Intervalos")
    st.latex(latex_eq)
    st.caption(f"📌 **Nota de análisis:** {nota}")
    
    st.markdown("---")
    st.markdown("### 📈 Gráfica del Resultado")
    
    conv_res = np.convolve(f_t, g_t, mode='same') * dt
    
    fig_conv, ax_conv = plt.subplots(figsize=(10, 4))
    ax_conv.plot(t, conv_res, label="(f * g)(t)", color="purple", lw=2)
    ax_conv.set_title(f"Convolución entre {tipo_f} y {tipo_g}")
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
