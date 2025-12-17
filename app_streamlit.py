import streamlit as st
import pandas as pd
import joblib
import pickle

st.set_page_config(page_title="Predicción de Ventas", layout="wide")

st.title("📊 Dashboard de Ventas Reales y Predichas")

st.markdown("""
<link rel="stylesheet"
href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
""", unsafe_allow_html=True)

# ============================
# 1. Cargar modelo
# ============================
#pipeline = joblib.load("modelo/modelo_ventas_ciudad.pkl")


with open("modelo/modelo_ventas_ciudad.pkl", "rb") as f:
    pipeline = pickle.load(f)
# ============================
# 2. Cargar dataset
# ============================
df = pd.read_csv("data/df_ml.csv")

# Preprocesamiento igual al entrenamiento
df["fecha"] = pd.to_datetime(df["fecha"])
df["año"] = df["fecha"].dt.year
df["mes"] = df["fecha"].dt.month
df["dia"] = df["fecha"].dt.day
df["dia_semana"] = df["fecha"].dt.dayofweek

df = df.drop(columns=["fecha", "fecha_alta"])

# ============================
# 3. Predicciones
# ============================
df_pred = df.copy()
df_pred["prediccion"] = pipeline.predict(df_pred)

# ============================
# 4. Filtros interactivos
# ============================
st.sidebar.header("🔎 Filtros")

ciudades = ["Todas"] + sorted(df_pred["ciudad"].unique())
categorias = ["Todas"] + sorted(df_pred["categoria"].unique())
meses = ["Todos"] + sorted(df_pred["mes"].unique())
medio_pago = ["Todos"] + sorted(df_pred["medio_pago"].unique())

f_ciudad = st.sidebar.selectbox("Ciudad", ciudades)
f_categoria = st.sidebar.selectbox("Categoría", categorias)
f_mes = st.sidebar.selectbox("Mes", meses)
f_medio_pago = st.sidebar.selectbox("Medio de Pago", medio_pago)

# Aplicar filtros
df_filtrado = df_pred.copy()

if f_ciudad != "Todas":
    df_filtrado = df_filtrado[df_filtrado["ciudad"] == f_ciudad]

if f_categoria != "Todas":
    df_filtrado = df_filtrado[df_filtrado["categoria"] == f_categoria]

if f_mes != "Todos":
    df_filtrado = df_filtrado[df_filtrado["mes"] == f_mes]

if f_medio_pago != "Todos":
    df_filtrado = df_filtrado[df_filtrado["medio_pago"] == f_medio_pago]  
    

# ============================
# 5. KPI CARDS
# ============================
st.subheader("🏙️ Ventas Reales por Ciudad")

ventas_por_ciudad = df_filtrado.groupby("ciudad")["importe"].sum()

p80 = ventas_por_ciudad.quantile(0.80)
p40 = ventas_por_ciudad.quantile(0.40)

cards_html = ""

for ciudad, valor in ventas_por_ciudad.items():

    if valor >= p80:
        color = "success"
    elif valor >= p40:
        color = "warning"
    else:
        color = "danger"

    badge_class = f"badge bg-{color} mb-2"

    cards_html += f"""<div class="col-sm-6 col-md-4">
    <div class="card border-{color} mb-3 shadow" style="border-radius: 12px;">
    <div class="card-body">
    <h5 class="card-title">📍 {ciudad}</h5>
    <span class="{badge_class}">Nivel: {color.capitalize()}</span>
    <p class="card-text" style="font-size: 1.4rem; font-weight: bold;">
    💰 ${valor:,.2f}
    </p>
    </div>
    </div>
    </div>
    """

st.markdown(f"""
<div class="container">
<div class="row">
{cards_html}
</div>
</div>
""", unsafe_allow_html=True)
  

st.subheader("📌 Indicadores Clave (KPI)")

col1, col2, col3 = st.columns(3)

ventas_totales = df_filtrado["importe"].sum()
ventas_predichas = df_filtrado["prediccion"].sum()
ticket_promedio = df_filtrado["importe"].mean()

col1.metric("💰 Ventas Reales", f"${ventas_totales:,.0f}")
col2.metric("🔮 Ventas Predichas", f"${ventas_predichas:,.0f}")
col3.metric("🧾 Ticket Promedio", f"${ticket_promedio:,.0f}")

# ============================
# 6. Tendencia temporal (ventas por mes)
# ============================
st.subheader("📈 Tendencia Temporal (Ventas por Mes)")

tendencia = df_filtrado.groupby("mes")[["importe", "prediccion"]].sum()

st.line_chart(tendencia)

# ============================
# 7. Comparación por categoría
# ============================
st.subheader("📦 Comparación de Ventas por Categoría")

ventas_cat = df_filtrado.groupby("categoria")[["importe", "prediccion"]].sum()

st.bar_chart(ventas_cat)

# ============================
# 8. Mapa geográfico de ventas
# ============================
st.subheader("🗺️ Mapa Geográfico de Ventas")

# Coordenadas aproximadas de ciudades
coords = {
    "Cordoba": (-31.4201, -64.1888),
    "Villa Maria": (-32.4075, -63.2402),
    "Alta Gracia": (-31.6583, -64.4283),
    "Rio Cuarto": (-33.1232, -64.3493),
    "Carlos Paz": (-31.4241, -64.4978),
    "Mendiolaza": (-31.2828, -64.2964)
}

df_map = df_filtrado.groupby("ciudad")["prediccion"].sum().reset_index()
df_map["lat"] = df_map["ciudad"].map(lambda x: coords[x][0])
df_map["lon"] = df_map["ciudad"].map(lambda x: coords[x][1])

st.map(df_map)

# ============================
# 9. Ranking de productos más vendidos
# ============================
st.subheader("🏆 Ranking de Productos Más Vendidos (Predicción)")

ranking_productos = (
    df_filtrado.groupby("id_producto")["prediccion"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(ranking_productos)


st.subheader("📄 Datos Filtrados con Predicciones") 
st.dataframe(df_filtrado)

# ============================
# 10. Exportar reportes (solo mostrar cómo hacerlo)
# ============================
#st.subheader("📤 Exportar Reportes")


#Para exportar reportes en CSV o Excel, podés usar:
csv_data = df_filtrado.to_csv(index=False)

st.download_button(
    label="Descargar CSV",
    data=csv_data,
    file_name="reporte.csv",
    mime="text/csv"
)
import io

# Crear buffer en memoria
buffer = io.BytesIO()

# Escribir Excel en el buffer
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    df_filtrado.to_excel(writer, sheet_name="Reporte", index=False)

# Volver al inicio del buffer
buffer.seek(0)

# Botón de descarga
st.download_button(
    label="Descargar Excel",
    data=buffer,
    file_name="reporte.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
