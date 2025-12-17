🛒 README — Tienda Aurelion
Predicción de Ventas y Dashboard Analítico
📌 Descripción del Proyecto
Tienda Aurelion es un proyecto de análisis y predicción de ventas desarrollado para una cadena de tiendas minoristas. El objetivo principal es:

Predecir ventas por ciudad y por producto

Analizar tendencias temporales

Comparar ventas reales vs predichas

Visualizar métricas clave (KPIs)

Explorar datos mediante filtros interactivos

El proyecto combina:

Machine Learning (XGBoost + Pipeline profesional)

Análisis de datos

Dashboard interactivo en Streamlit

🧠 Modelo de Machine Learning
Se entrenó un modelo de regresión basado en XGBoost, encapsulado dentro de un Pipeline de Scikit-Learn, que incluye:

OneHotEncoder para variables categóricas

Transformación de fechas

Selección automática de features

Entrenamiento y evaluación

Guardado del pipeline completo (modelo_ventas_ciudad.pkl)

El modelo predice el importe estimado de cada venta, lo que permite:

Agregar ventas por ciudad

Agregar ventas por producto

Comparar ventas reales vs predichas

📊 Dashboard en Streamlit
El dashboard incluye:

✔️ KPI Cards
Ventas reales

Ventas predichas

Ticket promedio

✔️ Gráficos
Tendencia temporal (ventas por mes)

Comparación por categoría

Ventas por ciudad

Ranking de productos más vendidos

Mapa geográfico de ventas

✔️ Filtros interactivos
Ciudad

Categoría

Mes

✔️ Exportación de reportes (código de ejemplo)
📁 Estructura del Proyecto
Código
tienda-aurelion/
│
├── data/
│   └── df_ml.csv
│
├── modelo/
│   └── modelo_ventas_ciudad.pkl
│
├── entrenar_modelo.py
├── predecir_ventas.py
├── app_streamlit.py
├── requirements.txt
└── README.md


⚙️ Instalación
1. Clonar el repositorio
Código
git clone https://github.com/usuario/tienda-aurelion.git
cd tienda-aurelion

2. Crear entorno conda
Código
conda create -n aurelion python=3.10
conda activate aurelion

3. Instalar dependencias
Código
pip install -r requirements.txt
🚀 Ejecutar el Dashboard

Código
streamlit run app_streamlit.py
El dashboard se abrirá en tu navegador.

🧪 Entrenar el Modelo Nuevamente
Si querés reentrenar el modelo:

Código
python entrenar_modelo.py
Esto generará un nuevo archivo:

Código
modelo/modelo_ventas_ciudad.pkl
📈 Predicción de Ventas por Ciudad

Código
python predecir_ventas.py
Esto mostrará ventas estimadas por ciudad.