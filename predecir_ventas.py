import pandas as pd
import joblib

# Cargar modelo entrenado
pipeline = joblib.load("modelo/modelo_ventas_ciudad.pkl")

# Cargar dataset original o uno nuevo
df = pd.read_csv("data/df_ml.csv")

# Convertir fecha igual que en el entrenamiento
df["fecha"] = pd.to_datetime(df["fecha"])
df["año"] = df["fecha"].dt.year
df["mes"] = df["fecha"].dt.month
df["dia"] = df["fecha"].dt.day
df["dia_semana"] = df["fecha"].dt.dayofweek

df = df.drop(columns=["fecha", "fecha_alta"])

# Predecir ventas por fila
df_pred = df.copy()
df_pred["prediccion"] = pipeline.predict(df_pred)

# Agrupar por ciudad
ventas_por_ciudad = df_pred.groupby("ciudad")["prediccion"].sum()

print("✅ Ventas estimadas por ciudad:")
print(ventas_por_ciudad)
