import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ============================
# 1. Cargar dataset
# ============================
df = pd.read_csv("data/df_ml.csv")

# ============================
# 2. Preprocesamiento inicial
# ============================

# Convertir fecha
df["fecha"] = pd.to_datetime(df["fecha"])
df["año"] = df["fecha"].dt.year
df["mes"] = df["fecha"].dt.month
df["dia"] = df["fecha"].dt.day
df["dia_semana"] = df["fecha"].dt.dayofweek

# Eliminar columnas irrelevantes
df = df.drop(columns=["fecha", "fecha_alta"])

# Target
y = df["importe"]

# Features
X = df.drop(columns=["importe"])

# Columnas categóricas
cat_cols = ["ciudad", "categoria", "medio_pago"]

# Columnas numéricas
num_cols = [c for c in X.columns if c not in cat_cols]

# ============================
# 3. Pipeline profesional
# ============================

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# ============================
# 4. Train-test split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# 5. Entrenar modelo
# ============================

pipeline.fit(X_train, y_train)

# ============================
# 6. Evaluación
# ============================

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# ============================
# 7. Guardar modelo
# ============================

joblib.dump(pipeline, "modelo/modelo_ventas_ciudad.pkl")

print("✅ Modelo guardado correctamente")
