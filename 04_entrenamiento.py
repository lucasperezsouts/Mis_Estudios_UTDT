import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import category_encoders as ce
import joblib

# ============================================
# 1. Cargar el dataset limpio
# ============================================
df = pd.read_csv("properati_features_clean_v2.csv", low_memory=False)
print("Shape:", df.shape)

# ============================================
# 2. Features — ya NO hace falta agrupar barrios chicos a mano,
#    el target encoding con smoothing lo resuelve automáticamente
# ============================================
features_numericas = [
    "surface_total", "surface_covered", "rooms", "bathrooms",
    "ratio_cubierta", "dist_centro_km", "inc_uva_21", "uni_edif_1", "fot_em_1"
]
features_categoricas = ["barrio", "property_type"]
target = "log_price"

cols_necesarias = features_numericas + features_categoricas + [target]
antes = len(df)
df_modelo = df.dropna(subset=cols_necesarias).copy()
print(f"\nFilas descartadas por NaN en features/target: {antes - len(df_modelo)}")
print("Shape para entrenar:", df_modelo.shape)
print("Cantidad de barrios únicos:", df_modelo["barrio"].nunique())

X = df_modelo[features_numericas + features_categoricas]
y = df_modelo[target]

# ============================================
# 3. Train/test split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ============================================
# 4. Preprocesador — target encoding (con smoothing) para "barrio",
#    one-hot normal para "property_type" (pocas categorías, sin
#    problema de casos raros)
# ============================================
preprocesador = ColumnTransformer(
    transformers=[
        ("num", "passthrough", features_numericas),
        ("barrio_encoded", ce.TargetEncoder(cols=["barrio"], smoothing=10), ["barrio"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["property_type"])
    ]
)

# ============================================
# 5. Regresión lineal
# ============================================
pipeline_lr = Pipeline([
    ("preprocesador", preprocesador),
    ("modelo", LinearRegression())
])
# OJO: el TargetEncoder necesita ver "y" para fitear (usa el precio
# para calcular la media por barrio) — por eso el .fit() del Pipeline
# recibe X_train Y y_train juntos, y sklearn se encarga de pasarle "y"
# a cada paso que lo necesite. Al llamar .predict() en test, NO usa
# el y de test, solo aplica el mapeo ya aprendido — así se evita el leakage.
pipeline_lr.fit(X_train, y_train)
pred_lr_train = pipeline_lr.predict(X_train)
pred_lr_test = pipeline_lr.predict(X_test)

# ============================================
# 6. XGBoost con regularización + early stopping
# ============================================
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

preprocesador_fit = preprocesador.fit(X_tr, y_tr)  # el fit necesita "y" acá también
X_tr_proc = preprocesador_fit.transform(X_tr)
X_val_proc = preprocesador_fit.transform(X_val)
X_test_proc = preprocesador_fit.transform(X_test)
X_train_proc = preprocesador_fit.transform(X_train)

modelo_xgb = XGBRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric="mae"
)

modelo_xgb.fit(
    X_tr_proc, y_tr,
    eval_set=[(X_val_proc, y_val)],
    verbose=False
)
print(f"\nXGBoost paró en el árbol #{modelo_xgb.best_iteration} (de 1000 posibles)")

pred_xgb_train = modelo_xgb.predict(X_train_proc)
pred_xgb_test = modelo_xgb.predict(X_test_proc)

# ============================================
# 7. Evaluar TRAIN vs TEST
# ============================================
def evaluar(y_true_log, y_pred_log):
    r2 = r2_score(y_true_log, y_pred_log)
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    y_true_usd, y_pred_usd = np.exp(y_true_log), np.exp(y_pred_log)
    mape_usd = np.mean(np.abs((y_true_usd - y_pred_usd) / y_true_usd)) * 100
    return r2, mae_log, mape_usd

print("\n" + "="*60)
print("REGRESIÓN LINEAL")
print("="*60)
r2_tr, mae_tr, mape_tr = evaluar(y_train, pred_lr_train)
r2_te, mae_te, mape_te = evaluar(y_test, pred_lr_test)
print(f"Train  -> R²: {r2_tr:.4f} | MAE(log): {mae_tr:.4f} | MAPE: {mape_tr:.1f}%")
print(f"Test   -> R²: {r2_te:.4f} | MAE(log): {mae_te:.4f} | MAPE: {mape_te:.1f}%")
print(f"Gap R² (train-test): {r2_tr - r2_te:.4f}")

print("\n" + "="*60)
print("XGBOOST")
print("="*60)
r2_tr, mae_tr, mape_tr = evaluar(y_train, pred_xgb_train)
r2_te, mae_te, mape_te = evaluar(y_test, pred_xgb_test)
print(f"Train  -> R²: {r2_tr:.4f} | MAE(log): {mae_tr:.4f} | MAPE: {mape_tr:.1f}%")
print(f"Test   -> R²: {r2_te:.4f} | MAE(log): {mae_te:.4f} | MAPE: {mape_te:.1f}%")
print(f"Gap R² (train-test): {r2_tr - r2_te:.4f}")

# ============================================
# 8. Feature importance
# ============================================
nombres_cat = preprocesador_fit.named_transformers_["cat"].get_feature_names_out(["property_type"])
nombres_completos = features_numericas + ["barrio_encoded"] + list(nombres_cat)

df_importancia = pd.DataFrame({
    "feature": nombres_completos,
    "importancia": modelo_xgb.feature_importances_
}).sort_values("importancia", ascending=False)

print("\n=== Top 15 features (XGBoost) ===")
print(df_importancia.head(15).to_string(index=False))

# ============================================
# 9. Guardar
# ============================================
joblib.dump(pipeline_lr, "modelo_regresion_lineal.pkl")
joblib.dump({"modelo": modelo_xgb, "preprocesador": preprocesador_fit}, "modelo_xgboost.pkl")
print("\nModelos guardados")
print("Tipos de propiedad:")
print(df_modelo["property_type"].value_counts())

print("\nBarrios:")
print(sorted(df_modelo["barrio"].unique()))
# ===== PROBAR LA OFICINA DEL AVISO =====

# Referencia para completar los datos que faltan
referencia = X_train[
    (X_train["barrio"] == "Palermo") &
    (X_train["property_type"] == "Oficina")
]

if referencia.empty:
    raise ValueError("No hay oficinas de Palermo en entrenamiento.")

faltantes = ["dist_centro_km", "inc_uva_21", "uni_edif_1", "fot_em_1"]
valores_aproximados = referencia[faltantes].median()

oficina = pd.DataFrame([{
    "surface_total": 80,
    "surface_covered": 80,
    "rooms": 2,
    "bathrooms": 3,
    "ratio_cubierta": 1.0,  # Asumiendo cubierta / total
    "dist_centro_km": valores_aproximados["dist_centro_km"],
    "inc_uva_21": valores_aproximados["inc_uva_21"],
    "uni_edif_1": valores_aproximados["uni_edif_1"],
    "fot_em_1": valores_aproximados["fot_em_1"],
    "barrio": "Palermo",  # Asignación para esta prueba
    "property_type": "Oficina"
}])

oficina = oficina[features_numericas + features_categoricas]

oficina_procesada = preprocesador_fit.transform(oficina)
prediccion_log = modelo_xgb.predict(oficina_procesada)[0]
precio_estimado = float(np.exp(prediccion_log))

precio_aviso = 290_000
diferencia = precio_estimado - precio_aviso

print("\n=== PRUEBA APROXIMADA: OFICINA LUIS MARÍA CAMPOS 559 ===")
print(f"Precio del aviso: USD {precio_aviso:,.0f}")
print(f"Precio estimado:  USD {precio_estimado:,.0f}")
print(f"Diferencia:      USD {diferencia:+,.0f}")
print(f"Diferencia porcentual: {diferencia / precio_aviso * 100:+.1f}%")
print("\nValores aproximados utilizados:")
print(valores_aproximados)
# Copiar las propiedades del test
resultados_test = X_test.copy()

# Pasar los precios de logaritmos a dólares
resultados_test["precio_real"] = np.exp(y_test.to_numpy())
resultados_test["precio_predicho"] = np.exp(pred_xgb_test)

# Calcular el error de cada propiedad
resultados_test["error_usd"] = (
    resultados_test["precio_real"]
    - resultados_test["precio_predicho"]
).abs()

resultados_test["error_porcentual"] = (
    resultados_test["error_usd"]
    / resultados_test["precio_real"]
) * 100

# Agrupar los resultados por tipo de propiedad
resumen = resultados_test.groupby("property_type").agg(
    cantidad=("error_porcentual", "size"),
    MAPE_porcentaje=("error_porcentual", "mean"),
    MAE_USD=("error_usd", "mean")
).sort_values("cantidad", ascending=False)

print("\n=== RESULTADOS DEL TEST POR TIPO DE PROPIEDAD ===")
print(resumen.round(2).to_string())