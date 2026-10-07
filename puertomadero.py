import pandas as pd
import numpy as np

df = pd.read_csv("properati_features_clean.csv", low_memory=False)

# ============================================
# 1. Log de precio (para el modelo)
# ============================================
df["log_price"] = np.log(df["price"])
df["log_price_per_m2"] = np.log(df["price_per_m2"])

print("=== Distribución de log_price ===")
print(df["log_price"].describe())
print("\n=== Distribución de log_price_per_m2 ===")
print(df["log_price_per_m2"].describe())

# ============================================
# 2. Chequear la hipótesis: ¿Puerto Madero es caro por su mix de
#    property_type, o es caro incluso solo mirando departamentos?
# ============================================
print("\n=== Mix de property_type en Puerto Madero ===")
print(df[df["barrio"] == "Puerto Madero"]["property_type"].value_counts())

print("\n=== price_per_m2 en Puerto Madero, SOLO Departamentos ===")
pm_deptos = df[(df["barrio"] == "Puerto Madero") & (df["property_type"] == "Departamento")]
print(pm_deptos["price_per_m2"].describe())

print("\n=== Comparación: ranking de barrios usando SOLO Departamentos ===")
solo_deptos = df[df["property_type"] == "Departamento"]
ranking_deptos = (
    solo_deptos.groupby("barrio")["price_per_m2"]
    .agg(["median", "count"])
    .sort_values("median", ascending=False)
)
print(ranking_deptos.head(10))

# ============================================
# 3. Guardar con las columnas de log agregadas
# ============================================
df.to_csv("properati_features_clean.csv", index=False)
print(f"\nGuardado (actualizado con log_price): {len(df)} filas")