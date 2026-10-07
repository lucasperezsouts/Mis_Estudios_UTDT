import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("properati_features_clean.csv", low_memory=False)
print("Shape:", df.shape)

# ============================================
# 1. Distribución general del precio y precio/m²
# ============================================
print("\n=== Distribución de price ===")
print(df["price"].describe())

print("\n=== Distribución de price_per_m2 ===")
print(df["price_per_m2"].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df["price"].hist(bins=50, ax=axes[0])
axes[0].set_title("Distribución de price (USD)")
df["price_per_m2"].hist(bins=50, ax=axes[1])
axes[1].set_title("Distribución de price_per_m2 (USD/m²)")
plt.tight_layout()
plt.savefig("eda_distribucion_precios.png")
print("\nGuardado: eda_distribucion_precios.png")

# ============================================
# 2. Precio por m² por barrio — ranking
# ============================================
precio_por_barrio = (
    df.groupby("barrio")["price_per_m2"]
    .agg(["mean", "median", "count"])
    .sort_values("median", ascending=False)
)
print("\n=== Precio por m² por barrio (ordenado por mediana) ===")
print(precio_por_barrio.head(15))
print("...")
print(precio_por_barrio.tail(10))

# Gráfico de los 15 barrios más caros y más baratos
top15 = precio_por_barrio.head(15)
bottom15 = precio_por_barrio.tail(15)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top15["median"].sort_values().plot(kind="barh", ax=axes[0], color="darkred")
axes[0].set_title("15 barrios más caros (mediana USD/m²)")
bottom15["median"].sort_values().plot(kind="barh", ax=axes[1], color="darkgreen")
axes[1].set_title("15 barrios más baratos (mediana USD/m²)")
plt.tight_layout()
plt.savefig("eda_precio_por_barrio.png")
print("\nGuardado: eda_precio_por_barrio.png")

# ============================================
# 3. Matriz de correlación con price y price_per_m2
# ============================================
cols_numericas = [
    "price", "price_per_m2", "surface_total", "surface_covered",
    "rooms", "bedrooms", "bathrooms", "ratio_cubierta",
    "dist_centro_km", "uni_edif_1", "fot_em_1", "inc_uva_21",
    "m2_construibles"
]
cols_disponibles = [c for c in cols_numericas if c in df.columns]

corr = df[cols_disponibles].corr()
print("\n=== Correlación con price ===")
print(corr["price"].sort_values(ascending=False))

print("\n=== Correlación con price_per_m2 ===")
print(corr["price_per_m2"].sort_values(ascending=False))

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols_disponibles)))
ax.set_yticks(range(len(cols_disponibles)))
ax.set_xticklabels(cols_disponibles, rotation=90)
ax.set_yticklabels(cols_disponibles)
plt.colorbar(im)
plt.tight_layout()
plt.savefig("eda_correlacion.png")
print("\nGuardado: eda_correlacion.png")

# ============================================
# 4. Precio vs. distancia al centro — ¿se cumple lo esperado?
# ============================================
print("\n=== Correlación dist_centro_km vs price_per_m2 ===")
print(df[["dist_centro_km", "price_per_m2"]].corr())

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["dist_centro_km"], df["price_per_m2"], alpha=0.1, s=5)
ax.set_xlabel("Distancia al centro (km)")
ax.set_ylabel("Precio por m² (USD)")
ax.set_title("Precio/m² vs. distancia al centro")
plt.tight_layout()
plt.savefig("eda_precio_vs_distancia.png")
print("\nGuardado: eda_precio_vs_distancia.png")

# ============================================
# 5. Precio según distrito especial
# ============================================
print("\n=== price_per_m2 según distrito_especial ===")
print(df.groupby("distrito_especial")["price_per_m2"].describe())

# ============================================
# 6. Property type — cuánto pesa cada tipo
# ============================================
print("\n=== Distribución por property_type ===")
print(df["property_type"].value_counts())
print("\n=== price_per_m2 promedio por property_type ===")
print(df.groupby("property_type")["price_per_m2"].median().sort_values(ascending=False))