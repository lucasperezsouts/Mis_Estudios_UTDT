import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ============================================
# 1. Cargar el dataset
# ============================================
df_modelo = pd.read_csv("properati_caba_con_normativa.csv", low_memory=False)
print("Shape inicial:", df_modelo.shape)

# ============================================
# 2. Fallback: donde falta surface_total pero hay surface_covered,
#    usamos ese valor (asumiendo superficie descubierta despreciable
#    o simplemente no reportada) — recupera ~4.000 filas
# ============================================
print("surface_total nulo ANTES del fallback:", df_modelo["surface_total"].isna().sum())
df_modelo["surface_total"] = df_modelo["surface_total"].fillna(df_modelo["surface_covered"])
print("surface_total nulo DESPUÉS del fallback:", df_modelo["surface_total"].isna().sum())

# ============================================
# 3. Limpieza de negativos / códigos de "sin dato"
# ============================================
print("\n=== Negativos detectados ===")
print("fot_em_1:", (df_modelo["fot_em_1"] < 0).sum())
print("uni_edif_1:", (df_modelo["uni_edif_1"] < 0).sum())
print("inc_uva_21:", (df_modelo["inc_uva_21"] < 0).sum())

for col in ["fot_em_1", "uni_edif_1", "inc_uva_21"]:
    df_modelo.loc[df_modelo[col] < 0, col] = None

# Descartar filas con precio o superficie inválidos (esto ahora
# descarta MENOS filas gracias al fallback de arriba)
antes = len(df_modelo)
df_modelo = df_modelo[(df_modelo["price"] > 0) & (df_modelo["surface_total"] > 0)].copy()
print(f"\nFilas descartadas por price/surface inválidos: {antes - len(df_modelo)}")
print("Shape tras limpieza:", df_modelo.shape)

# ============================================
# 4. Feature engineering
# ============================================
df_modelo["price_per_m2"] = df_modelo["price"] / df_modelo["surface_total"]
df_modelo["ratio_cubierta"] = df_modelo["surface_covered"] / df_modelo["surface_total"]

def distancia_haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

OBELISCO_LAT, OBELISCO_LON = -34.6037, -58.3816
df_modelo["dist_centro_km"] = df_modelo.apply(
    lambda row: distancia_haversine(row["lon"], row["lat"], OBELISCO_LAT, OBELISCO_LON),
    axis=1
)

df_modelo["distrito_especial"] = df_modelo["dist_1_grp"].notna()
df_modelo["m2_construibles"] = df_modelo["uni_edif_1"] * 3

# ============================================
# 5. Recorte de outliers extremos en price_per_m2
# ============================================
p1, p99 = df_modelo["price_per_m2"].quantile([0.01, 0.99])
print(f"\nRecortando price_per_m2 fuera de [{p1:.0f}, {p99:.0f}]")

antes = len(df_modelo)
df_modelo = df_modelo[df_modelo["price_per_m2"].between(p1, p99)].copy()
print(f"Filas descartadas por outliers extremos: {antes - len(df_modelo)}")

# ============================================
# 6. Chequeo final
# ============================================
print("\n=== Shape final ===")
print(df_modelo.shape)

print("\n=== Estadísticas descriptivas ===")
print(df_modelo[["price_per_m2", "dist_centro_km", "m2_construibles"]].describe())

# ============================================
# 7. Guardar
# ============================================
df_modelo.to_csv("properati_features_clean.csv", index=False)
print(f"\nGuardado: properati_features_clean.csv ({len(df_modelo)} filas)")