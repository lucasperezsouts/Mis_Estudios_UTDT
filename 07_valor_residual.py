import pandas as pd
import numpy as np
import joblib

# ============================================
# CONSTANTES — todas documentadas con fuente y fecha
# ============================================
DOLAR_OFICIAL_VENTA = 1545       # ARS/USD, Banco Nación, 29-sep-2026
VALOR_UVA_ARS = 2137.27          # BCRA, 29-sep-2026

COSTO_CONSTRUCCION_USD_M2 = {
    # Rango validado cruzando dos fuentes:
    # - ICCBA/CPAU (feb-2026): promedio de 4 modelos de multivivienda
    #   ≈ $1.436.496 ARS/m² → ≈ USD 930/m² al dólar oficial de hoy
    # - Estimación directa en USD: 700-950 USD/m² calidad estándar,
    #   >1200 premium (estudiolatarq, 2026)
    "conservador": 700,
    "base": 850,
    "premium": 1200
}

MARGEN_DESARROLLADOR = 0.18   # 18%, punto medio del rango estándar de industria (15-20%)
ALTURA_POR_PISO = 3.0          # metros, supuesto documentado

# ============================================
# 1. Cargar el modelo entrenado (Paso: 04_entrenamiento.py)
# ============================================
paquete = joblib.load("modelo_xgboost.pkl")
modelo = paquete["modelo"]
preprocesador = paquete["preprocesador"]

# ============================================
# 2. Función: predecir precio con el modelo hedónico
# ============================================
def predecir_precio_m2(surface_total, surface_covered, rooms, bathrooms,
                        dist_centro_km, inc_uva_21, uni_edif_1, fot_em_1,
                        barrio, property_type="Departamento"):
    ratio_cubierta = surface_covered / surface_total
    prop = pd.DataFrame([{
        "surface_total": surface_total,
        "surface_covered": surface_covered,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "ratio_cubierta": ratio_cubierta,
        "dist_centro_km": dist_centro_km,
        "inc_uva_21": inc_uva_21,
        "uni_edif_1": uni_edif_1,
        "fot_em_1": fot_em_1,
        "barrio": barrio,
        "property_type": property_type
    }])
    X_proc = preprocesador.transform(prop)
    log_pred = modelo.predict(X_proc)[0]
    precio_total = np.exp(log_pred)
    return precio_total / surface_total   # precio por m²

# ============================================
# 3. Función: valor residual de suelo
# ============================================
def valor_residual_suelo(m2_construibles, precio_venta_por_m2, escenario="base"):
    costo_m2 = COSTO_CONSTRUCCION_USD_M2[escenario]
    ingreso_esperado = m2_construibles * precio_venta_por_m2
    costo_total = m2_construibles * costo_m2
    margen = ingreso_esperado * MARGEN_DESARROLLADOR
    return ingreso_esperado - costo_total - margen

# ============================================
# 4. Función: convertir inc_uva_21 (UVA/m²) a USD/m²
# ============================================
def uva_a_usd_m2(inc_uva_21):
    valor_ars_m2 = inc_uva_21 * VALOR_UVA_ARS
    return valor_ars_m2 / DOLAR_OFICIAL_VENTA

# ============================================
# 5. Armar el ranking por barrio, usando promedios reales del dataset
# ============================================
df = pd.read_csv("properati_features_clean_v2.csv", low_memory=False)

# Propiedad "tipo" para comparar barrios en igualdad de condiciones
SURFACE_TOTAL_TIPO = 60
SURFACE_COVERED_TIPO = 55
ROOMS_TIPO = 2
BATHROOMS_TIPO = 1

resumen_barrios = df.groupby("barrio").agg(
    uni_edif_1_prom=("uni_edif_1", "mean"),
    inc_uva_21_prom=("inc_uva_21", "mean"),
    dist_centro_km_prom=("dist_centro_km", "mean"),
    fot_em_1_prom=("fot_em_1", "mean"),
    n_casos=("barrio", "count")
).reset_index()

# Solo barrios con suficiente data para que el promedio sea confiable
resumen_barrios = resumen_barrios[resumen_barrios["n_casos"] >= 30].copy()

resultados = []
for _, fila in resumen_barrios.iterrows():
    if pd.isna(fila["uni_edif_1_prom"]) or pd.isna(fila["inc_uva_21_prom"]):
        continue

    precio_m2_predicho = predecir_precio_m2(
        SURFACE_TOTAL_TIPO, SURFACE_COVERED_TIPO, ROOMS_TIPO, BATHROOMS_TIPO,
        fila["dist_centro_km_prom"], fila["inc_uva_21_prom"],
        fila["uni_edif_1_prom"], fila["fot_em_1_prom"] if not pd.isna(fila["fot_em_1_prom"]) else 0,
        fila["barrio"]
    )

    m2_construibles = (fila["uni_edif_1_prom"] / ALTURA_POR_PISO) * SURFACE_TOTAL_TIPO \
        if fila["uni_edif_1_prom"] > 0 else SURFACE_TOTAL_TIPO

    valor_residual_base = valor_residual_suelo(m2_construibles, precio_m2_predicho, "base")
    valor_oficial_uva_usd_m2 = uva_a_usd_m2(fila["inc_uva_21_prom"])

    resultados.append({
        "barrio": fila["barrio"],
        "precio_m2_predicho_usd": round(precio_m2_predicho, 0),
        "valor_oficial_uva_usd_m2": round(valor_oficial_uva_usd_m2, 0),
        "brecha_pct": round((precio_m2_predicho - valor_oficial_uva_usd_m2) / valor_oficial_uva_usd_m2 * 100, 1),
        "valor_residual_suelo_usd": round(valor_residual_base, 0),
        "n_casos": int(fila["n_casos"])
    })

tabla_final = pd.DataFrame(resultados).sort_values("brecha_pct", ascending=False)

print("=== Ranking de barrios: brecha entre precio de mercado (modelo) y tasación oficial (UVA) ===")
print(tabla_final.to_string(index=False))

tabla_final.to_csv("ranking_zonas_oportunidad.csv", index=False)
print("\nGuardado: ranking_zonas_oportunidad.csv")
