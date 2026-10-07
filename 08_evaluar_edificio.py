
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
from math import radians, sin, cos, sqrt, atan2

# ============================================
# CONSTANTES — documentadas, con fuente
# ============================================
COSTO_CONSTRUCCION_USD_M2 = {"conservador": 700, "base": 850, "premium": 1200}
MARGEN_DESARROLLADOR = 0.18
ALTURA_POR_PISO = 3.0
COSTO_DEMOLICION_PCT = 0.09

# Factor de eficiencia constructiva — el cálculo de m² por altura sola es
# una "cota superior" (no descuenta retiros ni fondo libre de manzana),
# según metodología del proyecto github.com/meryboth/buenos-aires-data,
# que usa la altura (uni_edif_1) como único criterio y excluye el FOT
# explícitamente por corresponder al Código de Planeamiento derogado.
FACTOR_EFICIENCIA_CONSTRUCTIVA = 0.75

# ============================================
# INPUTS MANUALES — lo único que cambia caso por caso
# ============================================
LAT_EDIFICIO = -34.6187
LON_EDIFICIO = -58.4358
M2_YA_CONSTRUIDOS = 800
ESCENARIO_COSTO = "base"

# ============================================
# 1. Cargar catastro (geometría + normativa)
# ============================================
parcelas_geom = gpd.read_file("parcelas-shp/parcelas_catastrales.shp")
atributos = pd.read_csv("codigo-urbanistico.csv", low_memory=False)

def normalizar_smp(x):
    return str(x).strip().replace(" ", "").upper()

col_smp_csv = "smp1" if "smp1" in atributos.columns else "smp"
parcelas_geom["smp_norm"] = parcelas_geom["smp"].apply(normalizar_smp)
atributos["smp_norm"] = atributos[col_smp_csv].apply(normalizar_smp)

parcelas = parcelas_geom.merge(
    atributos.drop(columns=["barrio", "comuna"], errors="ignore"),
    on="smp_norm", how="left"
)

if "smp_x" in parcelas.columns:
    parcelas = parcelas.rename(columns={"smp_x": "smp"})

parcelas_metros = parcelas.to_crs(epsg=3857)
parcelas["superficie_lote_m2"] = parcelas_metros.geometry.area

# ============================================
# 2. Ubicar la parcela real del edificio, por coordenadas
# ============================================
punto = gpd.GeoDataFrame(
    [{"lat": LAT_EDIFICIO, "lon": LON_EDIFICIO}],
    geometry=gpd.points_from_xy([LON_EDIFICIO], [LAT_EDIFICIO]),
    crs="EPSG:4326"
)

match = gpd.sjoin(punto, parcelas, how="left", predicate="within")

if match["index_right"].isna().all():
    parcelas_buffer = parcelas_metros.copy()
    parcelas_buffer["geometry"] = parcelas_metros.geometry.buffer(15)
    parcelas_buffer = parcelas_buffer.to_crs(epsg=4326)
    match = gpd.sjoin(punto, parcelas_buffer, how="left", predicate="within")

if match["index_right"].isna().all():
    raise ValueError("No se encontró ninguna parcela en esas coordenadas. Revisá lat/lon.")

parcela = parcelas.loc[int(match["index_right"].iloc[0])]

print("=== Parcela identificada ===")
print(f"Barrio: {parcela['barrio']}")
print(f"SMP: {parcela['smp']}")
print(f"Superficie del lote: {parcela['superficie_lote_m2']:.1f} m²")
print(f"Altura máxima permitida (uni_edif_1): {parcela['uni_edif_1']}")
print(f"FOT (fot_em_1, no se usa — código derogado): {parcela['fot_em_1']}")
print(f"Distrito especial: {parcela['dist_1_grp'] if pd.notna(parcela['dist_1_grp']) else 'No'}")

# ============================================
# 3. Cargar el modelo hedónico
# ============================================
paquete = joblib.load("modelo_xgboost.pkl")
modelo = paquete["modelo"]
preprocesador = paquete["preprocesador"]

def predecir_precio_m2(surface_total, surface_covered, rooms, bathrooms,
                        dist_centro_km, inc_uva_21, uni_edif_1, fot_em_1,
                        barrio, property_type="Departamento"):
    ratio_cubierta = surface_covered / surface_total
    prop = pd.DataFrame([{
        "surface_total": surface_total, "surface_covered": surface_covered,
        "rooms": rooms, "bathrooms": bathrooms, "ratio_cubierta": ratio_cubierta,
        "dist_centro_km": dist_centro_km, "inc_uva_21": inc_uva_21,
        "uni_edif_1": uni_edif_1, "fot_em_1": fot_em_1,
        "barrio": barrio, "property_type": property_type
    }])
    X_proc = preprocesador.transform(prop)
    return np.exp(modelo.predict(X_proc)[0]) / surface_total

def dist_haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2*atan2(sqrt(a), sqrt(1-a))

dist_centro = dist_haversine(LAT_EDIFICIO, LON_EDIFICIO, -34.6037, -58.3816)

precio_m2_predicho = predecir_precio_m2(
    surface_total=60, surface_covered=55, rooms=2, bathrooms=1,
    dist_centro_km=dist_centro,
    inc_uva_21=parcela["inc_uva_21"] if pd.notna(parcela["inc_uva_21"]) else 0,
    uni_edif_1=parcela["uni_edif_1"] if pd.notna(parcela["uni_edif_1"]) else 0,
    fot_em_1=parcela["fot_em_1"] if pd.notna(parcela["fot_em_1"]) else 0,
    barrio=parcela["barrio"]
)

print(f"\nPrecio de venta esperado: ${precio_m2_predicho:,.0f}/m²")

# ============================================
# 4. m² construibles — SOLO por altura (uni_edif_1), con factor
#    de eficiencia por retiros/fondo libre de manzana
# ============================================
uni_edif = parcela["uni_edif_1"] if pd.notna(parcela["uni_edif_1"]) and parcela["uni_edif_1"] > 0 else None

if not uni_edif:
    raise ValueError("La parcela no tiene uni_edif_1 válido — revisar caso a mano (puede ser distrito especial).")

m2_construibles_bruto = (uni_edif / ALTURA_POR_PISO) * parcela["superficie_lote_m2"]
m2_construibles = m2_construibles_bruto * FACTOR_EFICIENCIA_CONSTRUCTIVA

print(f"m² construibles (cota superior, solo altura): {m2_construibles_bruto:.0f} m²")
print(f"m² construibles (ajustado, factor {FACTOR_EFICIENCIA_CONSTRUCTIVA}): {m2_construibles:.0f} m²")

# ============================================
# 5. Escenario 1 — quedarse con el edificio tal como está
# ============================================
valor_escenario_1 = M2_YA_CONSTRUIDOS * precio_m2_predicho

# ============================================
# 6. Escenario 2 — demoler y construir de nuevo
# ============================================
costo_m2 = COSTO_CONSTRUCCION_USD_M2[ESCENARIO_COSTO]
costo_demolicion_m2 = costo_m2 * COSTO_DEMOLICION_PCT

ingreso_esperado = m2_construibles * precio_m2_predicho
costo_construccion_total = m2_construibles * costo_m2
margen = ingreso_esperado * MARGEN_DESARROLLADOR
costo_demolicion_total = M2_YA_CONSTRUIDOS * costo_demolicion_m2

valor_escenario_2 = ingreso_esperado - costo_construccion_total - margen - costo_demolicion_total

# ============================================
# 7. Decisión
# ============================================
print("\n" + "="*60)
print(f"Escenario 1 (quedarse como está): ${valor_escenario_1:,.0f}")
print(f"Escenario 2 (demoler y reconstruir): ${valor_escenario_2:,.0f}")
print("="*60)

techo_oferta = max(valor_escenario_1, valor_escenario_2)
mejor_escenario = "Quedarse con el edificio" if valor_escenario_1 >= valor_escenario_2 else "Demoler y reconstruir"

print(f"\n>>> Mejor estrategia: {mejor_escenario}")
print(f">>> Techo de oferta por el edificio: ${techo_oferta:,.0f}")