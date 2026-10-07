import os
import zipfile
import requests
import pandas as pd
import geopandas as gpd

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# ============================================
# 1. Geometría: dataset "Parcelas" (SHP) — este SÍ viene con
#    el CRS correcto (confirmado: bounding box real de CABA).
#    El SHP del "Código Urbanístico" tiene el CRS mal etiquetado
#    (bug documentado también por otros proyectos que usan estos
#    mismos datos abiertos de CABA).
# ============================================
PARCELAS_ZIP_URL = "https://data.buenosaires.gob.ar/dataset/parcelas/resource/juqdkmgo-1562-resource/download"
PARCELAS_ZIP_PATH = "parcelas.zip"
PARCELAS_EXTRACT_DIR = "parcelas-shp"

if not os.path.exists(PARCELAS_ZIP_PATH):
    print("Descargando Parcelas (zip)...", flush=True)
    r = requests.get(PARCELAS_ZIP_URL, headers=headers, stream=True, timeout=120)
    r.raise_for_status()
    with open(PARCELAS_ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

if not os.path.exists(PARCELAS_EXTRACT_DIR):
    with zipfile.ZipFile(PARCELAS_ZIP_PATH, "r") as z:
        z.extractall(PARCELAS_EXTRACT_DIR)

PARCELAS_SHP = os.path.join(PARCELAS_EXTRACT_DIR, "parcelas_catastrales.shp")
parcelas_geom = gpd.read_file(PARCELAS_SHP)

print("=== Parcelas (geometría) ===")
print("Cantidad:", len(parcelas_geom))
print("CRS:", parcelas_geom.crs)
print("Bounding box:", parcelas_geom.total_bounds)

# ============================================
# 2. Atributos normativos: CSV del Código Urbanístico, unidos
#    por SMP (normalizando el formato, que varía entre datasets:
#    "087-006A-015", "87-006A-015", "055 - 200 - 005", etc.)
# ============================================
atributos = pd.read_csv("codigo-urbanistico.csv", low_memory=False)

def normalizar_smp(x):
    return str(x).strip().replace(" ", "").upper()

col_smp_csv = "smp1" if "smp1" in atributos.columns else "smp"

parcelas_geom["smp_norm"] = parcelas_geom["smp"].apply(normalizar_smp)
atributos["smp_norm"] = atributos[col_smp_csv].apply(normalizar_smp)

parcelas = parcelas_geom.merge(
    atributos.drop(columns=["barrio", "comuna"], errors="ignore"),
    on="smp_norm",
    how="left"
)

print("\n=== Parcelas unidas (geometría + normativa) ===")
print("Cantidad:", len(parcelas))
print("Con normativa matcheada:", parcelas["uni_edif_1"].notna().sum())

# ============================================
# 3. Properati: filtrar a CABA + Venta, con el fix de lat/lon
#    invertidas (bug conocido del dataset de Kaggle: la columna
#    "lat" en realidad tiene longitud, y "lon" tiene latitud —
#    confirmado filtrando Palermo a mano).
# ============================================
properati = pd.read_csv("properati.csv")

properati_caba = properati[
    (properati["operation_type"] == "Venta") &
    (properati["l2"] == "Capital Federal") &
    (properati["lat"].notna()) &
    (properati["lon"].notna())
].copy()

properati_geo = gpd.GeoDataFrame(
    properati_caba,
    geometry=gpd.points_from_xy(properati_caba["lat"], properati_caba["lon"]),  # invertido a propósito
    crs="EPSG:4326"
)

print("\n=== Properati filtrado ===")
print("Filas:", len(properati_caba))
print("Bounding box:", properati_geo.total_bounds)

# ============================================
# 4. Spatial join CON buffer de 15m — las parcelas catastrales
#    no cubren calles/veredas, y confirmamos que la mediana de
#    distancia de los puntos "sin match" al polígono más cercano
#    era de apenas 5.7m (percentil 75: 11m) — o sea, error de
#    precisión de geocoding / borde de vereda, no un problema real.
# ============================================
parcelas_metros = parcelas.to_crs(epsg=3857)   # reproyectar a un CRS métrico
parcelas_buffer = parcelas_metros.copy()
parcelas_buffer["geometry"] = parcelas_metros.geometry.buffer(15)  # 15 metros
parcelas_buffer = parcelas_buffer.to_crs(epsg=4326)  # volver a grados para el join

df = gpd.sjoin(properati_geo, parcelas_buffer, how="left", predicate="within")

# Un punto puede caer en el buffer de más de una parcela (cerca de
# esquinas) — nos quedamos con la primera ocurrencia de cada fila
df = df[~df.index.duplicated(keep="first")]

print("\n=== Resultado del spatial join (con buffer 15m) ===")
print("Matcheadas:", df["index_right"].notna().sum())
print("Sin match:", df["index_right"].isna().sum())
print(df[["price", "surface_total", "barrio", "uni_edif_1", "fot_em_1", "inc_uva_21"]].head(10))

# ============================================
# 5. Guardar el resultado para no tener que rehacer todo esto
#    cada vez que sigamos trabajando en el modelo
# ============================================
df_modelo = df[df["index_right"].notna()].copy()
df_modelo.to_csv("properati_caba_con_normativa.csv", index=False)
print(f"\nGuardado: properati_caba_con_normativa.csv ({len(df_modelo)} filas)")