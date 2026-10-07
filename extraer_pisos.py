import pandas as pd
import re

df = pd.read_csv("properati_caba_con_normativa.csv", low_memory=False)

def extraer_piso(texto):
    if pd.isna(texto):
        return None
    texto = str(texto).lower()

    # Casos especiales primero (antes que el patrón numérico)
    if re.search(r"\bplanta baja\b|\bpb\b(?!\d)", texto):
        return 0
    if re.search(r"\bd[uú]plex\b|\btriplex\b", texto):
        return None  # ambiguo, mejor no asignar un piso único

    # "piso 5", "5to piso", "5° piso", "piso 5to"
    match = re.search(r"piso\s*(\d{1,2})|(\d{1,2})\s*(?:er|do|to|vo|no|mo)?\.?\s*piso", texto)
    if match:
        num = match.group(1) or match.group(2)
        piso = int(num)
        return piso if piso <= 40 else None  # descartar basura (año, dirección, etc.)
    return None

df["piso_extraido"] = df["title"].apply(extraer_piso)
mask_sin_piso = df["piso_extraido"].isna()
df.loc[mask_sin_piso, "piso_extraido"] = df.loc[mask_sin_piso, "description"].apply(extraer_piso)

print("Total filas:", len(df))
print("Con piso extraído:", df["piso_extraido"].notna().sum())
print("Proporción:", round(df["piso_extraido"].notna().mean() * 100, 1), "%")

print("\n=== Distribución de pisos extraídos ===")
print(df["piso_extraido"].value_counts().sort_index().head(20))

print("\n=== Muestra para validar a mano ===")
muestra_valida = df[df["piso_extraido"].notna()].sample(15, random_state=42)
print(muestra_valida[["title", "piso_extraido"]].to_string(index=False))
print("=== Distribución de piso_extraido, SOLO para property_type == 'PH' ===")
ph_con_piso = df[(df["property_type"] == "PH") & (df["piso_extraido"].notna())]
print(f"PH con piso extraído: {len(ph_con_piso)} de {(df['property_type']=='PH').sum()} PH totales")
print(ph_con_piso["piso_extraido"].value_counts().sort_index())

print("\n=== Lo mismo para Departamento, como comparación ===")
depto_con_piso = df[(df["property_type"] == "Departamento") & (df["piso_extraido"].notna())]
print(depto_con_piso["piso_extraido"].value_counts().sort_index().head(15))
casos_cero_sospechosos = df[
    (df["piso_extraido"] == 0) &
    (~df["title"].str.contains(r"\bpb\b|planta baja", case=False, na=False, regex=True))
]
print(f"Casos con piso=0 que NO mencionan PB en el título: {len(casos_cero_sospechosos)}")

for idx in casos_cero_sospechosos.sample(5, random_state=1).index:
    print(f"\nTítulo: {df.loc[idx, 'title']}")
    print(f"Descripción completa:\n{df.loc[idx, 'description']}")
    print("=" * 100)