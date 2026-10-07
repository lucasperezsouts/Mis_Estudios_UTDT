import pandas as pd
import numpy as np

df = pd.read_csv("properati_features_clean_dedup.csv", low_memory=False)
print("Shape inicial:", df.shape)

antes = len(df)
df = df[df["currency"] == "USD"].copy()
print(f"Descartadas por currency != USD: {antes - len(df)}")

antes = len(df)
df = df[df["ratio_cubierta"].isna() | (df["ratio_cubierta"] <= 1.05)].copy()
print(f"Descartadas por ratio_cubierta > 1.05 (solo valores reales, NaN se preserva): {antes - len(df)}")

antes = len(df)
df = df[(df["rooms"].isna() | (df["rooms"] <= 10)) &
        (df["bathrooms"].isna() | (df["bathrooms"] <= 6))].copy()
print(f"Descartadas por rooms/bathrooms absurdos: {antes - len(df)}")

antes = len(df)
df = df[(df["surface_total"] >= 10) & (df["surface_total"] <= 1000)].copy()
print(f"Descartadas por surface_total fuera de [10, 1000]: {antes - len(df)}")

antes = len(df)
conteo_tipo = df["property_type"].value_counts()
tipos_validos = conteo_tipo[conteo_tipo >= 30].index
df = df[df["property_type"].isin(tipos_validos)].copy()
print(f"Descartadas por property_type con <30 casos: {antes - len(df)}")

print("\nShape final:", df.shape)
df.to_csv("properati_features_clean_v2.csv", index=False)
print("Guardado: properati_features_clean_v2.csv")