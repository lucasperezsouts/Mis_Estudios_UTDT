import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("properati_features_clean_v2.csv", low_memory=False)

features_numericas = [
    "surface_total", "surface_covered", "rooms", "bathrooms",
    "ratio_cubierta", "dist_centro_km", "inc_uva_21", "uni_edif_1", "fot_em_1"
]
features_categoricas = ["barrio", "property_type"]
target = "log_price"

cols_necesarias = features_numericas + features_categoricas + [target]
df_modelo = df.dropna(subset=cols_necesarias).copy()

X = df_modelo[features_numericas + features_categoricas]
y = df_modelo[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

paquete = joblib.load("modelo_xgboost.pkl")
modelo = paquete["modelo"]
preprocesador = paquete["preprocesador"]

X_test_proc = preprocesador.transform(X_test)

nombres_cat = preprocesador.named_transformers_["cat"].get_feature_names_out(["property_type"])
nombres_completos = features_numericas + ["barrio_encoded"] + list(nombres_cat)

X_test_df = pd.DataFrame(
    X_test_proc.toarray() if hasattr(X_test_proc, "toarray") else X_test_proc,
    columns=nombres_completos
)

explainer = shap.TreeExplainer(modelo)
muestra = X_test_df.sample(n=min(2000, len(X_test_df)), random_state=42)
shap_values = explainer.shap_values(muestra)

print("Shape de shap_values:", shap_values.shape)

plt.figure()
shap.summary_plot(shap_values, muestra, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Guardado: shap_summary.png")

plt.figure()
shap.summary_plot(shap_values, muestra, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Guardado: shap_bar.png")

importancia_media = np.abs(shap_values).mean(axis=0)
top3_idx = np.argsort(importancia_media)[-3:][::-1]
top3_features = [nombres_completos[i] for i in top3_idx]
print("\nTop 3 features:", top3_features)

for feat in top3_features:
    plt.figure()
    shap.dependence_plot(feat, shap_values, muestra, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_dependence_{feat.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: shap_dependence_{feat}.png")

print("\n✅ SHAP completo.")