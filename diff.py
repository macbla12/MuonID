import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sns
import os

# ============================================================
# 1. WCZYTANIE DANYCH
# ============================================================

file_path = "ONNX/MLData.root"

print("Wczytywanie ROOT...")
with uproot.open(file_path) as f:
    df = f["MLDataTree"].arrays(library="pd")

print("Gotowe. Liczba zdarzeń:", len(df))

# ============================================================
# 2. WYBÓR MIONÓW
# ============================================================

mu = df[df["IsMuon"] == 1].copy()
print("Liczba mionów:", len(mu))

# ============================================================
# 3. DEFINICJA DWÓCH PRÓBEK DO PORÓWNANIA
# ============================================================
# Przykład: FileIndex 0 i 1 vs FileIndex 2 i 3
# ZMIEŃ TO NA SWOJE DWIE PRÓBKI

sampleA = mu[mu["FileIndex"].isin([0])]
sampleB = mu[mu["FileIndex"].isin([1])]

print("Sample A:", len(sampleA))
print("Sample B:", len(sampleB))

# ============================================================
# 4. LISTA CECH DO ANALIZY
# ============================================================

FEATURES = [
    "ECalEnergy", "HCalEnergy",
    "ECalNumber", "HCalNumber",
    "ECalEoverP", "HCalEoverP",
    "ECalFrac", "HCalFrac",
    "ECalDensity", "HCalDensity",
    "logECal", "logHCal",
    "HcalShape_phi_width", "HcalShape_lambda2",
    "HcalShape_lambda3", "HcalShape_theta_width",
    "HcalShape_radius", "HcalShape_dispersion",
    "EcalShape_phi_width", "EcalShape_lambda2",
    "EcalShape_lambda3", "EcalShape_theta_width",
    "EcalShape_radius", "EcalShape_dispersion"
]

# ============================================================
# 5. TWORZENIE FOLDERU NA WYKRESY
# ============================================================

os.makedirs("MuonDiffPlots", exist_ok=True)

# ============================================================
# 6. ANALIZA RÓŻNIC
# ============================================================

report = []

for col in FEATURES:
    if col not in mu.columns:
        continue

    A = sampleA[col].dropna()
    B = sampleB[col].dropna()

    # KS test
    ks_stat, ks_p = ks_2samp(A, B)

    # różnice statystyczne
    meanA, meanB = A.mean(), B.mean()
    medA, medB = A.median(), B.median()

    report.append((col, ks_stat, ks_p, meanA, meanB, medA, medB))

    # wykres
    plt.figure(figsize=(7,5))
    sns.histplot(A, bins=60, stat="density", color="blue", label="Sample A", element="step")
    sns.histplot(B, bins=60, stat="density", color="red", label="Sample B", element="step")
    plt.title(f"{col} (muons)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"MuonDiffPlots/{col}.png")
    plt.close()

# ============================================================
# 7. RAPORT TEKSTOWY
# ============================================================

report_df = pd.DataFrame(report, columns=[
    "Feature", "KS_stat", "KS_pvalue", "Mean_A", "Mean_B", "Median_A", "Median_B"
])

report_df.sort_values("KS_stat", ascending=False, inplace=True)
report_df.to_csv("MuonDiffPlots/muon_feature_differences.csv", index=False)

print("\n=== ANALIZA ZAKOŃCZONA ===")
print("Wyniki zapisane w folderze MuonDiffPlots/")
print("Najbardziej różniące się cechy:")
print(report_df.head(10))
