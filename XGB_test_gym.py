import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from xgboost import XGBClassifier
import seaborn as sns
import os
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import update_registered_converter
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

os.makedirs("Plots", exist_ok=True)

# =============================================================================
# KONFIGURACJA
# =============================================================================
TRAIN_FILE      = "MLDataContinuous.root"        # dane do treningu
TEST_FILE       = "MLData.root"  # dane do testu (inny plik)
TARGET_SB_RATIO = 10                   # docelowe S/B w teście
TREE_NAME       = "MLDataTree"

# =============================================================================
# 1. WCZYTYWANIE DANYCH
# =============================================================================
with uproot.open(TRAIN_FILE) as f:
    df_train_raw = f[TREE_NAME].arrays(library="pd")

with uproot.open(TEST_FILE) as f:
    df_test_raw = f[TREE_NAME].arrays(library="pd")

n_sig_total = (df_train_raw['IsMuon'] == 1).sum()
n_bkg_total = (df_train_raw['IsMuon'] == 0).sum()
print(f"TRENING  ({TRAIN_FILE}): {len(df_train_raw)} zdarzeń")
print(f"  Sygnał: {n_sig_total}  Tło: {n_bkg_total}  S/B=1:{n_bkg_total/n_sig_total:.1f}")
print(f"TEST     ({TEST_FILE}):  {len(df_test_raw)} zdarzeń")
print(f"  Sygnał: {(df_test_raw['IsMuon']==1).sum()}  "
      f"Tło: {(df_test_raw['IsMuon']==0).sum()}")

# =============================================================================
# 2. ROZWIJANIE WEKTORÓW
# =============================================================================
SHAPE_COLS = ['radius', 'dispersion', 'theta_width', 'phi_width',
              'x_width', 'y_width', 'z_width']

def expand_vector_column(df, col_name, col_labels=SHAPE_COLS):
    expanded = pd.DataFrame(df[col_name].to_list(), index=df.index)
    expanded.columns = [f"{col_name}_{c}" for c in col_labels]
    return expanded

# 3. FEATURE ENGINEERING (identyczne funkcje jak poprzednio)
def engineer_scalar_features(df):
    """Cechy pochodne ze zmiennych skalarnych."""
    d = df[['ECalEnergy', 'HCalEnergy', 'ECalNumber', 'HCalNumber',
            'EoverP', 'Momentum']].copy()

    # Stosunek energii ECal/HCal – kluczowy dla identyfikacji cząstek
    d['ECal_HCal_ratio'] = df['ECalEnergy'] / (df['HCalEnergy'] + 1e-6)

    # Całkowita energia kalorymetryczna
    d['TotalCalEnergy']  = df['ECalEnergy'] + df['HCalEnergy']

    # Gęstość energii (energia / liczba hitów)
    d['ECalDensity']     = df['ECalEnergy'] / (df['ECalNumber'] + 1e-6)
    d['HCalDensity']     = df['HCalEnergy'] / (df['HCalNumber'] + 1e-6)

    # Frakcja energii w ECal (mion: niska, elektron: wysoka)
    d['ECalFraction']    = df['ECalEnergy'] / (d['TotalCalEnergy'] + 1e-6)

    return d


def engineer_shape_features(df_ecal, df_hcal):
    """
    Cechy pochodne z parametrów kształtu showera - bez surowych.

    Fizyczna intuicja:
      Mion (MIP):    mały radius, mała dyspersja, duże z_width (ślad MIP),
                     małe x_width/y_width, symetryczne theta/phi
      Elektron:      duży radius ECal (shower EM), prawie nic w HCal
      Pion/hadron:   duży radius HCal (shower had.), duża dyspersja HCal
    """
    s = pd.DataFrame(index=df_ecal.index)

    # --- Stosunki ECal/HCal ---
    s['radius_ratio']  = df_ecal['EcalShape_radius']     / (df_hcal['HcalShape_radius']     + 1e-6)
    s['disp_ratio']    = df_ecal['EcalShape_dispersion'] / (df_hcal['HcalShape_dispersion'] + 1e-6)
    s['z_width_ratio'] = df_ecal['EcalShape_z_width']    / (df_hcal['HcalShape_z_width']    + 1e-6)

    # --- Rozmiar poprzeczny (x-y) ---
    s['Ecal_transverse']  = np.sqrt(df_ecal['EcalShape_x_width']**2 +
                                    df_ecal['EcalShape_y_width']**2)
    s['Hcal_transverse']  = np.sqrt(df_hcal['HcalShape_x_width']**2 +
                                    df_hcal['HcalShape_y_width']**2)
    s['transverse_ratio'] = s['Ecal_transverse'] / (s['Hcal_transverse'] + 1e-6)

    # --- Stosunek podłużny/poprzeczny – MIP signature ---
    # Mion jest bardzo podłużny: duże z_width, małe x,y
    s['Ecal_long_trans_ratio'] = df_ecal['EcalShape_z_width'] / (s['Ecal_transverse'] + 1e-6)
    s['Hcal_long_trans_ratio'] = df_hcal['HcalShape_z_width'] / (s['Hcal_transverse'] + 1e-6)

    # --- Asymetria kątowa theta/phi ---
    s['Ecal_angular_asym'] = (
        (df_ecal['EcalShape_theta_width'] - df_ecal['EcalShape_phi_width']) /
        (df_ecal['EcalShape_theta_width'] + df_ecal['EcalShape_phi_width'] + 1e-6)
    )
    s['Hcal_angular_asym'] = (
        (df_hcal['HcalShape_theta_width'] - df_hcal['HcalShape_phi_width']) /
        (df_hcal['HcalShape_theta_width'] + df_hcal['HcalShape_phi_width'] + 1e-6)
    )

    # --- Penetracja: jak bardzo shower przechodzi do HCal ---
    s['Shower_leakage'] = (df_hcal['HcalShape_radius'] /
                           (df_ecal['EcalShape_radius'] +
                            df_hcal['HcalShape_radius'] + 1e-6))

    return s

# =============================================================================
# 4. BUDOWANIE ZBIORÓW
# =============================================================================
def build_features(df_raw):
    df_e = expand_vector_column(df_raw, 'EcalShape')
    df_h = expand_vector_column(df_raw, 'HcalShape')
    X = pd.concat([engineer_scalar_features(df_raw),
                   engineer_shape_features(df_e, df_h)], axis=1)
    y = (df_raw['IsMuon'] == 1).astype(int)
    return X, y

X_train, y_train = build_features(df_train_raw)
X_full_test, y_full_test = build_features(df_test_raw)

# Zachowaj realistyczne S/B w teście — ogranicz sygnał
rng = np.random.RandomState(42)
idx_sig_te = np.where(y_full_test == 1)[0]
idx_bkg_te = np.where(y_full_test == 0)[0]

n_sig_test = min(max(1, len(idx_bkg_te) // TARGET_SB_RATIO), len(idx_sig_te))
idx_sig_pick = rng.choice(idx_sig_te, size=n_sig_test, replace=False)
idx_test = np.concatenate([idx_sig_pick, idx_bkg_te])
rng.shuffle(idx_test)

X_test, y_test = X_full_test.iloc[idx_test], y_full_test.iloc[idx_test]

n_sig_train = int(y_train.sum())
n_bkg_train = int((y_train == 0).sum())
n_sig_test  = int(y_test.sum())
n_bkg_test  = int((y_test == 0).sum())
all_features = X_train.columns.tolist()

print(f"\n{'='*55}")
print(f"  TRENING : Sygnał={n_sig_train:>6}  Tło={n_bkg_train:>6}  "
      f"S/B=1:{n_bkg_train/n_sig_train:.1f}")
print(f"  TEST    : Sygnał={n_sig_test:>6}  Tło={n_bkg_test:>6}  "
      f"S/B=1:{n_bkg_test/n_sig_test:.1f}  ← realistyczne")
print(f"  (pominięto {len(idx_sig_te)-n_sig_test} eventów sygnału z pliku testowego)")
print(f"{'='*55}")

# Skalowanie — fit TYLKO na treningu
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# zmienna dla raportu tekstowego
file_path = f"Train: {TRAIN_FILE} | Test: {TEST_FILE}"
df = df_train_raw  # dla len(df) w raporcie

# =============================================================================
# 5. XGBOOST – TRENING
#    Trening na 1:1 → scale_pos_weight ≈ 1 (brak korekcji wagowej)
#    Test na 1:10  → progi cięcia będą kalibrowane pod realistyczny S/B
# =============================================================================
scale_pos_weight = n_bkg_train / n_sig_train  # ≈ 1.0 przy 1:1

xgb = XGBClassifier(
    # --- Architektura ---
    n_estimators          = 3000,
    learning_rate         = 0.05,
    max_depth             = 4,

    # --- Regularyzacja ---
    min_child_weight      = 5,        # można zmniejszyć bo trening 1:1 jest stabilny
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    gamma                 = 1.0,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,

    # --- Wagi klas (≈1 bo trening 1:1) ---
    scale_pos_weight      = scale_pos_weight,

    # --- Technikalia ---
    eval_metric           = 'auc',
    early_stopping_rounds = 30,
    tree_method           = 'hist',
    random_state          = 42,
    n_jobs                = -1,
    verbosity             = 1,
)

xgb.fit(
    X_train_sc, y_train,
    eval_set=[(X_train_sc, y_train), (X_test_sc, y_test)],
    verbose=50,
)

y_pred  = xgb.predict(X_test_sc)
y_probs = xgb.predict_proba(X_test_sc)[:, 1]

best_iteration = xgb.best_iteration
print(f"\nNajlepszy iteration (early stopping): {best_iteration}")

n_features = X_train_sc.shape[1]
onnx_model = convert_xgboost(
    xgb,
    initial_types=[('input', OnnxFloat([None, n_features]))]
)
onnx_path = "Plots/xgb_muonID.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"Model ONNX zapisany: {onnx_path}")

# Szybka weryfikacja — sprawdź czy ONNX daje te same wyniki
import onnxruntime as rt
sess    = rt.InferenceSession(onnx_path)
inp     = sess.get_inputs()[0].name
onnx_probs = sess.run(None, {inp: X_test_sc.astype(np.float32)})[1]
onnx_probs = np.array([p[1] for p in onnx_probs])
max_diff = np.max(np.abs(onnx_probs - y_probs))
print(f"Max różnica XGB vs ONNX: {max_diff:.2e}  {'✓ OK' if max_diff < 1e-4 else '⚠ SPRAWDŹ'}")

# =============================================================================
# 6. METRYKI I PROGI CIĘCIA
# =============================================================================
fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr_arr, tpr_arr)

prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_test, y_probs)
f1_arr         = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
best_f1_idx    = np.argmax(f1_arr[:-1])
best_f1_thresh = pr_thresholds[best_f1_idx]
ap             = average_precision_score(y_test, y_probs)

# Próg przy 95% odcięcia tła (FPR = 0.05)
idx_95        = np.argmin(np.abs(fpr_arr - 0.05))
thresh_95bkg  = roc_thresholds[idx_95]
sig_eff_at_95 = tpr_arr[idx_95]

# Próg przy 99% odcięcia tła (FPR = 0.01) – bardziej agresywny
idx_99        = np.argmin(np.abs(fpr_arr - 0.01))
thresh_99bkg  = roc_thresholds[idx_99]
sig_eff_at_99 = tpr_arr[idx_99]

print(f"\nROC AUC             : {roc_auc:.4f}")
print(f"Average Precision   : {ap:.4f}")
print(f"Próg max-F1         : {best_f1_thresh:.3f}  (F1={f1_arr[best_f1_idx]:.3f})")
print(f"Próg 95% rej. tła   : {thresh_95bkg:.3f}  (sig. eff.={sig_eff_at_95:.3f})")
print(f"Próg 99% rej. tła   : {thresh_99bkg:.3f}  (sig. eff.={sig_eff_at_99:.3f})")

# Historia treningu
evals          = xgb.evals_result()
train_auc_hist = evals['validation_0']['auc']
val_auc_hist   = evals['validation_1']['auc']

# =============================================================================
# 7. GENEROWANIE PDF
# =============================================================================
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'font.family':      'sans-serif',
    'axes.spines.top':  False,
    'axes.spines.right':    False,
})

SIG_COLOR = '#1f77b4'   
BKG_COLOR = "#ff0e0e"   
ACC_COLOR = '#2ca02c'   
PUR_COLOR = '#9467bd'   

with PdfPages("Plots/XGB_Output_test"+TEST_FILE+".pdf") as pdf:

    # =========================================================================
    # STRONA 1: METRYKI PODSTAWOWE
    # =========================================================================
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Muon Identification — XGBoost Analysis\n'
                 f'[Train S/B=1:{n_bkg_train/n_sig_train:.0f}  |  '
                 f'Test S/B=1:{n_bkg_test/n_sig_test:.0f} (realistic)]',
                 fontsize=16, y=0.98, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # --- Macierz pomyłek @ 95% bkg rejection ---
    ax_cm     = fig.add_subplot(gs[0, 0])
    y_pred_95 = (y_probs >= thresh_95bkg).astype(int)
    cm_mat    = confusion_matrix(y_test, y_pred_95)
    cm_norm   = cm_mat.astype(float) / cm_mat.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', ax=ax_cm,
                cmap='Blues', linewidths=0.5,
                xticklabels=['Background', 'Muon'],
                yticklabels=['Background', 'Muon'],
                cbar_kws={'label': 'Fraction'})
    ax_cm.set_title(f'Confusion Matrix\n@ 95% bkg rejection (thr={thresh_95bkg:.2f})',
                    color=SIG_COLOR, pad=10)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    for i in range(2):
        for j in range(2):
            ax_cm.text(j+0.5, i+0.72, f'({cm_mat[i,j]})',
                       ha='center', va='center', fontsize=9)

    # --- Krzywa ROC ---
    ax_roc = fig.add_subplot(gs[0, 1])
    ax_roc.plot(fpr_arr, tpr_arr, color=SIG_COLOR, lw=2,
                label=f'AUC = {roc_auc:.4f}')
    ax_roc.fill_between(fpr_arr, tpr_arr, alpha=0.1, color=SIG_COLOR)
    ax_roc.plot([0,1],[0,1],linestyle='--', lw=1)
    ax_roc.axvline(0.05, color=ACC_COLOR, linestyle=':', lw=1.5,
                   label=f'FPR=5%  (TPR={sig_eff_at_95:.2f})')
    ax_roc.axvline(0.01, color=PUR_COLOR, linestyle=':', lw=1.5,
                   label=f'FPR=1%  (TPR={sig_eff_at_99:.2f})')
    ax_roc.set_title('ROC Curve', color=SIG_COLOR, pad=10)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(fontsize=9)
    ax_roc.grid(True)

    # --- Precision-Recall ---
    ax_pr = fig.add_subplot(gs[0, 2])
    ax_pr.plot(rec_arr, prec_arr, color=BKG_COLOR, lw=2,
               label=f'AP = {ap:.4f}')
    ax_pr.fill_between(rec_arr, prec_arr, alpha=0.1, color=BKG_COLOR)
    ax_pr.scatter(rec_arr[best_f1_idx], prec_arr[best_f1_idx],
                  color=ACC_COLOR, s=80, zorder=5,
                  label=f'Best F1={f1_arr[best_f1_idx]:.3f}\n(thr={best_f1_thresh:.2f})')
    ax_pr.set_title('Precision-Recall Curve', color=SIG_COLOR, pad=10)
    ax_pr.set_xlabel('Recall (Signal Efficiency)')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(fontsize=9)
    ax_pr.grid(True)

    # --- Rozkład odpowiedzi klasyfikatora ---
    ax_resp = fig.add_subplot(gs[1, 0:2])
    bins = np.linspace(0, 1, 50)
    ax_resp.hist(y_probs[y_test==0], bins=bins, alpha=0.6, density=True,
                 color=BKG_COLOR, label=f'Background (N={n_bkg_test})',
                 hatch='//', edgecolor=BKG_COLOR)
    ax_resp.hist(y_probs[y_test==1], bins=bins, alpha=0.6, density=True,
                 color=SIG_COLOR, label=f'Muon/Signal (N={n_sig_test})')
    ax_resp.axvline(best_f1_thresh, color=ACC_COLOR, linestyle='--', lw=2,
                    label=f'Cut (max-F1): {best_f1_thresh:.2f}')
    ax_resp.axvline(thresh_95bkg, color=PUR_COLOR, linestyle=':', lw=2,
                    label=f'Cut (95% bkg rej.): {thresh_95bkg:.2f}')
    ax_resp.axvline(thresh_99bkg, linestyle=':', lw=1.5,
                    label=f'Cut (99% bkg rej.): {thresh_99bkg:.2f}')
    ax_resp.set_title('Classifier Response Distribution  [Test: realistic S/B]',
                      color=SIG_COLOR, pad=10)
    ax_resp.set_xlabel('P(Muon)')
    ax_resp.set_ylabel('Normalized counts (log)')
    ax_resp.legend(fontsize=8)
    ax_resp.grid(True)
    ax_resp.set_yscale('log')

    # --- Efektywność sygnału vs odrzucenie tła ---
    ax_eff = fig.add_subplot(gs[1, 2])
    bkg_rej = 1 - fpr_arr
    ax_eff.plot(bkg_rej, tpr_arr, color=PUR_COLOR, lw=2)
    ax_eff.axvline(0.95, color=ACC_COLOR, linestyle=':', lw=1.5,
                   label=f'95% bkg rej.\nSig eff={sig_eff_at_95:.2f}')
    ax_eff.axvline(0.99, linestyle=':', lw=1.5,
                   label=f'99% bkg rej.\nSig eff={sig_eff_at_99:.2f}')
    ax_eff.set_title('Signal Eff. vs Background Rejection', color=SIG_COLOR, pad=10)
    ax_eff.set_xlabel('Background Rejection (1 - FPR)')
    ax_eff.set_ylabel('Signal Efficiency (TPR)')
    ax_eff.legend(fontsize=9)
    ax_eff.grid(True)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # STRONA 2: HISTORIA TRENINGU + OVERTRAINING CHECK
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('XGBoost Training Diagnostics',
                 fontsize=16, color='#ccd6f6', y=1.01, fontweight='bold')

    # Historia AUC train vs val
    iters = range(1, len(train_auc_hist) + 1)
    axes[0].plot(iters, train_auc_hist, color=SIG_COLOR, lw=1.5, label='Train AUC')
    axes[0].plot(iters, val_auc_hist,   color=BKG_COLOR, lw=1.5, label='Val AUC (test realistic)')
    axes[0].axvline(best_iteration, color=ACC_COLOR, linestyle='--', lw=1.5,
                    label=f'Best iter: {best_iteration}')
    axes[0].set_title('AUC vs Boosting Rounds', color=SIG_COLOR, pad=10)
    axes[0].set_xlabel('Boosting round')
    axes[0].set_ylabel('AUC')
    axes[0].legend(fontsize=10)
    axes[0].grid(True)

    # Overtraining check
    y_probs_train = xgb.predict_proba(X_train_sc)[:, 1]
    bins = np.linspace(0, 1, 40)
    axes[1].hist(y_probs_train[y_train==0], bins=bins, alpha=0.4, density=True,
                 color=BKG_COLOR, label='Bkg (train)', hatch='//')
    axes[1].hist(y_probs[y_test==0],        bins=bins, alpha=0.7, density=True,
                 color=BKG_COLOR, label='Bkg (test)',
                 histtype='step', linewidth=2)
    axes[1].hist(y_probs_train[y_train==1], bins=bins, alpha=0.4, density=True,
                 color=SIG_COLOR, label='Signal (train)')
    axes[1].hist(y_probs[y_test==1],        bins=bins, alpha=0.7, density=True,
                 color=SIG_COLOR, label='Signal (test)',
                 histtype='step', linewidth=2)
    axes[1].set_title('Overtraining Check (Train vs Test)', color=SIG_COLOR, pad=10)
    axes[1].set_xlabel('P(Muon)')
    axes[1].set_ylabel('Normalized counts (log)')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=9)
    axes[1].grid(True)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # STRONA 3: WAŻNOŚĆ CECH
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Feature Importances — XGBoost',
                 fontsize=16, y=1.01, fontweight='bold')

    importances = xgb.feature_importances_
    indices_all = np.argsort(importances)
    top_n       = min(20, len(all_features))
    idx_top     = indices_all[-top_n:]

    colors = [SIG_COLOR if 'Ecal' in all_features[i] or 'ECal' in all_features[i]
              else BKG_COLOR if 'Hcal' in all_features[i] or 'HCal' in all_features[i]
              else ACC_COLOR
              for i in idx_top]

    axes[0].barh(range(top_n), importances[idx_top], color=colors,
                 align='center', edgecolor='#0f1117', linewidth=0.5)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels([all_features[i] for i in idx_top], fontsize=9)
    axes[0].set_title(f'Top {top_n} Features', color=SIG_COLOR, pad=10)
    axes[0].set_xlabel('Importance (gain)')
    axes[0].grid(True, axis='x')
    legend_elements = [Patch(facecolor=SIG_COLOR, label='ECal features'),
                       Patch(facecolor=BKG_COLOR, label='HCal features'),
                       Patch(facecolor=ACC_COLOR, label='Combined/scalar')]
    axes[0].legend(handles=legend_elements, fontsize=9, loc='lower right')

    # Skumulowana ważność
    sorted_imp = np.sort(importances)[::-1]
    cum_imp    = np.cumsum(sorted_imp)
    n_feats_90 = np.argmax(cum_imp >= 0.90) + 1
    axes[1].plot(range(1, len(sorted_imp)+1), cum_imp, color=SIG_COLOR, lw=2)
    axes[1].axhline(0.90, color=ACC_COLOR, linestyle='--', lw=1.5,
                    label=f'90% importance ({n_feats_90} features)')
    axes[1].axhline(0.95, color=PUR_COLOR, linestyle=':', lw=1.5,
                    label='95% importance')
    axes[1].fill_between(range(1, len(sorted_imp)+1), cum_imp,
                         alpha=0.15, color=SIG_COLOR)
    axes[1].set_title('Cumulative Feature Importance', color=SIG_COLOR, pad=10)
    axes[1].set_xlabel('Number of features')
    axes[1].set_ylabel('Cumulative importance')
    axes[1].legend(fontsize=10)
    axes[1].grid(True)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # STRONA 4: ROZKŁADY FIZYCZNE KLUCZOWYCH CECH
    # =========================================================================
    phys_features = [
        'ECal_HCal_ratio', 'ECalFraction',        'TotalCalEnergy',
        'ECalDensity',      'HCalDensity',         'EoverP',
        'radius_ratio',     'disp_ratio',          'z_width_ratio',
        'Ecal_long_trans_ratio', 'Hcal_long_trans_ratio', 'Shower_leakage',
    ]
    phys_features = [f for f in phys_features if f in X_train.columns]

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle('Key Physics Variable Distributions: Signal vs Background\n'
                 '[rozkłady z całego datasetu przed podziałem]',
                 fontsize=14, color='#ccd6f6', y=1.01, fontweight='bold')
    axes = axes.flatten()

    for ax, feat in zip(axes, phys_features):
        sig_vals = X_train.loc[y_train==1, feat].dropna()
        bkg_vals = X_train.loc[y_train==0, feat].dropna()
        lo   = np.percentile(pd.concat([sig_vals, bkg_vals]), 1)
        hi   = np.percentile(pd.concat([sig_vals, bkg_vals]), 99) # type: ignore
        bins = np.linspace(lo, hi, 50)
        ax.hist(bkg_vals.clip(lo, hi), bins=bins, alpha=0.6, density=True,
                color=BKG_COLOR, label='Background', hatch='//', edgecolor=BKG_COLOR)
        ax.hist(sig_vals.clip(lo, hi), bins=bins, alpha=0.6, density=True,
                color=SIG_COLOR, label='Signal')
        ax.set_title(feat, color=SIG_COLOR, fontsize=9, pad=5)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Norm.',  fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7)

    for ax in axes[len(phys_features):]:
        ax.set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # STRONA 5: RAPORT TEKSTOWY
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))

    # Classification report przy progu 95% bkg rejection
    report_95 = classification_report(y_test, y_pred_95,
                                      target_names=['Background', 'Muon'])

    text_content = f"""
╔══════════════════════════════════════════════════════════════════╗
║           Physics Analysis — Muon Identification Report          ║
║                        [ XGBoost ]                               ║
╚══════════════════════════════════════════════════════════════════╝

  Input file      : {file_path}
  Total events    : {len(df)}
  Signal (muons)  : {n_sig_total}  ({100*n_sig_total/len(df):.1f}%)
  Background      : {n_bkg_total}  ({100*n_bkg_total/len(df):.1f}%)
  Features used   : {len(all_features)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  STRATEGIA PODZIAŁU:
    Trening  : 80% sygnału + 80% tła   →  S/B = 1:{n_bkg_train/n_sig_train:.1f}
               Sygnał={n_sig_train}  Tło={n_bkg_train}
    Test     : 10% puli syg. + 100% puli tła  →  S/B = 1:{n_bkg_test/n_sig_test:.1f}  (realistyczne)
               Sygnał={n_sig_test}   Tło={n_bkg_test}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MODEL: XGBoost
    n_estimators      = 500  (best iter: {best_iteration})
    learning_rate     = 0.05
    max_depth         = 4
    min_child_weight  = 5
    subsample         = 0.8   colsample_bytree = 0.8
    gamma             = 1.0   scale_pos_weight = {scale_pos_weight:.2f}
    reg_alpha         = 0.1   reg_lambda       = 1.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ROC AUC             : {roc_auc:.4f}
  Average Precision   : {ap:.4f}

  Cut @ max-F1        : {best_f1_thresh:.3f}   (F1 = {f1_arr[best_f1_idx]:.3f})
  Cut @ 95% bkg rej.  : {thresh_95bkg:.3f}   →  signal eff. = {sig_eff_at_95:.3f}
  Cut @ 99% bkg rej.  : {thresh_99bkg:.3f}   →  signal eff. = {sig_eff_at_99:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Classification report @ 95% bkg rejection threshold:

{report_95}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Feature groups:
    ECal shape (pochodne) : {sum(1 for f in all_features if 'Ecal' in f)} features
    HCal shape (pochodne) : {sum(1 for f in all_features if 'Hcal' in f)} features
    Scalar + derived      : {sum(1 for f in all_features if 'Ecal' not in f and 'Hcal' not in f)} features
"""

    plt.text(0.03, 0.97, text_content, transform=fig.transFigure,
             fontsize=9.5, family='monospace', color='#ccd6f6',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1d27', alpha=0.8,
                       edgecolor='#444466'))
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

plt.rcdefaults()
print("\nAnaliza zakończona. PDF zapisany jako 'Plots/XGB_Output.pdf'.")
print("  Strony:")
print("    1 = Metryki podstawowe (ROC, PR, confusion matrix, response)")
print("    2 = Historia treningu + overtraining check")
print("    3 = Feature importances")
print("    4 = Rozkłady fizyczne")
print("    5 = Raport tekstowy")