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
# 1. WCZYTYWANIE DANYCH
# =============================================================================
#file_path = "MLDataContinuous.root"
file_path = "MLDataGrape.root"


with uproot.open(file_path) as f:
    df = f["MLDataTree"].arrays(library="pd")

n_sig_total = (df['IsMuon'] == 1).sum()
n_bkg_total = (df['IsMuon'] == 0).sum()
print(f"Wczytano {len(df)} zdarzeń.")
print(f"  Sygnał (miony) : {n_sig_total}")
print(f"  Tło            : {n_bkg_total}")
print(f"  Stosunek S/B   : 1 : {n_bkg_total/n_sig_total:.1f}")

# =============================================================================
# 2. ROZWIJANIE WEKTORÓW Z SEMANTYCZNYMI NAZWAMI
# =============================================================================
SHAPE_COLS = ['radius', 'dispersion', 'theta_width', 'phi_width',
              'x_width', 'y_width', 'z_width']

def expand_vector_column(df, col_name, col_labels=SHAPE_COLS):
    expanded = pd.DataFrame(df[col_name].to_list(), index=df.index)
    expanded.columns = [f"{col_name}_{c}" for c in col_labels]
    return expanded

df_ecal = expand_vector_column(df, 'EcalShape')
df_hcal = expand_vector_column(df, 'HcalShape')

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def safe_divide(num, denom):
    denom_nonzero = denom != 0
    result = np.zeros_like(num, dtype=float)
    result[denom_nonzero] = num[denom_nonzero] / denom[denom_nonzero]
    return result


def engineer_scalar_features(df):
    d = df[['ECalEnergy', 'HCalEnergy', 'ECalNumber', 'HCalNumber',
            'ECalEoverP','HCalEoverP']].copy()
    d['ECal_HCal_ratio'] = safe_divide(df['ECalEnergy'], df['HCalEnergy'])
    d['TotalCalEnergy']  = df['ECalEnergy'] + df['HCalEnergy']
    d['ECalDensity']     = safe_divide(df['ECalEnergy'], df['ECalNumber'])
    d['HCalDensity']     = safe_divide(df['HCalEnergy'], df['HCalNumber'])
    d['HCalFraction']    = safe_divide(df['HCalEnergy'], d['TotalCalEnergy'])
    return d


def engineer_shape_features(df_ecal, df_hcal):
    s = pd.DataFrame(index=df_ecal.index)
    s['radius_ratio']  = safe_divide(df_ecal['EcalShape_radius'], df_hcal['HcalShape_radius'])
    s['disp_ratio']    = safe_divide(df_ecal['EcalShape_dispersion'], df_hcal['HcalShape_dispersion'])
    s['z_width_ratio'] = safe_divide(df_ecal['EcalShape_z_width'], df_hcal['HcalShape_z_width'])
    s['Ecal_transverse']  = np.sqrt(df_ecal['EcalShape_x_width']**2 +
                                    df_ecal['EcalShape_y_width']**2)
    s['Hcal_transverse']  = np.sqrt(df_hcal['HcalShape_x_width']**2 +
                                    df_hcal['HcalShape_y_width']**2)
    s['transverse_ratio'] = safe_divide(s['Ecal_transverse'], s['Hcal_transverse'])
    s['Ecal_long_trans_ratio'] = safe_divide(df_ecal['EcalShape_z_width'], s['Ecal_transverse'])
    s['Hcal_long_trans_ratio'] = safe_divide(df_hcal['HcalShape_z_width'], s['Hcal_transverse'])
    s['Ecal_angular_asym'] = safe_divide(
        (df_ecal['EcalShape_theta_width'] - df_ecal['EcalShape_phi_width']),
        (df_ecal['EcalShape_theta_width'] + df_ecal['EcalShape_phi_width'])
    )
    s['Hcal_angular_asym'] = safe_divide(
        (df_hcal['HcalShape_theta_width'] - df_hcal['HcalShape_phi_width']),
        (df_hcal['HcalShape_theta_width'] + df_hcal['HcalShape_phi_width'])
    )
    s['Radial_HCal_Fraction'] = safe_divide(
        df_hcal['HcalShape_radius'],
        (df_ecal['EcalShape_radius'] + df_hcal['HcalShape_radius'])
    )
    s = pd.concat([s, df_ecal, df_hcal], axis=1)
    return s


X_scalar = engineer_scalar_features(df)
X_shape  = engineer_shape_features(df_ecal, df_hcal)
X        = pd.concat([X_scalar, X_shape], axis=1)
y        = (df['IsMuon'] == 1).astype(int)

print(f"\nŁączna liczba cech: {X.shape[1]}")
print(f"  Skalarne + pochodne : {X_scalar.shape[1]}")
print(f"  Shape pochodne      : {X_shape.shape[1]}")
all_features = X.columns.tolist()

# =============================================================================
# 4. PODZIAŁ DANYCH
# =============================================================================
rng = np.random.RandomState(42)

idx_sig = np.where(y == 1)[0]
idx_bkg = np.where(y == 0)[0]

n_sig_test_pool = int(len(idx_sig) * 0.20)
n_bkg_test_pool = int(len(idx_bkg) * 0.20)

idx_sig_test_pool = rng.choice(idx_sig, size=n_sig_test_pool, replace=False)
idx_bkg_test_pool = rng.choice(idx_bkg, size=n_bkg_test_pool, replace=False)

idx_sig_train = np.setdiff1d(idx_sig, idx_sig_test_pool)
idx_bkg_train = np.setdiff1d(idx_bkg, idx_bkg_test_pool)

n_bkg_test_available = len(idx_bkg_test_pool)
n_sig_test_target    = max(1, n_bkg_test_available //1)

n_sig_test   = min(n_sig_test_target, len(idx_sig_test_pool))
idx_sig_test = rng.choice(idx_sig_test_pool, size=n_sig_test, replace=False)
idx_bkg_test = idx_bkg_test_pool

idx_train = np.concatenate([idx_sig_train, idx_bkg_train])
idx_test  = np.concatenate([idx_sig_test,  idx_bkg_test])
rng.shuffle(idx_train)
rng.shuffle(idx_test)

X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
X_test,  y_test  = X.iloc[idx_test],  y.iloc[idx_test]

n_sig_train = int(y_train.sum())
n_bkg_train = int((y_train == 0).sum())
n_sig_test  = int(y_test.sum())
n_bkg_test  = int((y_test == 0).sum())

print(f"\n{'='*55}")
print(f"  TRENING : Sygnał={n_sig_train:>5}  Tło={n_bkg_train:>5}  "
      f"S/B = 1:{n_bkg_train/n_sig_train:.1f}")
print(f"  TEST    : Sygnał={n_sig_test:>5}  Tło={n_bkg_test:>5}  "
      f"S/B = 1:{n_bkg_test/n_sig_test:.1f}  ")
print(f"{'='*55}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
means = scaler.mean_
scales = scaler.scale_

print("Wklej to do C++ jako scaler_mean:")
print("{" + ", ".join([f"{x:.8f}f" for x in means]) + "};")

print("\nWklej to do C++ jako scaler_scale:")
print("{" + ", ".join([f"{x:.8f}f" for x in scales]) + "};")

# =============================================================================
# 5. XGBOOST – TRENING
# =============================================================================
scale_pos_weight = n_bkg_train / n_sig_train

xgb = XGBClassifier(
    n_estimators          = 2000,
    learning_rate         = 0.05,
    max_depth             = 4,
    min_child_weight      = 5,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    gamma                 = 1.0,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    scale_pos_weight      = scale_pos_weight,
    eval_metric           = 'auc',
    early_stopping_rounds = 30,
    tree_method           = 'hist',
    random_state          = 42,
    n_jobs                = -1,
    verbosity             = 1,
)

fp_penalty = 3.0
sample_weights = np.where(y_train == 0, fp_penalty, 1.0)

xgb.fit(
    X_train_sc, y_train,
    sample_weight=sample_weights,
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

idx_95        = np.argmin(np.abs(fpr_arr - 0.05))
thresh_95bkg  = roc_thresholds[idx_95]
sig_eff_at_95 = tpr_arr[idx_95]

idx_99        = np.argmin(np.abs(fpr_arr - 0.01))
thresh_99bkg  = roc_thresholds[idx_99]
sig_eff_at_99 = tpr_arr[idx_99]

print(f"\nROC AUC             : {roc_auc:.4f}")
print(f"Average Precision   : {ap:.4f}")
print(f"Próg max-F1         : {best_f1_thresh:.3f}  (F1={f1_arr[best_f1_idx]:.3f})")
print(f"Próg 95% rej. tła   : {thresh_95bkg:.3f}  (sig. eff.={sig_eff_at_95:.3f})")
print(f"Próg 99% rej. tła   : {thresh_99bkg:.3f}  (sig. eff.={sig_eff_at_99:.3f})")

evals          = xgb.evals_result()
train_auc_hist = evals['validation_0']['auc']
val_auc_hist   = evals['validation_1']['auc']

# =============================================================================
# 7. PRZYGOTOWANIE DANYCH DO MACIERZY KORELACJI
#    Obliczamy korelację Spearmana (odporna na outliery, lepsza dla danych
#    fizycznych z długimi ogonami) + korelację z etykietą y.
# =============================================================================
# Dodaj etykietę jako ostatnią kolumnę do wizualizacji korelacji z y
X_corr = X.copy()

# Korelacja Spearmana – rankingowa, lepsza dla danych z ogonami
corr_matrix = X_corr.corr(method='spearman')

# Korelacja każdej cechy z etykietą (sygnał/tło)
y_float = y.astype(float)
feature_label_corr = X_corr.corrwith(y_float, method='spearman').sort_values(
    key=lambda x: x.abs(), ascending=False
)

# =============================================================================
# 8. GENEROWANIE PDF
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

with PdfPages("Plots/XGB_Output.pdf") as pdf:

# =========================================================================
    # STRONA 1: METRYKI PODSTAWOWE (Poprawione na F1)
    # =========================================================================
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Muon Identification — XGBoost Analysis\n'
                  f'[Train S/B=1:{n_bkg_train/n_sig_train:.0f}  |  '
                  f'Test S/B=1:{n_bkg_test/n_sig_test:.0f} (realistic)]',
                  fontsize=16, y=0.98, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_cm = fig.add_subplot(gs[0, 0])
    
    # Używamy progu F1 do binarnej klasyfikacji
    y_pred_f1 = (y_probs >= best_f1_thresh).astype(int)
    cm_mat = confusion_matrix(y_test, y_pred_f1)
    
    # Normalizacja dla heatmapy (wiersze sumują się do 100%)
    cm_norm = cm_mat.astype(float) / cm_mat.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', ax=ax_cm,
                cmap='Blues', linewidths=0.5,
                xticklabels=['Background', 'Muon'],
                yticklabels=['Background', 'Muon'],
                cbar_kws={'label': 'Fraction'})
    
    # Aktualizacja tytułu na F1
    ax_cm.set_title(f'Confusion Matrix\n@ Max F1-Score (thr={best_f1_thresh:.2f})',
                    color=SIG_COLOR, pad=10)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    
    # Dodanie surowych liczb w nawiasach
    for i in range(2):
        for j in range(2):
            ax_cm.text(j+0.5, i+0.72, f'({cm_mat[i,j]})',
                        ha='center', va='center', fontsize=9)

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


    plots = {
        "01_confusion_matrix": ax_cm,
        "02_roc_curve": ax_roc,
        "03_precision_recall": ax_pr,
        "04_distribution": ax_resp,
        "05_efficiency": ax_eff
    }

    import matplotlib.transforms as mtransforms # Potrzebne do obliczeń
    


    pdf.savefig(fig, bbox_inches='tight')
    
    plt.close()

    # 1. CONFUSION MATRIX
    fig_cm = plt.figure(figsize=(10, 10))
    ax_cm_single = fig_cm.add_subplot(111)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', ax=ax_cm_single,
                cmap='Blues', linewidths=0.5, cbar=False,
                xticklabels=['Background', 'Muon'],
                yticklabels=['Background', 'Muon'],
                annot_kws={"size": 14})
    ax_cm_single.tick_params(axis='both', labelsize=16)
    ax_cm_single.set_title(f'Confusion Matrix\n(threshold={best_f1_thresh:.2f})', fontsize=20)
    ax_cm_single.set_xlabel('Predicted', fontsize=16)
    ax_cm_single.set_ylabel('True', fontsize=16)
    # Dodanie surowych liczb
    for i in range(2):
        for j in range(2):
            ax_cm_single.text(j+0.5, i+0.72, f'({cm_mat[i,j]})', 
                            ha='center', va='center', fontsize=11)

    fig_cm.savefig('Plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig_cm)

    # 2. ROC CURVE
    fig_roc = plt.figure(figsize=(10, 10))
    ax_roc_single = fig_roc.add_subplot(111)
    ax_roc_single.plot(fpr_arr, tpr_arr, color=SIG_COLOR, lw=3, label=f'AUC = {roc_auc:.4f}')
    ax_roc_single.plot([0,1], [0,1], linestyle='--', color='gray', lw=1.5)
    ax_roc_single.set_title('ROC Curve', fontsize=20)
    ax_roc_single.set_xlabel('False Positive Rate', fontsize=16)
    ax_roc_single.set_ylabel('True Positive Rate', fontsize=16)
    ax_roc_single.tick_params(axis='both', labelsize=12)
    ax_roc_single.legend(fontsize=16)
    ax_roc_single.grid(True, alpha=0.3)
    fig_roc.savefig('Plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig_roc)

    # 3. CLASSIFIER RESPONSE
    fig_resp = plt.figure(figsize=(10, 10))
    ax_resp_single = fig_resp.add_subplot(111)
    bins = np.linspace(0, 1, 50)
    ax_resp_single.hist(y_probs[y_test==0], bins=bins, alpha=0.6, density=True,
                        color=BKG_COLOR, label='Background', hatch='//', edgecolor=BKG_COLOR)
    ax_resp_single.hist(y_probs[y_test==1], bins=bins, alpha=0.6, density=True,
                        color=SIG_COLOR, label='Muon/Signal')
    ax_resp_single.axvline(best_f1_thresh, color=ACC_COLOR, linestyle='--', lw=2.5, 
                        label=f'Cut (max-F1): {best_f1_thresh:.2f}')
    ax_resp_single.set_yscale('log')
    ax_resp_single.set_title('Response Distribution', fontsize=20)
    ax_resp_single.set_xlabel('Probability of Muon)', fontsize=16)
    ax_resp_single.set_ylabel('Normalized counts (log)', fontsize=16)
    ax_resp_single.legend(fontsize=16)
    ax_resp_single.tick_params(axis='both', labelsize=12)

    ax_resp_single.grid(True, alpha=0.3)
    fig_resp.savefig('Plots/response.png', dpi=300, bbox_inches='tight')
    plt.close(fig_resp)

    # =========================================================================
    # STRONA 2: HISTORIA TRENINGU + OVERTRAINING CHECK
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('XGBoost Training Diagnostics',
                 fontsize=16, y=1.01, fontweight='bold')

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
    # --- DODATKOWY ZAPIS: FEATURE IMPORTANCE (Estetyczny prostokąt) ---

    # Ustawiamy proporcje 10x12, aby 20 cech miało odpowiednio dużo miejsca w pionie
    fig_feat = plt.figure(figsize=(10, 12))
    ax_feat = fig_feat.add_subplot(111)

    # Rysowanie słupków z nieco większym zaokrągleniem/wyrazistością
    bars = ax_feat.barh(range(top_n), importances[idx_top], color=colors,
                        align='center', edgecolor='#2c3e50', linewidth=0.8)

    # 1. NAZWY CECH - Większe i wyraźne
    ax_feat.set_yticks(range(top_n))
    ax_feat.set_yticklabels([all_features[i] for i in idx_top], fontsize=16)

    # 2. OŚ X - Powiększone cyfry i opis
    ax_feat.tick_params(axis='x', labelsize=12)
    ax_feat.set_xlabel('Importance (Gain)', fontsize=16, labelpad=15, fontweight='bold')

    # 3. TYTUŁ - Bardziej elegancki i odsunięty
    ax_feat.set_title(f'Top {top_n} Feature Importances', 
                    fontsize=20, pad=30, fontweight='bold',x=0.0, ha='left')

    # 4. LEGENDA - Przeniesiona na zewnątrz lub w wolne miejsce, by nie zasłaniać
    ax_feat.legend(handles=legend_elements, fontsize=16, loc='lower right', 
                frameon=True, shadow=True, borderpad=1)

    # 5. DOPRACOWANIE WYGLĄDU
    ax_feat.grid(True, axis='x', linestyle=':', alpha=0.7, color='gray') # Subtelna siatka
    ax_feat.set_axisbelow(True) # Siatka pod słupkami

    # Usuwamy zbędne ramki (górną i prawą), żeby wykres był "lżejszy"
    ax_feat.spines['top'].set_visible(False)
    ax_feat.spines['right'].set_visible(False)

    # Zapis z automatycznym dopasowaniem marginesów
    fig_feat.savefig('Plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig_feat)

    # =========================================================================
    # STRONA 4: ROZKŁADY FIZYCZNE KLUCZOWYCH CECH
    # =========================================================================
    imp_series = pd.Series(importances, index=all_features)
    phys_features = imp_series.nlargest(12).index.tolist()

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle('Key Physics Variable Distributions: Signal vs Background\n'
                 '[rozkłady z całego datasetu przed podziałem]',
                 fontsize=14, y=1.01, fontweight='bold')
    axes = axes.flatten()

    for ax, feat in zip(axes, phys_features):
        sig_vals = X.loc[y==1, feat].dropna()
        bkg_vals = X.loc[y==0, feat].dropna()
        lo   = np.percentile(pd.concat([sig_vals, bkg_vals]), 2)
        hi   = np.percentile(pd.concat([sig_vals, bkg_vals]), 98) # type: ignore
        bins = np.linspace(lo, hi, 50)
        ax.hist(bkg_vals.clip(lo, hi), bins=bins, alpha=0.6, density=True,
                color=BKG_COLOR, label='Background', hatch='//', edgecolor=BKG_COLOR)
        ax.hist(sig_vals.clip(lo, hi), bins=bins, alpha=0.6, density=True,
                color=SIG_COLOR, label='Signal')
        ax.set_title(feat, color=SIG_COLOR, fontsize=20, pad=5)
        ax.set_xlabel('Value', fontsize=16)
        ax.set_ylabel('Norm.',  fontsize=16)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=16)

    for ax in axes[len(phys_features):]:
        ax.set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # STRONA 5: MACIERZ KORELACJI SPEARMANA – pełna
    # =========================================================================
    fig, ax = plt.subplots(figsize=(20, 17))
    fig.suptitle('Spearman Correlation Matrix — All Features\n'
                 '(wartości z całego datasetu)',
                 fontsize=14, fontweight='bold', y=1.005)

    # Maska górnego trójkąta (symetria – wystarczy dolny)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)   # niebieski-biały-czerwony
    sns.heatmap(
        corr_matrix,
        mask=mask,
        ax=ax,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 6},
        linewidths=0.3,
        linecolor='#cccccc',
        square=True,
        cbar_kws={'label': 'Spearman ρ', 'shrink': 0.6},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7)
    ax.set_title('Wyższe |ρ| → silniejsza monotonna zależność między cechami',
                 fontsize=9, color='gray', pad=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # STRONA 6: KORELACJA CECH Z ETYKIETĄ + MACIERZ KORELACJI – CLUSTERED
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    fig.suptitle('Feature–Label Correlations & Clustered Heatmap',
                 fontsize=14, fontweight='bold', y=1.01)

    # --- Lewy panel: korelacja każdej cechy z etykietą ---
    colors_bar = [SIG_COLOR if v > 0 else BKG_COLOR for v in feature_label_corr.values]
    y_pos = range(len(feature_label_corr))
    axes[0].barh(list(y_pos), feature_label_corr.values, color=colors_bar,
                 edgecolor='#333333', linewidth=0.4)
    axes[0].set_yticks(list(y_pos))
    axes[0].set_yticklabels(feature_label_corr.index, fontsize=8)
    axes[0].axvline(0, color='black', lw=0.8)
    axes[0].set_xlabel('Spearman ρ z etykietą (IsMuon)', fontsize=10)
    axes[0].set_title('Korelacja cech z klasą sygnał/tło\n'
                      'Niebieski = pozytywna (→muon), Czerwony = negatywna (→tło)',
                      color=SIG_COLOR, pad=8, fontsize=9)
    axes[0].grid(True, axis='x', alpha=0.4)
    # Adnotuj wartości
    for i, (val, name) in enumerate(zip(feature_label_corr.values,
                                        feature_label_corr.index)):
        axes[0].text(val + (0.005 if val >= 0 else -0.005), i,
                     f'{val:+.3f}', va='center',
                     ha='left' if val >= 0 else 'right',
                     fontsize=6.5, color='#222222')

    # --- Prawy panel: klasterowana macierz korelacji (mniejsza, top 15 cech) ---
    top15_by_label = feature_label_corr.abs().nlargest(15).index.tolist()
    corr_top15 = corr_matrix.loc[top15_by_label, top15_by_label]
    sns.heatmap(
        corr_top15,
        ax=axes[1],
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 8},
        linewidths=0.5,
        linecolor='#cccccc',
        square=True,
        cbar_kws={'label': 'Spearman ρ', 'shrink': 0.7},
    )
    axes[1].set_xticklabels(axes[1].get_xticklabels(),
                             rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=8)
    axes[1].set_title('Top 15 cech (wg |ρ| z etykietą)\n— korelacje wzajemne',
                      color=SIG_COLOR, pad=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =============================================================================
# 6a. OPTYMALIZACJA PUNKTU ODCIĘCIA (PHYSICS-BASED)
# =============================================================================

    # Zakładamy, że chcemy zoptymalizować punkt odcięcia dla konkretnej skali danych 
    # (np. spodziewanej liczby zdarzeń w eksperymencie)
    expected_S = n_sig_test  # Możesz tu wpisać realną oczekiwaną liczbę sygnału
    expected_B = n_bkg_test  # Możesz tu wpisać realną oczekiwaną liczbę tła

    def calculate_significance(threshold, y_true, y_prob, S_norm, B_norm):
        # Wybieramy zdarzenia powyżej progu
        selected = (y_prob >= threshold)
        s = (y_true[selected] == 1).sum()
        b = (y_true[selected] == 0).sum()
        
        # Skalujemy do oczekiwanych wartości (jeśli test nie jest reprezentatywny ilościowo)
        # Jeśli test jest reprezentatywny, mnożniki to 1
        if (s + b) == 0: return 0
        return s / np.sqrt(s + b)

    # Przeszukujemy progi (używamy tych z krzywej ROC dla wydajności)
    significances = []
    for thr in roc_thresholds:
        # Pomijamy progi > 1 lub < 0
        if thr > 1 or thr < 0: 
            significances.append(0)
            continue
        sig = calculate_significance(thr, y_test.values, y_probs, expected_S, expected_B)
        significances.append(sig)

    best_sig_idx = np.argmax(significances)
    best_sig_thresh = roc_thresholds[best_sig_idx]
    max_sig = significances[best_sig_idx]

    print(f"Optymalny próg (max S/√S+B): {best_sig_thresh:.3f}")
    print(f"Maksymalna istotność: {max_sig:.2f}")

    # =========================================================================
    # STRONA 7: RAPORT TEKSTOWY + PEŁNA NUMERYCZNA TABELA WAŻNOŚCI CECH
    # =========================================================================

    # Przygotuj pełną posortowaną tabelę ważności
    feat_imp_df = pd.DataFrame({
        'Feature'           : all_features,
        'XGB_Importance'    : importances,
        'Spearman_vs_Label' : [feature_label_corr.get(f, np.nan) for f in all_features],
    }).sort_values('XGB_Importance', ascending=False).reset_index(drop=True)
    feat_imp_df['Rank']       = feat_imp_df.index + 1
    feat_imp_df['Cumul_Imp']  = feat_imp_df['XGB_Importance'].cumsum()
    feat_imp_df['Group'] = feat_imp_df['Feature'].apply(
        lambda f: 'ECal' if ('Ecal' in f or 'ECal' in f)
        else ('HCal' if ('Hcal' in f or 'HCal' in f)
        else 'Scalar/Combined')
    )

    report_95 = classification_report(y_test, y_pred_f1,
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
    Test     : 10% puli syg. + 100% puli tła  →  S/B = 1:{n_bkg_test/n_sig_test:.1f}
               Sygnał={n_sig_test}   Tło={n_bkg_test}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MODEL: XGBoost
    n_estimators      = 3000  (best iter: {best_iteration})
    learning_rate     = 0.05     max_depth         = 4
    min_child_weight  = 5        subsample         = 0.8
    colsample_bytree  = 0.8      gamma             = 1.0
    reg_alpha         = 0.1      reg_lambda        = 1.0
    scale_pos_weight  = {scale_pos_weight:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ROC AUC             : {roc_auc:.4f}
  Average Precision   : {ap:.4f}

  Cut @ max-F1        : {best_f1_thresh:.3f}   (F1 = {f1_arr[best_f1_idx]:.3f})
  Cut @ 95% bkg rej.  : {thresh_95bkg:.3f}   →  signal eff. = {sig_eff_at_95:.3f}
  Cut @ 99% bkg rej.  : {thresh_99bkg:.3f}   →  signal eff. = {sig_eff_at_99:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Classification report @ 95% bkg rejection threshold:

{report_95}
"""

    fig = plt.figure(figsize=(14, 10))
    plt.text(0.03, 0.97, text_content, transform=fig.transFigure,
             fontsize=9.5, family='monospace', color='#1a1d27',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.85,
                       edgecolor='#444466'))
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # STRONA 8: PEŁNA TABELA NUMERYCZNYCH WAŻNOŚCI CECH
    #   Dwie kolumny: lewa = ranki 1..N/2, prawa = N/2+1..N
    #   Kolory wierszy według grupy (ECal / HCal / Scalar)
    # =========================================================================
    GROUP_COLORS = {
        'ECal'            : '#d6eaff',   # jasnoniebieski
        'HCal'            : '#ffe0e0',   # jasnoczerwony
        'Scalar/Combined' : '#e6ffe6',   # jasnozielony
    }
    HEADER_COLOR = '#2c3e50'
    TEXT_COLOR   = '#111111'

    n_total   = len(feat_imp_df)
    n_half    = (n_total + 1) // 2
    left_df   = feat_imp_df.iloc[:n_half]
    right_df  = feat_imp_df.iloc[n_half:]

    col_labels = ['#', 'Feature', 'XGB Imp.', 'Cumul.', 'ρ(label)', 'Group']
    col_widths = [0.04, 0.22, 0.09, 0.09, 0.09, 0.13]   # suma ≈ 0.66 per panel

    def draw_half_table(ax, df_half, x_start, title):
        """Rysuje połowę tabeli na osi ax, zaczynając od x_start."""
        ax.axis('off')
        y_top    = 0.96
        row_h    = 0.88 / (n_half + 1)   # +1 na nagłówek
        col_x    = [x_start + sum(col_widths[:i]) for i in range(len(col_labels))]

        # Nagłówek
        for cx, cw, cl in zip(col_x, col_widths, col_labels):
            ax.add_patch(plt.Rectangle((cx, y_top - row_h),
                                       cw - 0.002, row_h * 0.95,
                                       transform=ax.transAxes,
                                       color=HEADER_COLOR, zorder=2))
            ax.text(cx + cw/2, y_top - row_h/2, cl,
                    transform=ax.transAxes,
                    ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color='white', zorder=3)

        # Wiersze danych
        for row_i, (_, row) in enumerate(df_half.iterrows()):
            y_row   = y_top - row_h * (row_i + 2)
            bg_col  = GROUP_COLORS.get(row['Group'], '#ffffff')
            # Tło wiersza
            ax.add_patch(plt.Rectangle((x_start, y_row),
                                       sum(col_widths) - 0.002, row_h * 0.95,
                                       transform=ax.transAxes,
                                       color=bg_col, zorder=1, alpha=0.6))
            # Dane
            values = [
                str(int(row['Rank'])),
                row['Feature'],
                f"{row['XGB_Importance']:.5f}",
                f"{row['Cumul_Imp']:.3f}",
                f"{row['Spearman_vs_Label']:+.3f}",
                row['Group'],
            ]
            aligns = ['center', 'left', 'right', 'right', 'right', 'center']
            offsets= [cw/2, 0.005, cw-0.004, cw-0.004, cw-0.004, cw/2]
            for cx, cw, val, ha, off in zip(col_x, col_widths, values, aligns, offsets):
                ax.text(cx + off, y_row + row_h * 0.45, val,
                        transform=ax.transAxes,
                        ha=ha, va='center',
                        fontsize=6.8, color=TEXT_COLOR, zorder=4)

        ax.set_title(title, fontsize=9, color=SIG_COLOR, pad=4)

    fig, axes = plt.subplots(1, 2, figsize=(20, max(12, n_half * 0.38 + 2)))
    fig.suptitle(
        'Pełna numeryczna tabela ważności cech (XGBoost gain)\n'
        f'Wszystkie {n_total} cech posortowane malejąco  |  '
        f'ρ(label) = korelacja Spearmana z etykietą (IsMuon)',
        fontsize=12, fontweight='bold', y=1.01
    )

    draw_half_table(axes[0], left_df,  x_start=0.01,
                    title=f'Ranki 1 – {n_half}')
    draw_half_table(axes[1], right_df, x_start=0.01,
                    title=f'Ranki {n_half+1} – {n_total}')

    # Legenda grup
    legend_patches = [
        Patch(color=GROUP_COLORS['ECal'],            label='ECal features'),
        Patch(color=GROUP_COLORS['HCal'],            label='HCal features'),
        Patch(color=GROUP_COLORS['Scalar/Combined'], label='Scalar / Combined'),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=3, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

plt.rcdefaults()
print("\nAnaliza zakończona. PDF zapisany jako 'Plots/XGB_Output.pdf'.")
print("  Strony:")
print("    1 = Metryki podstawowe (ROC, PR, confusion matrix, response)")
print("    2 = Historia treningu + overtraining check")
print("    3 = Feature importances (wykres)")
print("    4 = Rozkłady fizyczne")
print("    5 = Macierz korelacji Spearmana – pełna (dolny trójkąt)")
print("    6 = Korelacja cech z etykietą + clustered heatmap top-15")
print("    7 = Raport tekstowy + classification report")
print("    8 = Pełna numeryczna tabela ważności cech")
