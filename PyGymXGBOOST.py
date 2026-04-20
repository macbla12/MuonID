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
import shap

from sklearn.base import BaseEstimator, TransformerMixin
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common._apply_operation import (
    apply_sub, apply_div, apply_mul, apply_add, apply_sqrt
)

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

os.makedirs("Plots", exist_ok=True)



# =============================================================================
# 1. WCZYTYWANIE DANYCH (WSZYSTKIE PLIKI JUŻ ZŁĄCZONE W MLDataGrape.root)
# =============================================================================
file_path = "MLData.root"

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
              'lambda1', 'lambda2', 'lambda3']

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

def engineer_shape_features(df_ecal, df_hcal):
    s = pd.DataFrame(index=df_ecal.index)

    λ1e = df_ecal['EcalShape_lambda1']
    λ2e = df_ecal['EcalShape_lambda2']
    λ3e = df_ecal['EcalShape_lambda3']

    λ1h = df_hcal['HcalShape_lambda1']
    λ2h = df_hcal['HcalShape_lambda2']
    λ3h = df_hcal['HcalShape_lambda3']

    s['Ecal_trans'] = np.sqrt(np.clip(λ1e * λ2e, 0, None))
    s['Hcal_trans'] = np.sqrt(np.clip(λ1h * λ2h, 0, None))

    s['Ecal_long'] = λ3e
    s['Hcal_long'] = λ3h

    s['Ecal_LoverT'] = safe_divide(λ3e, s['Ecal_trans'])
    s['Hcal_LoverT'] = safe_divide(λ3h, s['Hcal_trans'])

    s['Ecal_sphericity'] = safe_divide(λ1e, λ3e)
    s['Hcal_sphericity'] = safe_divide(λ1h, λ3h)

    s['Ecal_angular_asym'] = safe_divide(
        df_ecal['EcalShape_theta_width'] - df_ecal['EcalShape_phi_width'],
        df_ecal['EcalShape_theta_width'] + df_ecal['EcalShape_phi_width']
    )
    s['Hcal_angular_asym'] = safe_divide(
        df_hcal['HcalShape_theta_width'] - df_hcal['HcalShape_phi_width'],
        df_hcal['HcalShape_theta_width'] + df_hcal['HcalShape_phi_width']
    )

    s['radius_ratio'] = safe_divide(df_ecal['EcalShape_radius'],
                                    df_hcal['HcalShape_radius'])
    s['disp_ratio'] = safe_divide(df_ecal['EcalShape_dispersion'],
                                  df_hcal['HcalShape_dispersion'])
    s['trans_ratio'] = safe_divide(s['Ecal_trans'], s['Hcal_trans'])
    s['long_ratio']  = safe_divide(s['Ecal_long'],  s['Hcal_long'])

    s['LoverT_mismatch'] = abs(s['Ecal_LoverT'] - s['Hcal_LoverT'])
    s['sphericity_mismatch'] = abs(s['Ecal_sphericity'] - s['Hcal_sphericity'])

    s['Radial_HCal_Fraction'] = safe_divide(
        df_hcal['HcalShape_radius'],
        df_ecal['EcalShape_radius'] + df_hcal['HcalShape_radius']
    )

    s = pd.concat([s, df_ecal, df_hcal], axis=1)
    return s

#X_scalar = engineer_scalar_features(df)
X = engineer_shape_features(df_ecal, df_hcal)
#X        = pd.concat([X_scalar, X_shape], axis=1)
y        = (df['IsMuon'] == 1).astype(int)
file_idx = df['FileIndex'].astype(int)

print(f"\nŁączna liczba cech: {X.shape[1]}")
#print(f"  Skalarne + pochodne : {X_scalar.shape[1]}")
print(f"  Shape pochodne      : {X.shape[1]}")
all_features = X.columns.tolist()

# =============================================================================
# 4. PODZIAŁ DANYCH — STRATYFIKACJA PO KLASIE I FileIndex
# =============================================================================
rng = np.random.RandomState(42)

train_idx_list = []
test_idx_list  = []

for fidx in sorted(file_idx.unique()):
    mask_f = (file_idx == fidx)
    idx_f  = np.where(mask_f)[0]
    y_f    = y.iloc[idx_f].values

    idx_sig_f = idx_f[y_f == 1]
    idx_bkg_f = idx_f[y_f == 0]

    n_sig_test_f = max(1, int(len(idx_sig_f) * 0.20))
    n_bkg_test_f = max(1, int(len(idx_bkg_f) * 0.20))

    idx_sig_test_f = rng.choice(idx_sig_f, size=n_sig_test_f, replace=False) if len(idx_sig_f) > 0 else np.array([], dtype=int)
    idx_bkg_test_f = rng.choice(idx_bkg_f, size=n_bkg_test_f, replace=False) if len(idx_bkg_f) > 0 else np.array([], dtype=int)

    idx_test_f  = np.concatenate([idx_sig_test_f, idx_bkg_test_f])
    idx_train_f = np.setdiff1d(idx_f, idx_test_f)

    train_idx_list.append(idx_train_f)
    test_idx_list.append(idx_test_f)

idx_train = np.concatenate(train_idx_list)
idx_test  = np.concatenate(test_idx_list)

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
# 7. KORELACJE (Spearman) + korelacja z etykietą
# =============================================================================
X_corr = X.copy()
corr_matrix = X_corr.corr(method='spearman')
y_float = y.astype(float)
feature_label_corr = X_corr.corrwith(y_float, method='spearman').sort_values(
    key=lambda x: x.abs(), ascending=False
)

# =============================================================================
# 8. PDF — BEZ DUPLIKATÓW PNG, Z DODANYM SHAP
# =============================================================================
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'font.family':      'sans-serif',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

SIG_COLOR = '#1f77b4'
BKG_COLOR = "#ff0e0e"
ACC_COLOR = '#2ca02c'
PUR_COLOR = '#9467bd'

with PdfPages("Plots/XGB_Output.pdf") as pdf:

    # STRONA 1: METRYKI PODSTAWOWE
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Muon Identification — XGBoost Analysis\n'
                  f'[Train S/B=1:{n_bkg_train/n_sig_train:.0f}  |  '
                  f'Test S/B=1:{n_bkg_test/n_sig_test:.0f} (realistic, all files)]',
                  fontsize=16, y=0.98, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_cm = fig.add_subplot(gs[0, 0])
    y_pred_f1 = (y_probs >= best_f1_thresh).astype(int)
    cm_mat = confusion_matrix(y_test, y_pred_f1)
    cm_norm = cm_mat.astype(float) / cm_mat.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', ax=ax_cm,
                cmap='Blues', linewidths=0.5,
                xticklabels=['Background', 'Muon'],
                yticklabels=['Background', 'Muon'],
                cbar_kws={'label': 'Fraction'})
    ax_cm.set_title(f'Confusion Matrix\n@ Max F1-Score (thr={best_f1_thresh:.2f})',
                    color=SIG_COLOR, pad=10)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
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

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # STRONA 2: TRAINING DIAGNOSTICS
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

    # STRONA 3: FEATURE IMPORTANCE
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

    # STRONA 4: ROZKŁADY KLUCZOWYCH CECH
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
        ax.set_title(feat, color=SIG_COLOR, fontsize=12, pad=5)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Norm.',  fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)

    for ax in axes[len(phys_features):]:
        ax.set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # STRONA 5: MACIERZ KORELACJI SPEARMANA
    fig, ax = plt.subplots(figsize=(20, 17))
    fig.suptitle('Spearman Correlation Matrix — All Features\n'
                 '(wartości z całego datasetu)',
                 fontsize=14, fontweight='bold', y=1.005)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        ax=ax,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=False,
        linewidths=0.3,
        linecolor='#cccccc',
        square=True,
        cbar_kws={'label': 'Spearman ρ', 'shrink': 0.6},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6)
    ax.set_title('Wyższe |ρ| → silniejsza monotonna zależność między cechami',
                 fontsize=9, color='gray', pad=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # STRONA 6: KORELACJA CECH Z ETYKIETĄ
    fig, ax = plt.subplots(figsize=(12, 16))
    colors_bar = [SIG_COLOR if v > 0 else BKG_COLOR for v in feature_label_corr.values]
    y_pos = range(len(feature_label_corr))
    ax.barh(list(y_pos), feature_label_corr.values, color=colors_bar,
            edgecolor='#333333', linewidth=0.4)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feature_label_corr.index, fontsize=7)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Spearman ρ z etykietą (IsMuon)', fontsize=10)
    ax.set_title('Korelacja cech z klasą sygnał/tło\n'
                 'Niebieski = pozytywna (→muon), Czerwony = negatywna (→tło)',
                 color=SIG_COLOR, pad=8, fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # STRONA 7: SHAP — WYJAŚNIENIE MODELU
    explainer = shap.TreeExplainer(xgb)
    # Bierzemy próbkę, żeby nie zabić pamięci
    sample_idx = np.random.choice(len(X_test_sc), size=min(5000, len(X_test_sc)), replace=False)
    X_shap = X_test_sc[sample_idx]
    shap_values = explainer.shap_values(X_shap)

    # Summary plot
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=all_features, show=False)
    plt.title('SHAP Summary Plot — Feature Impact on P(Muon)', fontsize=14)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Dependence plot dla najważniejszej cechy
    top_feat = imp_series.nlargest(1).index[0]
    fig = plt.figure(figsize=(8, 6))
    shap.dependence_plot(top_feat, shap_values, X_shap,
                         feature_names=all_features, show=False)
    plt.title(f'SHAP Dependence: {top_feat}', fontsize=14)
    
    # STRONY 8+: SHAP PER FILEINDEX — DIAGNOSTYKA DOMAIN SHIFT
    
    unique_files = sorted(file_idx.unique())

    for fidx in unique_files:
        mask_f = (file_idx.iloc[idx_test] == fidx)
        if mask_f.sum() < 50:
            continue  # za mało zdarzeń, pomijamy

        X_f = X_test_sc[mask_f]
        y_f = y_test[mask_f]

        # Bierzemy próbkę, żeby nie zabić pamięci
        sample_idx_f = np.random.choice(len(X_f), size=min(3000, len(X_f)), replace=False)
        X_shap_f = X_f[sample_idx_f]

        shap_values_f = explainer.shap_values(X_shap_f)

        # --- STRONA: SHAP SUMMARY ---
        fig = plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_f, X_shap_f, feature_names=all_features, show=False)
        plt.title(f'SHAP Summary — FileIndex={fidx}', fontsize=14)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # --- STRONA: SHAP DEPENDENCE (najważniejsza cecha) ---
        top_feat = imp_series.nlargest(1).index[0]
        fig = plt.figure(figsize=(8, 6))
        shap.dependence_plot(top_feat, shap_values_f, X_shap_f,
                             feature_names=all_features, show=False)
        plt.title(f'SHAP Dependence: {top_feat}  (FileIndex={fidx})', fontsize=14)

    # =========================================================================
    # STRONA: SHAP SIMILARITY MATRIX (FileIndex vs FileIndex)
    # =========================================================================

    # 1. Liczymy SHAP importance per FileIndex
    shap_importances = {}

    for fidx in unique_files:
        mask_f = (file_idx.iloc[idx_test] == fidx)
        if mask_f.sum() < 50:
            continue

        X_f = X_test_sc[mask_f]
        sample_idx_f = np.random.choice(len(X_f), size=min(3000, len(X_f)), replace=False)
        X_shap_f = X_f[sample_idx_f]

        shap_values_f = explainer.shap_values(X_shap_f)

        # średnia absolutna wartość SHAP dla każdej cechy
        shap_importances[fidx] = np.mean(np.abs(shap_values_f), axis=0)

    # 2. Tworzymy macierz (n_files × n_features)
    file_ids = sorted(shap_importances.keys())
    shap_matrix = np.vstack([shap_importances[f] for f in file_ids])

    # 3. Korelacja między FileIndex
    similarity = np.corrcoef(shap_matrix)

    # 4. Rysujemy heatmapę
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity, annot=True, fmt=".2f",
                xticklabels=file_ids, yticklabels=file_ids,
                cmap="coolwarm", vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation'})
    ax.set_title("SHAP Similarity Matrix — FileIndex vs FileIndex\n"
                 "(korelacja między wektorami ważności SHAP)", fontsize=14)
    ax.set_xlabel("FileIndex")
    ax.set_ylabel("FileIndex")

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


    plt.close()

print("Gotowe: trening na wszystkich plikach, podział per FileIndex, PDF + SHAP.")
