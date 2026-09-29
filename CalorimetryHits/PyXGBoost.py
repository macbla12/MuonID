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

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

import torch
import torch.nn as nn
import onnx

os.makedirs("Plots", exist_ok=True)
os.makedirs("ONNX", exist_ok=True)


# =============================================================================
# 0. Raw branch order. This list is the single source of truth for how the
#    per-track features are laid out. Both the pandas feature-engineering
#    functions and the Torch preprocessor read RAW_COLS in this exact order,
#    so the ONNX pipeline and the pandas pipeline stay in sync.
#
#    IMPORTANT: this must be a single flat array of floats, in this order,
#    if you ever build the input by hand in C++ / ONNX Runtime.
# =============================================================================
RAW_COLS = [
    'ECalEnergy', 'HCalEnergy', 'ECalNumber', 'HCalNumber', 'ECalEoverP', 'HCalEoverP',
    'ECalAvgHitEnergy', 'HCalAvgHitEnergy',
    'ECalSpreadPhi', 'ECalSpreadEta', 'ECalSpreadR',
    'HCalSpreadPhi', 'HCalSpreadEta', 'HCalSpreadR',
    'ECalMaxHitFrac', 'HCalMaxHitFrac',
    'ECalEnergyStdDev', 'HCalEnergyStdDev',
    'ECalEnergyConcentration', 'HCalEnergyConcentration',
    'ECalR_Disp', 'ECalR_DispWeighted', 'ECalEta_DispWeighted', 'ECalPhi_DispWeighted',
    'HCalR_Disp', 'HCalR_DispWeighted', 'HCalEta_DispWeighted', 'HCalPhi_DispWeighted',
    'TrackMomentum', 'TrackEta',
]

with open("ONNX/raw_feature_order.txt", "w") as f:
    f.write(",".join(RAW_COLS) + "\n")


# =============================================================================
# 1. Load the combined training data
# =============================================================================
file_path = "ONNX/MLDataHits.root"

with uproot.open(file_path) as f:
    df = f["MLDataTree"].arrays(library="pd")

n_sig_total = (df['IsMuon'] == 1).sum()
n_bkg_total = (df['IsMuon'] == 0).sum()
print(f"Loaded {len(df)} events.")
print(f"  Signal (muons) : {n_sig_total}")
print(f"  Background     : {n_bkg_total}")
print(f"  S/B ratio       : 1 : {n_bkg_total/n_sig_total:.1f}")

is_mu    = df["IsMuon"].values

# =============================================================================
# 2. Build engineered features
# =============================================================================

def safe_divide(num, denom):
    num = np.asarray(num, dtype=float)
    denom = np.asarray(denom, dtype=float)
    denom_nonzero = denom != 0
    result = np.zeros_like(num, dtype=float)
    result[denom_nonzero] = num[denom_nonzero] / denom[denom_nonzero]
    return result


def engineer_scalar_features(df):
    # Raw pass-through (all per-track hit-based features straight from the tree)
    s = df[RAW_COLS].copy()

    totalE = df['ECalEnergy'] + df['HCalEnergy']
    s['ECalFrac']     = safe_divide(df['ECalEnergy'].values, totalE.values)
    s['HCalFrac']     = safe_divide(df['HCalEnergy'].values, totalE.values)
    s['HitRatio']     = safe_divide(df['ECalNumber'].values, df['HCalNumber'].values)
    s['EoverP_ratio'] = safe_divide(df['ECalEoverP'].values, df['HCalEoverP'].values)
    s['logECal']      = np.log1p(df['ECalEnergy'])
    s['logHCal']      = np.log1p(df['HCalEnergy'])

    return s


def engineer_shape_features(df):
    # Purely derived features built from the hit-level dispersion/spread
    # variables. No raw pass-through here (already in engineer_scalar_features)
    # to avoid duplicate columns.
    s = pd.DataFrame(index=df.index)

    ECalEta_w = df['ECalEta_DispWeighted']
    ECalPhi_w = df['ECalPhi_DispWeighted']
    HCalEta_w = df['HCalEta_DispWeighted']
    HCalPhi_w = df['HCalPhi_DispWeighted']

    ECalR_uw = df['ECalR_Disp']          # analog of the old "radius" (unweighted)
    ECalR_w  = df['ECalR_DispWeighted']  # analog of the old "dispersion" (energy-weighted)
    HCalR_uw = df['HCalR_Disp']
    HCalR_w  = df['HCalR_DispWeighted']

    # "Transverse" width = combined eta-phi energy-weighted spread
    s['Ecal_trans'] = np.sqrt(np.clip(ECalEta_w * ECalPhi_w, 0, None))
    s['Hcal_trans'] = np.sqrt(np.clip(HCalEta_w * HCalPhi_w, 0, None))

    # "Longitudinal" extent = radial (energy-weighted) dispersion
    s['Ecal_long'] = ECalR_w
    s['Hcal_long'] = HCalR_w

    s['Ecal_LoverT'] = safe_divide(s['Ecal_long'].values, s['Ecal_trans'].values)
    s['Hcal_LoverT'] = safe_divide(s['Hcal_long'].values, s['Hcal_trans'].values)

    # "Sphericity" analog = ratio of the two angular widths (eta vs phi)
    s['Ecal_sphericity'] = safe_divide(ECalEta_w.values, ECalPhi_w.values)
    s['Hcal_sphericity'] = safe_divide(HCalEta_w.values, HCalPhi_w.values)

    s['Ecal_angular_asym'] = safe_divide((ECalEta_w - ECalPhi_w).values, (ECalEta_w + ECalPhi_w).values)
    s['Hcal_angular_asym'] = safe_divide((HCalEta_w - HCalPhi_w).values, (HCalEta_w + HCalPhi_w).values)

    s['radius_ratio'] = safe_divide(ECalR_uw.values, HCalR_uw.values)
    s['disp_ratio']   = safe_divide(ECalR_w.values, HCalR_w.values)
    s['trans_ratio']  = safe_divide(s['Ecal_trans'].values, s['Hcal_trans'].values)
    s['long_ratio']   = safe_divide(s['Ecal_long'].values, s['Hcal_long'].values)

    s['LoverT_mismatch']     = (s['Ecal_LoverT'] - s['Hcal_LoverT']).abs()
    s['sphericity_mismatch'] = (s['Ecal_sphericity'] - s['Hcal_sphericity']).abs()

    s['Radial_HCal_Fraction'] = safe_divide(HCalR_w.values, (ECalR_w + HCalR_w).values)

    # Extra mismatch features enabled by the new hit-level information
    # (had no equivalent in the old cluster-shape pipeline)
    s['EnergyConcentration_mismatch'] = (df['ECalEnergyConcentration'] - df['HCalEnergyConcentration']).abs()
    s['MaxHitFrac_mismatch']          = (df['ECalMaxHitFrac'] - df['HCalMaxHitFrac']).abs()
    s['EnergyStdDev_ratio']           = safe_divide(df['ECalEnergyStdDev'].values, df['HCalEnergyStdDev'].values)

    return s


X_scalar = engineer_scalar_features(df)
X_shape  = engineer_shape_features(df)
X        = pd.concat([X_scalar, X_shape], axis=1)
y        = (df['IsMuon'] == 1).astype(int)
file_idx = df['FileIndex'].astype(int)

print(f"\nTotal number of features: {X.shape[1]}")
all_features = X.columns.tolist()

with open("ONNX/feature_order.txt", "w") as f:
    f.write(",".join(all_features) + "\n")

# =============================================================================
# 3. Split data stratified by class and file index
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
print(f"  TRAIN : Signal={n_sig_train:>5}  Background={n_bkg_train:>5}  "
      f"S/B = 1:{n_bkg_train/n_sig_train:.1f}")
print(f"  TEST  : Signal={n_sig_test:>5}  Background={n_bkg_test:>5}  "
      f"S/B = 1:{n_bkg_test/n_sig_test:.1f}  ")
print(f"{'='*55}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
means  = scaler.mean_
scales = scaler.scale_

with open("ONNX/scalars.txt", "w") as f:
    f.write(",".join([f"{x:.8f}" for x in means]) + "\n")
    f.write(",".join([f"{x:.8f}" for x in scales]) + "\n")

# =============================================================================
# 4. Train the XGBoost classifier
#
#    Single, explicit knob for the sig/bkg balance: fp_penalty.
#    (scale_pos_weight is intentionally NOT used here, to avoid stacking two
#    mechanisms that both reweight background vs signal.)
# =============================================================================
xgb = XGBClassifier(
    n_estimators          = 2000,
    learning_rate         = 0.05,
    max_depth             = 3,
    min_child_weight      = 10,
    subsample             = 0.6,
    colsample_bytree      = 0.6,
    gamma                 = 2.0,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    eval_metric           = 'auc',
    early_stopping_rounds = 30,
    tree_method           = 'hist',
    random_state          = 42,
    n_jobs                = -1,
    verbosity             = 1,
)

# Single source of sig/bkg reweighting: penalize false positives (background
# mis-tagged as muon) by this factor. This is the ONLY scaling knob used.
fp_penalty = 1
sample_weights = np.where(y_train == 0, fp_penalty, 1.0)

xgb.fit(
    X_train_sc, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_train_sc, y_train), (X_test_sc, y_test)],
    verbose=50,
)

y_probs = xgb.predict_proba(X_test_sc)[:, 1]

best_iteration = xgb.best_iteration
print(f"\nBest iteration (early stopping): {best_iteration}")

n_features = X_train_sc.shape[1]
onnx_model = convert_xgboost(
    xgb,
    initial_types=[('input', OnnxFloat([None, n_features]))]
)
onnx_path = "ONNX/xgb_muonID_raw.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"ONNX model saved: {onnx_path}")


# =============================================================================
# A. Torch preprocessor matching the feature engineering pipeline
#
#    Single input "raw_features": shape (N, len(RAW_COLS)), columns in the
#    exact order of RAW_COLS. This is the same order the C++ side should
#    fill a flat float array with (see ONNX/raw_feature_order.txt).
# =============================================================================
N_RAW = len(RAW_COLS)


class MuonIDPreprocessor(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(np.asarray(scale), dtype=torch.float32))

    @staticmethod
    def safe_divide(num, denom):
        is_zero = denom == 0
        denom_safe = torch.where(is_zero, torch.ones_like(denom), denom)
        result = num / denom_safe
        return torch.where(is_zero, torch.zeros_like(result), result)

    @staticmethod
    def col(raw, name):
        i = RAW_COLS.index(name)
        return raw[:, i:i + 1]

    def forward(self, raw_features):
        sd = self.safe_divide
        col = self.col
        raw = raw_features

        ECalEnergy   = col(raw, 'ECalEnergy')
        HCalEnergy   = col(raw, 'HCalEnergy')
        ECalNumber   = col(raw, 'ECalNumber')
        HCalNumber   = col(raw, 'HCalNumber')
        ECalEoverP   = col(raw, 'ECalEoverP')
        HCalEoverP   = col(raw, 'HCalEoverP')

        ECalEta_w = col(raw, 'ECalEta_DispWeighted')
        ECalPhi_w = col(raw, 'ECalPhi_DispWeighted')
        HCalEta_w = col(raw, 'HCalEta_DispWeighted')
        HCalPhi_w = col(raw, 'HCalPhi_DispWeighted')

        ECalR_uw = col(raw, 'ECalR_Disp')
        ECalR_w  = col(raw, 'ECalR_DispWeighted')
        HCalR_uw = col(raw, 'HCalR_Disp')
        HCalR_w  = col(raw, 'HCalR_DispWeighted')

        ECalEnergyConcentration = col(raw, 'ECalEnergyConcentration')
        HCalEnergyConcentration = col(raw, 'HCalEnergyConcentration')
        ECalMaxHitFrac          = col(raw, 'ECalMaxHitFrac')
        HCalMaxHitFrac          = col(raw, 'HCalMaxHitFrac')
        ECalEnergyStdDev        = col(raw, 'ECalEnergyStdDev')
        HCalEnergyStdDev        = col(raw, 'HCalEnergyStdDev')

        # --- engineer_scalar_features: raw pass-through + derived ---
        totalE = ECalEnergy + HCalEnergy
        ECalFrac     = sd(ECalEnergy, totalE)
        HCalFrac     = sd(HCalEnergy, totalE)
        HitRatio     = sd(ECalNumber, HCalNumber)
        EoverP_ratio = sd(ECalEoverP, HCalEoverP)
        logECal      = torch.log1p(ECalEnergy)
        logHCal      = torch.log1p(HCalEnergy)

        scalar_features = torch.cat([
            raw,  # raw pass-through, same order as RAW_COLS
            ECalFrac, HCalFrac, HitRatio, EoverP_ratio, logECal, logHCal,
        ], dim=1)

        # --- engineer_shape_features ---
        Ecal_trans = torch.sqrt(torch.clamp(ECalEta_w * ECalPhi_w, min=0.0))
        Hcal_trans = torch.sqrt(torch.clamp(HCalEta_w * HCalPhi_w, min=0.0))

        Ecal_long, Hcal_long = ECalR_w, HCalR_w

        Ecal_LoverT = sd(Ecal_long, Ecal_trans)
        Hcal_LoverT = sd(Hcal_long, Hcal_trans)

        Ecal_sphericity = sd(ECalEta_w, ECalPhi_w)
        Hcal_sphericity = sd(HCalEta_w, HCalPhi_w)

        Ecal_angular_asym = sd(ECalEta_w - ECalPhi_w, ECalEta_w + ECalPhi_w)
        Hcal_angular_asym = sd(HCalEta_w - HCalPhi_w, HCalEta_w + HCalPhi_w)

        radius_ratio = sd(ECalR_uw, HCalR_uw)
        disp_ratio   = sd(ECalR_w, HCalR_w)
        trans_ratio  = sd(Ecal_trans, Hcal_trans)
        long_ratio   = sd(Ecal_long, Hcal_long)

        LoverT_mismatch     = torch.abs(Ecal_LoverT - Hcal_LoverT)
        sphericity_mismatch = torch.abs(Ecal_sphericity - Hcal_sphericity)

        Radial_HCal_Fraction = sd(HCalR_w, ECalR_w + HCalR_w)

        EnergyConcentration_mismatch = torch.abs(ECalEnergyConcentration - HCalEnergyConcentration)
        MaxHitFrac_mismatch          = torch.abs(ECalMaxHitFrac - HCalMaxHitFrac)
        EnergyStdDev_ratio           = sd(ECalEnergyStdDev, HCalEnergyStdDev)

        shape_features = torch.cat([
            Ecal_trans, Hcal_trans, Ecal_long, Hcal_long,
            Ecal_LoverT, Hcal_LoverT, Ecal_sphericity, Hcal_sphericity,
            Ecal_angular_asym, Hcal_angular_asym,
            radius_ratio, disp_ratio, trans_ratio, long_ratio,
            LoverT_mismatch, sphericity_mismatch, Radial_HCal_Fraction,
            EnergyConcentration_mismatch, MaxHitFrac_mismatch, EnergyStdDev_ratio,
        ], dim=1)

        X = torch.cat([scalar_features, shape_features], dim=1)

        return (X - self.mean) / self.scale


def export_preprocessor(mean, scale, out_path, opset_version=18):
    model = MuonIDPreprocessor(mean, scale)
    model.eval()

    dummy = (torch.rand(2, N_RAW),)
    dyn_axes = {"raw_features": {0: "N"}, "features_scaled": {0: "N"}}

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["raw_features"],
        output_names=["features_scaled"],
        dynamic_axes=dyn_axes,
        opset_version=opset_version,
    )
    return out_path


# =============================================================================
# B. Merge preprocessing and XGBoost models into one ONNX pipeline
# =============================================================================
def merge_with_xgboost(preprocessing_path, xgboost_path, output_path):
    model_pre = onnx.load(preprocessing_path)
    model_xgb = onnx.load(xgboost_path)

    if model_pre.ir_version != model_xgb.ir_version:
        target_ir = min(model_pre.ir_version, model_xgb.ir_version)
        model_pre.ir_version = target_ir
        model_xgb.ir_version = target_ir

    xgb_input_name = model_xgb.graph.input[0].name
    pre_output_name = model_pre.graph.output[0].name

    merged = onnx.compose.merge_models(
        model_pre, model_xgb,
        io_map=[(pre_output_name, xgb_input_name)],
    )
    onnx.checker.check_model(merged)
    onnx.save(merged, output_path)
    return merged


# =============================================================================
# C. Export and test the full ONNX pipeline
# =============================================================================
preprocessing_path = export_preprocessor(
    means, scales, out_path="ONNX/preprocessing.onnx"
)

merged_model = merge_with_xgboost(
    preprocessing_path=preprocessing_path,
    xgboost_path="ONNX/xgb_muonID_raw.onnx",
    output_path="ONNX/xgb_muonID.onnx",
)

print("Done: ONNX/xgb_muonID.onnx")
print("Inputs for the C++ ONNX Runtime pipeline:")
for inp in merged_model.graph.input:
    dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {dims}")

# Quick end-to-end check on one test event
import onnxruntime as ort

sess = ort.InferenceSession("ONNX/xgb_muonID.onnx")

row = df.iloc[[0]]
raw_row = row[RAW_COLS].values.astype(np.float32)
onnx_out = sess.run(None, {"raw_features": raw_row})[0]

row_features = X.iloc[[0]]
row_scaled = scaler.transform(row_features)
pandas_out = xgb.predict_proba(row_scaled)[:, 1]

print("\nComparison for one event:")
print("  full ONNX pipeline :", onnx_out.ravel())
print("  original pandas    :", pandas_out)

# =============================================================================
# 5. Metrics and decision thresholds
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
print(f"Best F1 threshold             : {best_f1_thresh:.3f}  (F1={f1_arr[best_f1_idx]:.3f})")
print(f"Threshold at 95% bkg rejection: {thresh_95bkg:.3f}  (sig. eff.={sig_eff_at_95:.3f})")
print(f"Threshold at 99% bkg rejection: {thresh_99bkg:.3f}  (sig. eff.={sig_eff_at_99:.3f})")

evals          = xgb.evals_result()
train_auc_hist = evals['validation_0']['auc']
val_auc_hist   = evals['validation_1']['auc']

# =============================================================================
# 6. Spearman correlations and correlation with the label
# =============================================================================
X_corr = X.copy()
corr_matrix = X_corr.corr(method='spearman')
y_float = y.astype(float)
feature_label_corr = X_corr.corrwith(y_float, method='spearman').sort_values(
    key=lambda x: x.abs(), ascending=False
)

# =============================================================================
# 7. Create the PDF report
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

    # Page 1: basic metrics
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Muon Identification — XGBoost Analysis (hit-based features)\n'
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
                annot_kws={"size": 16},
                xticklabels=['Background', 'Muon'],
                yticklabels=['Background', 'Muon'],
                cbar_kws={'label': 'Fraction'})

    ax_cm.set_title(f'Confusion Matrix\n@ Max F1-Score (thr={best_f1_thresh:.3f})',
                    color=SIG_COLOR, pad=10, fontsize=16)
    ax_cm.set_xlabel('Predicted', fontsize=16)
    ax_cm.set_ylabel('True', fontsize=16)
    ax_cm.tick_params(axis='both', labelsize=16)

    for i in range(2):
        for j in range(2):
            ax_cm.text(j+0.5, i+0.72, f'({cm_mat[i,j]})',
                    ha='center', va='center', fontsize=16)

    ax_roc = fig.add_subplot(gs[0, 1])
    ax_roc.plot(fpr_arr, tpr_arr, color=SIG_COLOR, lw=2,
                label=f'AUC = {roc_auc:.4f}')
    ax_roc.fill_between(fpr_arr, tpr_arr, alpha=0.1, color=SIG_COLOR)
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=1)
    ax_roc.set_title('ROC Curve', color=SIG_COLOR, pad=10, fontsize=16)
    ax_roc.set_xlabel('False Positive Rate', fontsize=16)
    ax_roc.set_ylabel('True Positive Rate', fontsize=16)
    ax_roc.legend(fontsize=16)
    ax_roc.tick_params(axis='both', labelsize=16)
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
    ax_resp.set_title('Classifier Response Distribution [Test: realistic S/B]',
                    color=SIG_COLOR, pad=10, fontsize=16)
    ax_resp.set_xlabel('P(Muon)', fontsize=16)
    ax_resp.set_ylabel('Normalized counts (log)', fontsize=16)
    ax_resp.tick_params(axis='both', labelsize=16)
    ax_resp.legend(fontsize=16)
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

    # Page 2: training diagnostics
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

    # Page 3: feature importance
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Feature Importances — XGBoost',
                fontsize=20, y=1.02, fontweight='bold')

    importances = xgb.feature_importances_
    indices_all = np.argsort(importances)
    top_n = min(20, len(all_features))
    idx_top = indices_all[-top_n:]

    colors = [SIG_COLOR if 'Ecal' in all_features[i] or 'ECal' in all_features[i]
            else BKG_COLOR if 'Hcal' in all_features[i] or 'HCal' in all_features[i]
            else ACC_COLOR
            for i in idx_top]

    axes[0].barh(range(top_n), importances[idx_top], color=colors,
                align='center', edgecolor='#0f1117', linewidth=0.5)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels([all_features[i] for i in idx_top], fontsize=14)
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].set_title(f'Top {top_n} Features', color=SIG_COLOR, pad=10, fontsize=16)
    axes[0].set_xlabel('Importance (gain)', fontsize=16)
    axes[0].grid(True, axis='x')

    legend_elements = [Patch(facecolor=SIG_COLOR, label='ECal features'),
                    Patch(facecolor=BKG_COLOR, label='HCal features'),
                    Patch(facecolor=ACC_COLOR, label='Combined/scalar')]
    axes[0].legend(handles=legend_elements, fontsize=16, loc='lower right')

    sorted_imp = np.sort(importances)[::-1]
    cum_imp = np.cumsum(sorted_imp)
    n_feats_90 = np.argmax(cum_imp >= 0.90) + 1

    axes[1].plot(range(1, len(sorted_imp)+1), cum_imp, color=SIG_COLOR, lw=2)
    axes[1].axhline(0.90, color=ACC_COLOR, linestyle='--', lw=1.5,
                    label=f'90% importance ({n_feats_90} features)')
    axes[1].axhline(0.95, color=PUR_COLOR, linestyle=':', lw=1.5,
                    label='95% importance')
    axes[1].fill_between(range(1, len(sorted_imp)+1), cum_imp,
                        alpha=0.15, color=SIG_COLOR)
    axes[1].set_title('Cumulative Feature Importance', color=SIG_COLOR, pad=10, fontsize=16)
    axes[1].set_xlabel('Number of features', fontsize=16)
    axes[1].set_ylabel('Cumulative importance', fontsize=16)
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].legend(fontsize=16)
    axes[1].grid(True)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 4: distributions of key features
    imp_series = pd.Series(importances, index=all_features)
    phys_features = imp_series.nlargest(15).index.tolist()

    fig, axes = plt.subplots(3, 5, figsize=(20, 14))
    fig.suptitle('Key Physics Variable Distributions: Signal vs Background\n'
                 '[full dataset before splitting]',
                 fontsize=14, y=1.01, fontweight='bold')
    axes = axes.flatten()

    for ax, feat in zip(axes, phys_features):
        sig_vals = X.loc[y==1, feat].dropna()
        bkg_vals = X.loc[y==0, feat].dropna()
        lo   = np.percentile(pd.concat([sig_vals, bkg_vals]), 2)
        hi   = np.percentile(pd.concat([sig_vals, bkg_vals]), 98) #type: ignore
        bins = np.linspace(lo, hi, 50)
        ax.hist(bkg_vals.clip(lo, hi), bins=bins, alpha=0.6, density=True,
                color=BKG_COLOR, label='Background', hatch='//', edgecolor=BKG_COLOR)
        ax.hist(sig_vals.clip(lo, hi), bins=bins, alpha=0.6, density=True,
                color=SIG_COLOR, label='Signal')
        ax.set_title(feat, color=SIG_COLOR, fontsize=13, pad=5)
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Norm.',  fontsize=12)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=10)

    for ax in axes[len(phys_features):]:
        ax.set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 5: Spearman correlation matrix
    fig, ax = plt.subplots(figsize=(20, 17))
    fig.suptitle('Spearman Correlation Matrix — All Features\n'
                 '(values from the full dataset)',
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
    ax.set_title('Higher |ρ| means a stronger monotonic relationship between features',
                 fontsize=9, color='gray', pad=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 6: feature correlation with the label
    fig, ax = plt.subplots(figsize=(12, 16))
    colors_bar = [SIG_COLOR if v > 0 else BKG_COLOR for v in feature_label_corr.values]
    y_pos = range(len(feature_label_corr))
    ax.barh(list(y_pos), feature_label_corr.values, color=colors_bar,
            edgecolor='#333333', linewidth=0.4)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feature_label_corr.index, fontsize=6)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Spearman ρ with the label (IsMuon)', fontsize=10)
    ax.set_title('Feature correlation with signal/background class\n'
                 'Blue = positive (→ muon), Red = negative (→ background)',
                 color=SIG_COLOR, pad=8, fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =============================================================================
    # INSERT THIS BLOCK INSIDE:  with PdfPages("Plots/XGB_Output.pdf") as pdf:
    # preferably at the very end, just before:  print("Done: training on all files...")
    #
    # Uses variables computed earlier in the script:
    #   corr_matrix, y_probs, y_test, n_sig_test, n_bkg_test,
    #   fpr_arr, tpr_arr, roc_auc, sig_eff_at_95, sig_eff_at_99,
    #   SIG_COLOR, BKG_COLOR, ACC_COLOR, PUR_COLOR
    # =============================================================================

    # ---------------------------------------------------------------------------
    # Page A: FEATURE IMPORTANCE — large, readable version
    # Uses: importances = xgb.feature_importances_, all_features (both computed
    # earlier in the script for page 3 / feature importance).
    # ---------------------------------------------------------------------------
    top_n_big = min(20, len(all_features))
    indices_all_big = np.argsort(importances)
    idx_top_big = indices_all_big[-top_n_big:]

    colors_big = [SIG_COLOR if ('Ecal' in all_features[i] or 'ECal' in all_features[i])
                else BKG_COLOR if ('Hcal' in all_features[i] or 'HCal' in all_features[i])
                else ACC_COLOR
                for i in idx_top_big]

    fig, ax = plt.subplots(figsize=(16, 14))
    bars = ax.barh(range(top_n_big), importances[idx_top_big], color=colors_big,
                    align='center', edgecolor='black', linewidth=0.8)

    ax.set_yticks(range(top_n_big))
    ax.set_yticklabels([all_features[i].upper() for i in idx_top_big],
                        fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlabel('FEATURE IMPORTANCE', fontsize=24, fontweight='bold', labelpad=15)
    ax.set_title(f'FEATURE IMPORTANCE — TOP {top_n_big}', fontsize=30, fontweight='bold', pad=25)
    ax.grid(True, axis='x', alpha=0.4, linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Numeric values at the ends of the bars
    for bar, val in zip(bars, importances[idx_top_big]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=15, fontweight='bold')

    legend_elements = [Patch(facecolor=SIG_COLOR, label='ECAL FEATURES'),
                        Patch(facecolor=BKG_COLOR, label='HCAL FEATURES'),
                        Patch(facecolor=ACC_COLOR, label='COMBINED / SCALAR')]
    ax.legend(handles=legend_elements, fontsize=20, loc='lower right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------------------------
    # Page B: CLASSIFIER SCORE DISTRIBUTION S_mu (MUONS vs PIONS/BACKGROUND)
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 11))
    bins = np.linspace(0, 1, 60)
    ax.hist(y_probs[y_test == 0], bins=bins, density=True, alpha=0.55,
            color=BKG_COLOR, hatch='//', edgecolor=BKG_COLOR, linewidth=1.5,
            label=f'PIONS / BACKGROUND  (N = {n_bkg_test})')
    ax.hist(y_probs[y_test == 1], bins=bins, density=True, alpha=0.65,
            color=SIG_COLOR, edgecolor='black', linewidth=0.8,
            label=f'MUONS  (N = {n_sig_test})')
    ax.set_yscale('log')
    ax.set_xlabel('CLASSIFIER OUTPUT', fontsize=26, fontweight='bold', labelpad=15)
    ax.set_ylabel('NORMALIZED COUNTS ', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_title('CLASSIFIER SCORE DISTRIBUTION  \nMUONS vs PIONS', fontsize=30, fontweight='bold', pad=25)
    ax.tick_params(axis='both', labelsize=22, width=1.5, length=8)
    ax.legend(fontsize=22, loc='upper center', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.4, linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------------------------
    # Page C: ROC CURVE / PERFORMANCE  eps_mu  vs  eps_pi_rej
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 13))
    bkg_rej = 1 - fpr_arr
    ax.plot(bkg_rej, tpr_arr, color=PUR_COLOR, lw=4, label=f'AUC = {roc_auc:.4f}')
    ax.fill_between(bkg_rej, tpr_arr, alpha=0.12, color=PUR_COLOR)
    ax.axvline(0.95, color=ACC_COLOR, linestyle=':', lw=2.5,
            label=f'$\\epsilon_{{\\pi,rej}}$=95%  →  $\\epsilon_{{\\mu}}$={sig_eff_at_95:.3f}')
    ax.axvline(0.99, color=BKG_COLOR, linestyle=':', lw=2.5,
            label=f'$\\epsilon_{{\\pi,rej}}$=99%  →  $\\epsilon_{{\\mu}}$={sig_eff_at_99:.3f}')
    ax.set_xlabel(r'PION REJECTION EFFICIENCY', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel(r'MUON IDENTIFICATION EFFICIENCY', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_title('ROC CURVE:  $\\epsilon_{\\mu}$  vs  $\\epsilon_{\\pi,\\mathrm{rej}}$', fontsize=30, fontweight='bold', pad=25)
    ax.tick_params(axis='both', labelsize=22, width=1.5, length=8)
    ax.legend(fontsize=19, loc='lower left', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.4, linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # =============================================================================
    # Page D: Engineered Calorimeter Comparison Features — po jednej stronie
    # na cechę, ten sam "duży/czytelny" styl co strony A-C powyżej.
    #
    # UWAGA: ta wersja pliku (engineer_scalar_features / engineer_shape_features)
    # nie ma ECalHasHit/HCalHasHit ani AvgHitEnergy_ratio/R_DispWeighted_ratio —
    # to nazwy z innej wersji pipeline'u. Tu używamy odpowiedników, które
    # faktycznie istnieją w X: HitRatio (zamiast HasHit binaries) i disp_ratio
    # (dokładny odpowiednik R_DR,w = ECalR_DispWeighted / HCalR_DispWeighted).
    # =============================================================================
    calo_engineered_features = [
        ('ECalFrac', r'$f_{ECal}$', False),
        ('HCalFrac', r'$f_{HCal}$', False),
        ('EoverP_ratio', r'$R_{E/p}$', False),
        ('logECal', r'$E^{log}_{ECal}$', False),
        ('logHCal', r'$E^{log}_{HCal}$', False),
        ('EnergyStdDev_ratio', r'$R_{\sigma_E}$', False),
        ('EnergyConcentration_mismatch', r'$S_E$', False),
        ('disp_ratio', r'$R_{D_{R,w}}$', False),
        ('MaxHitFrac_mismatch', r'$\Delta_{maxhit} $', False),
        ('HitRatio', r'$R_{hit}$', False),
    ]

    # Zabezpieczenie: gdyby nazwa kolumny nadal się nie zgadzała (np. w kolejnej
    # wersji pipeline'u), pomiń ją zamiast wywalać cały skrypt wyjątkiem.
    missing_feats = [f for f, _, _ in calo_engineered_features if f not in X.columns]
    if missing_feats:
        print(f"  [Page D] WARNING - missing from X, skipping: {missing_feats}")

    for feat_name, feat_label, is_binary in calo_engineered_features:
        if feat_name not in X.columns:
            continue

        sig_vals = X.loc[y == 1, feat_name]
        bkg_vals = X.loc[y == 0, feat_name]

        fig, ax = plt.subplots(figsize=(16, 11))

        if is_binary:
            frac_sig = [np.mean(sig_vals == 0), np.mean(sig_vals == 1)]
            frac_bkg = [np.mean(bkg_vals == 0), np.mean(bkg_vals == 1)]
            x_pos = np.arange(2)
            width = 0.35
            ax.bar(x_pos - width/2, frac_bkg, width, color=BKG_COLOR,
                   edgecolor='black', linewidth=1.2,
                   label=f'Pions')
            ax.bar(x_pos + width/2, frac_sig, width, color=SIG_COLOR,
                   edgecolor='black', linewidth=1.2,
                   label=f'Muons')
            ax.set_yscale('log')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['NO HIT (0)', 'HIT (1)'], fontsize=22, fontweight='bold')
            ax.set_ylabel('FRACTION OF EVENTS', fontsize=24, fontweight='bold', labelpad=15)
            for i, (fb, fs) in enumerate(zip(frac_bkg, frac_sig)):
                ax.text(i - width/2, fb + 0.01, f'{fb:.2%}', ha='center', fontsize=16, fontweight='bold')
                ax.text(i + width/2, fs + 0.01, f'{fs:.2%}', ha='center', fontsize=16, fontweight='bold')
        else:
            lo = np.percentile(pd.concat([sig_vals, bkg_vals]), 1)
            hi = np.percentile(pd.concat([sig_vals, bkg_vals]), 99)
            bins = np.linspace(lo, hi, 60)
            ax.hist(bkg_vals.clip(lo, hi), bins=bins, density=True, alpha=0.55,
                    color=BKG_COLOR, hatch='//', edgecolor=BKG_COLOR, linewidth=1.5,
                    label=f'Pions')
            ax.hist(sig_vals.clip(lo, hi), bins=bins, density=True, alpha=0.65,
                    color=SIG_COLOR, edgecolor='black', linewidth=0.8,
                    label=f'Muons')
            ax.set_yscale('log')
            ax.set_xlabel(feat_label, fontsize=26, fontweight='bold', labelpad=15)
            ax.set_ylabel('NORMALIZED COUNTS', fontsize=22, fontweight='bold', labelpad=15)

        ax.set_title(f'ENGINEERED CALORIMETER FEATURE\n{feat_name}',
                     fontsize=28, fontweight='bold', pad=25)
        ax.tick_params(axis='both', labelsize=20, width=1.5, length=8)
        ax.legend(fontsize=20, loc='best', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.4, linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

print("Done: training on all files, split by FileIndex, PDF report.")