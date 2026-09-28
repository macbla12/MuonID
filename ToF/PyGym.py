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

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

import torch
import torch.nn as nn
import numpy as np
import onnx

os.makedirs("Plots", exist_ok=True)
os.makedirs("ONNX", exist_ok=True)

# =============================================================================
# 0. Raw branch order. Single source of truth: matches the flat TTree written
#    by CombinedCaloToFAnalysis.cxx exactly, in that order.
#
#    Sentinel convention from the C++ side:
#      -999  -> "no measurement" (track never left a hit in that subsystem /
#                ToF never found a match). NOT a physical value.
#      0     -> genuine physical zero (e.g. ECalNumber = 0 hits is real,
#                not "unknown").
#    SENTINEL_COLS below lists every raw column that can carry the -999
#    convention; everything else is a real value with no missing marker.
#
#    NOTE: this now mirrors the FULL calorimeter shower-shape feature set
#    (Spread*, dispersions, energy concentration, ...) added on the C++ side,
#    not just Energy/Number/EoverP/MaxHitFrac.
# =============================================================================
RAW_COLS = [
    # ECal (full shower-shape set)
    'ECalEnergy', 'ECalNumber', 'ECalEoverP', 'ECalAvgHitEnergy',
    'ECalSpreadPhi', 'ECalSpreadEta', 'ECalSpreadR', 'ECalMaxHitFrac',
    'ECalEnergyStdDev', 'ECalEnergyConcentration',
    'ECalR_Disp', 'ECalR_DispWeighted', 'ECalEta_DispWeighted', 'ECalPhi_DispWeighted',
    # HCal (full shower-shape set)
    'HCalEnergy', 'HCalNumber', 'HCalEoverP', 'HCalAvgHitEnergy',
    'HCalSpreadPhi', 'HCalSpreadEta', 'HCalSpreadR', 'HCalMaxHitFrac',
    'HCalEnergyStdDev', 'HCalEnergyConcentration',
    'HCalR_Disp', 'HCalR_DispWeighted', 'HCalEta_DispWeighted', 'HCalPhi_DispWeighted',
    # ToF
    'ToFBeta', 'ToFMassSq', 'ToFNHitsBarrel', 'ToFNHitsEndcap', 'ToFNHitsTotal',
    'ToFMinDistBarrel', 'ToFMinDistEndcap', 'ToFAvgLenBarrel', 'ToFAvgLenEndcap', 'ToFHasToF',
    # Track kinematics
    'TrackMomentum', 'TrackEta', 'TrackPhi',
]

SENTINEL = -999.0

# Raw columns that use the -999 "missing" convention (need imputation before
# they're used in ratios / mismatches, otherwise -999 corrupts the math).
# Energy/Number stay OUT of this list on purpose: they are genuinely 0 when
# a track left no hits in that subsystem, never -999.
SENTINEL_COLS = [
    'ECalEoverP', 'ECalAvgHitEnergy', 'ECalSpreadPhi', 'ECalSpreadEta', 'ECalSpreadR',
    'ECalMaxHitFrac', 'ECalEnergyStdDev', 'ECalEnergyConcentration',
    'ECalR_Disp', 'ECalR_DispWeighted', 'ECalEta_DispWeighted', 'ECalPhi_DispWeighted',

    'HCalEoverP', 'HCalAvgHitEnergy', 'HCalSpreadPhi', 'HCalSpreadEta', 'HCalSpreadR',
    'HCalMaxHitFrac', 'HCalEnergyStdDev', 'HCalEnergyConcentration',
    'HCalR_Disp', 'HCalR_DispWeighted', 'HCalEta_DispWeighted', 'HCalPhi_DispWeighted',

    'ToFBeta', 'ToFMassSq',
    'ToFMinDistBarrel', 'ToFMinDistEndcap',
    'ToFAvgLenBarrel', 'ToFAvgLenEndcap',
]

with open("ONNX/raw_feature_order.txt", "w") as f:
    f.write(",".join(RAW_COLS) + "\n")


# =============================================================================
# 1. Load the combined training data
# =============================================================================
file_path = "ONNX/MLDataCaloToF.root"

with uproot.open(file_path) as f:
    df = f["MLDataTree"].arrays(library="pd")

# Binary label: muon (PDG 13) = signal, pion (PDG 211) = background.
df['IsMuon'] = (df['TruePDG'] == 13).astype(int)

n_sig_total = (df['IsMuon'] == 1).sum()
n_bkg_total = (df['IsMuon'] == 0).sum()
print(f"Loaded {len(df)} events.")
print(f"  Signal (muons) : {n_sig_total}")
print(f"  Background (pions): {n_bkg_total}")
print(f"  S/B ratio       : 1 : {n_bkg_total/n_sig_total:.1f}")

file_idx = df["FileIndex"].values
is_mu    = df["IsMuon"].values


# =============================================================================
# 2. Impute the -999 sentinels (median of valid entries, computed once on the
#    full sample). This is a fixed "typical missing value" substitute, not a
#    fitted statistic, so computing it on the full sample rather than
#    train-only is an acceptable simplification here -- flag it if you'd
#    rather fit it strictly on the training split only.
# =============================================================================
impute_values = {}

def impute_sentinel_column(series, sentinel=SENTINEL):
    valid = ~np.isclose(series.values, sentinel)
    median = float(np.median(series.values[valid])) if valid.any() else 0.0
    out = series.to_numpy(copy=True)
    out[~valid] = median
    return out, median

for col in SENTINEL_COLS:
    imputed, med = impute_sentinel_column(df[col])
    df[col] = imputed
    impute_values[col] = med


def safe_divide(num, denom):
    num = np.asarray(num, dtype=float)
    denom = np.asarray(denom, dtype=float)
    denom_nonzero = denom != 0
    result = np.zeros_like(num, dtype=float)
    result[denom_nonzero] = num[denom_nonzero] / denom[denom_nonzero]
    return result


# =============================================================================
# 3. Build engineered features
# =============================================================================

def engineer_calo_features(df):
    # Raw pass-through (already imputed) + calorimeter-derived ratios.
    # Full shower-shape set for both ECal and HCal, matching RAW_COLS order.
    s = df[[
        'ECalEnergy', 'ECalNumber', 'ECalEoverP', 'ECalAvgHitEnergy',
        'ECalSpreadPhi', 'ECalSpreadEta', 'ECalSpreadR', 'ECalMaxHitFrac',
        'ECalEnergyStdDev', 'ECalEnergyConcentration',
        'ECalR_Disp', 'ECalR_DispWeighted', 'ECalEta_DispWeighted', 'ECalPhi_DispWeighted',

        'HCalEnergy', 'HCalNumber', 'HCalEoverP', 'HCalAvgHitEnergy',
        'HCalSpreadPhi', 'HCalSpreadEta', 'HCalSpreadR', 'HCalMaxHitFrac',
        'HCalEnergyStdDev', 'HCalEnergyConcentration',
        'HCalR_Disp', 'HCalR_DispWeighted', 'HCalEta_DispWeighted', 'HCalPhi_DispWeighted',
    ]].copy()

    s['ECalHasHit'] = (df['ECalNumber'] > 0).astype(float)
    s['HCalHasHit'] = (df['HCalNumber'] > 0).astype(float)

    totalE = df['ECalEnergy'] + df['HCalEnergy']
    s['ECalFrac']     = safe_divide(df['ECalEnergy'].values, totalE.values)
    s['HCalFrac']     = safe_divide(df['HCalEnergy'].values, totalE.values)
    s['HitRatio']     = safe_divide(df['ECalNumber'].values, df['HCalNumber'].values)
    s['EoverP_ratio'] = safe_divide(df['ECalEoverP'].values, df['HCalEoverP'].values)
    s['logECal']      = np.log1p(df['ECalEnergy'])
    s['logHCal']      = np.log1p(df['HCalEnergy'])
    s['MaxHitFrac_mismatch'] = (df['ECalMaxHitFrac'] - df['HCalMaxHitFrac']).abs()

    # -- extra mismatch/ratio features for the newly-added shower-shape set --
    s['AvgHitEnergy_ratio']         = safe_divide(df['ECalAvgHitEnergy'].values, df['HCalAvgHitEnergy'].values)
    s['SpreadR_mismatch']           = (df['ECalSpreadR'] - df['HCalSpreadR']).abs()
    s['EnergyConcentration_mismatch'] = (df['ECalEnergyConcentration'] - df['HCalEnergyConcentration']).abs()
    s['R_DispWeighted_ratio']       = safe_divide(df['ECalR_DispWeighted'].values, df['HCalR_DispWeighted'].values)

    return s


def engineer_tof_features(df):
    # Raw pass-through (already imputed) + ToF-derived combinations.
    s = df[['ToFBeta', 'ToFMassSq', 'ToFNHitsBarrel', 'ToFNHitsEndcap', 'ToFNHitsTotal',
            'ToFMinDistBarrel', 'ToFMinDistEndcap', 'ToFAvgLenBarrel', 'ToFAvgLenEndcap',
            'ToFHasToF']].copy()

    nb = df['ToFNHitsBarrel'].values
    ne = df['ToFNHitsEndcap'].values
    s['ToFHitFracBarrel'] = safe_divide(nb, nb + ne)

    # Which subdetector actually produced the ToF match. Computed on the
    # *pre-imputation* raw values would be more precise, but ToFNHitsBarrel/
    # Endcap (real counts, never sentinel) already tell us the same thing,
    # so we use those instead of re-deriving from the (now-imputed) distances.
    matched_where = np.full(len(df), -1.0)
    matched_where[(nb > 0) & (ne == 0)] = 0.0   # barrel only
    matched_where[(nb == 0) & (ne > 0)] = 1.0   # endcap only
    matched_where[(nb > 0) & (ne > 0)]  = 2.0   # both
    s['ToFMatchedSubdet'] = matched_where

    s['ToFMinDistCombined'] = np.where(
        (nb > 0) & (ne > 0),
        np.minimum(df['ToFMinDistBarrel'].values, df['ToFMinDistEndcap'].values),
        np.where(nb > 0, df['ToFMinDistBarrel'].values,
                 np.where(ne > 0, df['ToFMinDistEndcap'].values, impute_values['ToFMinDistBarrel'])),
    )

    return s


def engineer_track_features(df):
    s = df[['TrackMomentum', 'TrackEta', 'TrackPhi']].copy()
    return s


X_calo  = engineer_calo_features(df)
X_tof   = engineer_tof_features(df)
X_track = engineer_track_features(df)
X       = pd.concat([X_calo, X_tof, X_track], axis=1)
y        = df['IsMuon'].astype(int)
file_idx = df['FileIndex'].astype(int)

print(f"\nTotal number of features: {X.shape[1]}")
all_features = X.columns.tolist()

with open("ONNX/feature_order.txt", "w") as f:
    f.write(",".join(all_features) + "\n")

with open("ONNX/tof_impute_values.txt", "w") as f:
    for col, val in impute_values.items():
        f.write(f"{col},{val:.8f}\n")


# =============================================================================
# 4. Split data stratified by class and file index
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
# 5. Train the XGBoost classifier
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

fp_penalty = 0.5
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
# A. Torch preprocessor matching the feature-engineering pipeline
#
#    Single input "raw_features": shape (N, len(RAW_COLS)), columns in the
#    exact order of RAW_COLS -- the same order the C++ side fills a flat
#    float array with (see ONNX/raw_feature_order.txt).
#
#    Sentinel handling: any RAW_COLS entry equal to -999 (within tolerance)
#    is replaced by the stored per-column median BEFORE any ratio/derived
#    feature is computed, mirroring the pandas pipeline exactly.
# =============================================================================
N_RAW = len(RAW_COLS)
SENTINEL_TOL = 1.0  # physical values never come within 1.0 of -999


class MuonIDPreprocessor(nn.Module):
    def __init__(self, mean, scale, impute_values):
        super().__init__()
        self.register_buffer("mean", torch.tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(np.asarray(scale), dtype=torch.float32))

        # Build a per-RAW_COLS-index impute buffer (0 where not applicable).
        impute_vec = np.zeros(N_RAW, dtype=np.float32)
        impute_mask = np.zeros(N_RAW, dtype=np.float32)
        for col, val in impute_values.items():
            i = RAW_COLS.index(col)
            impute_vec[i] = val
            impute_mask[i] = 1.0
        self.register_buffer("impute_vec", torch.tensor(impute_vec))
        self.register_buffer("impute_mask", torch.tensor(impute_mask))
        self.register_buffer(
            "impute_dist_barrel",
            torch.tensor(float(impute_values['ToFMinDistBarrel']), dtype=torch.float32),
        )

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
        raw = raw_features

        # ---- 1. impute sentinels on the whole raw block at once ----
        is_sentinel = torch.abs(raw - SENTINEL) < SENTINEL_TOL
        raw_imputed = torch.where(
            is_sentinel & (self.impute_mask.unsqueeze(0) > 0),
            self.impute_vec.unsqueeze(0).expand_as(raw),
            raw,
        )

        col = lambda name: self.col(raw_imputed, name)

        # ---- ECal (full shower-shape set) ----
        ECalEnergy              = col('ECalEnergy')
        ECalNumber              = col('ECalNumber')
        ECalEoverP              = col('ECalEoverP')
        ECalAvgHitEnergy        = col('ECalAvgHitEnergy')
        ECalSpreadPhi           = col('ECalSpreadPhi')
        ECalSpreadEta           = col('ECalSpreadEta')
        ECalSpreadR             = col('ECalSpreadR')
        ECalMaxHitFrac          = col('ECalMaxHitFrac')
        ECalEnergyStdDev        = col('ECalEnergyStdDev')
        ECalEnergyConcentration = col('ECalEnergyConcentration')
        ECalR_Disp              = col('ECalR_Disp')
        ECalR_DispWeighted      = col('ECalR_DispWeighted')
        ECalEta_DispWeighted    = col('ECalEta_DispWeighted')
        ECalPhi_DispWeighted    = col('ECalPhi_DispWeighted')

        # ---- HCal (full shower-shape set) ----
        HCalEnergy              = col('HCalEnergy')
        HCalNumber               = col('HCalNumber')
        HCalEoverP               = col('HCalEoverP')
        HCalAvgHitEnergy         = col('HCalAvgHitEnergy')
        HCalSpreadPhi            = col('HCalSpreadPhi')
        HCalSpreadEta            = col('HCalSpreadEta')
        HCalSpreadR              = col('HCalSpreadR')
        HCalMaxHitFrac           = col('HCalMaxHitFrac')
        HCalEnergyStdDev         = col('HCalEnergyStdDev')
        HCalEnergyConcentration  = col('HCalEnergyConcentration')
        HCalR_Disp                = col('HCalR_Disp')
        HCalR_DispWeighted        = col('HCalR_DispWeighted')
        HCalEta_DispWeighted      = col('HCalEta_DispWeighted')
        HCalPhi_DispWeighted      = col('HCalPhi_DispWeighted')

        ToFBeta   = col('ToFBeta')
        ToFMassSq = col('ToFMassSq')
        ToFNHitsBarrel = col('ToFNHitsBarrel')
        ToFNHitsEndcap = col('ToFNHitsEndcap')
        ToFNHitsTotal  = col('ToFNHitsTotal')
        ToFMinDistBarrel = col('ToFMinDistBarrel')
        ToFMinDistEndcap = col('ToFMinDistEndcap')
        ToFAvgLenBarrel  = col('ToFAvgLenBarrel')
        ToFAvgLenEndcap  = col('ToFAvgLenEndcap')
        ToFHasToF = col('ToFHasToF')

        TrackMomentum = col('TrackMomentum')
        TrackEta      = col('TrackEta')
        TrackPhi      = col('TrackPhi')

        # ---- 2. calo engineered features (order MUST match engineer_calo_features) ----
        ECalHasHit = (ECalNumber > 0).float()
        HCalHasHit = (HCalNumber > 0).float()

        totalE = ECalEnergy + HCalEnergy
        ECalFrac     = sd(ECalEnergy, totalE)
        HCalFrac     = sd(HCalEnergy, totalE)
        HitRatio     = sd(ECalNumber, HCalNumber)
        EoverP_ratio = sd(ECalEoverP, HCalEoverP)
        logECal      = torch.log1p(ECalEnergy)
        logHCal      = torch.log1p(HCalEnergy)
        MaxHitFrac_mismatch = torch.abs(ECalMaxHitFrac - HCalMaxHitFrac)

        AvgHitEnergy_ratio           = sd(ECalAvgHitEnergy, HCalAvgHitEnergy)
        SpreadR_mismatch             = torch.abs(ECalSpreadR - HCalSpreadR)
        EnergyConcentration_mismatch = torch.abs(ECalEnergyConcentration - HCalEnergyConcentration)
        R_DispWeighted_ratio         = sd(ECalR_DispWeighted, HCalR_DispWeighted)

        calo_features = torch.cat([
            ECalEnergy, ECalNumber, ECalEoverP, ECalAvgHitEnergy,
            ECalSpreadPhi, ECalSpreadEta, ECalSpreadR, ECalMaxHitFrac,
            ECalEnergyStdDev, ECalEnergyConcentration,
            ECalR_Disp, ECalR_DispWeighted, ECalEta_DispWeighted, ECalPhi_DispWeighted,

            HCalEnergy, HCalNumber, HCalEoverP, HCalAvgHitEnergy,
            HCalSpreadPhi, HCalSpreadEta, HCalSpreadR, HCalMaxHitFrac,
            HCalEnergyStdDev, HCalEnergyConcentration,
            HCalR_Disp, HCalR_DispWeighted, HCalEta_DispWeighted, HCalPhi_DispWeighted,

            ECalHasHit, HCalHasHit,
            ECalFrac, HCalFrac, HitRatio, EoverP_ratio, logECal, logHCal,
            MaxHitFrac_mismatch,
            AvgHitEnergy_ratio, SpreadR_mismatch, EnergyConcentration_mismatch, R_DispWeighted_ratio,
        ], dim=1)

        # ---- 3. ToF engineered features (unchanged) ----
        ToFHitFracBarrel = sd(ToFNHitsBarrel, ToFNHitsBarrel + ToFNHitsEndcap)

        both = (ToFNHitsBarrel > 0) & (ToFNHitsEndcap > 0)
        barrel_only = (ToFNHitsBarrel > 0) & (ToFNHitsEndcap == 0)
        endcap_only = (ToFNHitsBarrel == 0) & (ToFNHitsEndcap > 0)

        ToFMatchedSubdet = torch.full_like(ToFNHitsBarrel, -1.0)
        ToFMatchedSubdet = torch.where(barrel_only, torch.zeros_like(ToFMatchedSubdet), ToFMatchedSubdet)
        ToFMatchedSubdet = torch.where(endcap_only, torch.ones_like(ToFMatchedSubdet), ToFMatchedSubdet)
        ToFMatchedSubdet = torch.where(both, torch.full_like(ToFMatchedSubdet, 2.0), ToFMatchedSubdet)

        min_dist_both = torch.minimum(ToFMinDistBarrel, ToFMinDistEndcap)
        ToFMinDistCombined = torch.where(
            both, min_dist_both,
            torch.where(barrel_only, ToFMinDistBarrel,
                        torch.where(endcap_only, ToFMinDistEndcap,
                                    self.impute_dist_barrel.expand_as(ToFMinDistBarrel))),
        )

        tof_features = torch.cat([
            ToFBeta, ToFMassSq, ToFNHitsBarrel, ToFNHitsEndcap, ToFNHitsTotal,
            ToFMinDistBarrel, ToFMinDistEndcap, ToFAvgLenBarrel, ToFAvgLenEndcap,
            ToFHasToF, ToFHitFracBarrel, ToFMatchedSubdet, ToFMinDistCombined,
        ], dim=1)

        # ---- 4. track kinematics pass-through ----
        track_features = torch.cat([TrackMomentum, TrackEta, TrackPhi], dim=1)

        X = torch.cat([calo_features, tof_features, track_features], dim=1)

        return (X - self.mean) / self.scale


def export_preprocessor(mean, scale, impute_values, out_path, opset_version=18):
    model = MuonIDPreprocessor(mean, scale, impute_values)
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
    means, scales, impute_values, out_path="ONNX/preprocessing.onnx"
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

# Quick end-to-end check on one test event -- note this must use the RAW
# (pre-imputation) branch values straight from the tree, since the ONNX
# preprocessor does the sentinel imputation internally.
import onnxruntime as ort

with uproot.open(file_path) as f:
    df_raw_check = f["MLDataTree"].arrays(library="pd")

sess = ort.InferenceSession("ONNX/xgb_muonID.onnx")

raw_row = df_raw_check.iloc[[0]][RAW_COLS].values.astype(np.float32)
onnx_out = sess.run(None, {"raw_features": raw_row})[0]

row_features = X.iloc[[0]]
row_scaled = scaler.transform(row_features)
pandas_out = xgb.predict_proba(row_scaled)[:, 1]

print("\nComparison for one event:")
print("  full ONNX pipeline :", onnx_out.ravel())
print("  original pandas    :", pandas_out)


# =============================================================================
# 6. Metrics and decision thresholds
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
# 7. Spearman correlations and correlation with the label
# =============================================================================
X_corr = X.copy()
corr_matrix = X_corr.corr(method='spearman')
y_float = y.astype(float)
feature_label_corr = X_corr.corrwith(y_float, method='spearman').sort_values(
    key=lambda x: x.abs(), ascending=False
)


# =============================================================================
# 8. Create the PDF report with SHAP diagnostics
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
    fig.suptitle('Muon vs Pion ID — XGBoost (Calorimeter + ToF features)\n'
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
                xticklabels=['Pion', 'Muon'],
                yticklabels=['Pion', 'Muon'],
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
                color=BKG_COLOR, label=f'Pion (N={n_bkg_test})',
                hatch='//', edgecolor=BKG_COLOR)
    ax_resp.hist(y_probs[y_test==1], bins=bins, alpha=0.6, density=True,
                color=SIG_COLOR, label=f'Muon (N={n_sig_test})')
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
                 color=BKG_COLOR, label='Pion (train)', hatch='//')
    axes[1].hist(y_probs[y_test==0],        bins=bins, alpha=0.7, density=True,
                 color=BKG_COLOR, label='Pion (test)',
                 histtype='step', linewidth=2)
    axes[1].hist(y_probs_train[y_train==1], bins=bins, alpha=0.4, density=True,
                 color=SIG_COLOR, label='Muon (train)')
    axes[1].hist(y_probs[y_test==1],        bins=bins, alpha=0.7, density=True,
                 color=SIG_COLOR, label='Muon (test)',
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

    def feat_group_color(fname):
        if 'ECal' in fname:
            return SIG_COLOR
        if 'HCal' in fname:
            return BKG_COLOR
        if 'ToF' in fname:
            return PUR_COLOR
        return ACC_COLOR

    colors = [feat_group_color(all_features[i]) for i in idx_top]

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
                    Patch(facecolor=PUR_COLOR, label='ToF features'),
                    Patch(facecolor=ACC_COLOR, label='Track kinematics')]
    axes[0].legend(handles=legend_elements, fontsize=14, loc='lower right')

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
    fig.suptitle('Key Physics Variable Distributions: Muon vs Pion\n'
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
                color=BKG_COLOR, label='Pion', hatch='//', edgecolor=BKG_COLOR)
        ax.hist(sig_vals.clip(lo, hi), bins=bins, alpha=0.6, density=True,
                color=SIG_COLOR, label='Muon')
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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7)
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
    ax.set_yticklabels(feature_label_corr.index, fontsize=7)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Spearman ρ with the label (IsMuon)', fontsize=10)
    ax.set_title('Feature correlation with muon/pion class\n'
                 'Blue = positive (→ muon), Red = negative (→ pion)',
                 color=SIG_COLOR, pad=8, fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 7: SHAP model explanation
    explainer = shap.TreeExplainer(xgb)
    sample_idx = np.random.choice(len(X_test_sc), size=min(5000, len(X_test_sc)), replace=False)
    X_shap = X_test_sc[sample_idx]
    shap_values = explainer.shap_values(X_shap)

    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=all_features, show=False)
    plt.title('SHAP Summary Plot — Feature Impact on P(Muon)', fontsize=14)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    top_feat = imp_series.nlargest(1).index[0]
    fig = plt.figure(figsize=(8, 6))
    shap.dependence_plot(top_feat, shap_values, X_shap,
                         feature_names=all_features, show=False)
    plt.title(f'SHAP Dependence: {top_feat}', fontsize=14)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # SHAP per FileIndex for domain-shift diagnostics
    unique_files = sorted(file_idx.unique())

    for fidx in unique_files:
        mask_f = (file_idx.iloc[idx_test] == fidx)
        if mask_f.sum() < 50:
            continue

        X_f = X_test_sc[mask_f]
        sample_idx_f = np.random.choice(len(X_f), size=min(3000, len(X_f)), replace=False)
        X_shap_f = X_f[sample_idx_f]

        shap_values_f = explainer.shap_values(X_shap_f)

        fig = plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_f, X_shap_f, feature_names=all_features, show=False)
        plt.title(f'SHAP Summary — FileIndex={fidx}', fontsize=14)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        top_feat = imp_series.nlargest(1).index[0]
        fig = plt.figure(figsize=(8, 6))
        shap.dependence_plot(top_feat, shap_values_f, X_shap_f,
                             feature_names=all_features, show=False)
        plt.title(f'SHAP Dependence: {top_feat}  (FileIndex={fidx})', fontsize=14)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # SHAP similarity matrix (FileIndex vs FileIndex)
    # =========================================================================
    shap_importances = {}

    for fidx in unique_files:
        mask_f = (file_idx.iloc[idx_test] == fidx)
        if mask_f.sum() < 50:
            continue

        X_f = X_test_sc[mask_f]
        sample_idx_f = np.random.choice(len(X_f), size=min(3000, len(X_f)), replace=False)
        X_shap_f = X_f[sample_idx_f]

        shap_values_f = explainer.shap_values(X_shap_f)
        shap_importances[fidx] = np.mean(np.abs(shap_values_f), axis=0)

    file_ids = sorted(shap_importances.keys())
    if len(file_ids) > 1:
        shap_matrix = np.vstack([shap_importances[f] for f in file_ids])
        similarity = np.corrcoef(shap_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(similarity, annot=True, fmt=".2f",
                    xticklabels=file_ids, yticklabels=file_ids,
                    cmap="coolwarm", vmin=-1, vmax=1,
                    cbar_kws={'label': 'Correlation'})
        ax.set_title("SHAP Similarity Matrix — FileIndex vs FileIndex\n"
                     "(correlation between SHAP importance vectors)", fontsize=14)
        ax.set_xlabel("FileIndex")
        ax.set_ylabel("FileIndex")

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

print("Done: training on Calo+ToF features, muon vs pion, PDF report and SHAP diagnostics.")