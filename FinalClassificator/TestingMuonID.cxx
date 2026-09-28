// TestingMacro_Combined_Podio.cxx
//
// Combined podio/EDM4eic version merging TestingMacro_Podio.cxx (large-p,
// calorimeter-only feature set / model) and TestingMacro_Fixed_Podio.cxx
// (low-p, calo+ToF feature set / model).
//
// ROUTING LOGIC
// -------------
//   trackP >= P_SPLIT_GEV  -> "HighP" path:
//       * detailed calo features (spread in phi/eta/R, energy std-dev,
//         concentration, weighted dispersions, ...) exactly as in
//         TestingMacro_Podio.cxx
//       * 29-feature vector, fed into session_HighP
//         (default: "ONNX/xgb_muonID.onnx", unchanged from the original)
//
//   trackP <  P_SPLIT_GEV  -> "LowP" path:
//       * FULL calo shower-shape features (same detailed set as HighP:
//         Energy, Number, EoverP, AvgHitEnergy, Spread phi/eta/R,
//         MaxHitFrac, EnergyStdDev, EnergyConcentration, R_Disp,
//         R_DispWeighted, Eta/Phi_DispWeighted) for BOTH ECal and HCal,
//         PLUS ToF features (beta, mass^2, hit counts/distances/lengths),
//         PLUS track kinematics (p, eta, phi) -- exactly the RAW_COLS
//         layout produced by CombinedCaloToFAnalysis.cxx and consumed by
//         TrainMuonID.py.
//       * 41-feature vector, fed into session_LowP
//         (default: "../ToF/ONNX/xgb_muonID.onnx", retrained on the full
//         feature set -- the merged ONNX graph does sentinel imputation +
//         feature engineering + scaling internally, so this macro only
//         needs to supply the RAW_COLS values in the exact order below).
//
// >>> TODO (you said you'll provide this yourself): <<<
//   1) Set LOWP_ONNX_PATH below to the .onnx file you trained for the
//      low-momentum regime (retrained with the full calo feature set --
//      see TrainMuonID.py / ONNX/raw_feature_order.txt for the exact
//      column order this macro must match).
//   2) If your low-p model does NOT expect exactly the same 41-value
//      "RAW_COLS" layout as TrainMuonID.py (LowP_build_raw_features),
//      edit that function (or the call site) to match your model's input
//      order/shape.
//   3) MUON_ID_CUT_LOWP is a placeholder (0.5f) - retune it for your own
//      low-p model.
//
// Everything else (file globbing, collection names, cuts, histogram
// definitions, high-p model/feature code) is left as close as possible to
// the two original macros so it should run out of the box against the same
// input files.

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TROOT.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TMath.h>
#include <TString.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <glob.h>

#include <onnxruntime_cxx_api.h>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include <edm4hep/MCParticleCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/MCRecoParticleAssociationCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/MCRecoClusterParticleAssociationCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <edm4eic/TrackerHitCollection.h>

// Needed for the LowP path (RK4 propagation / field map used by the ToF
// feature calculation). Must sit alongside this file, unchanged.
#include "ToFSim.cxx"

using namespace std;

// =====================================================================
// Momentum split point between the two models. Tracks with trackP below
// this value go through the LowP (calo+ToF) path/model; at or above it
// they go through the HighP (calo-only, detailed shower shape) path/model.
// =====================================================================
static const double P_SPLIT_GEV = 1.0;

// >>> TODO: point this at your own low-momentum ONNX model (retrained on
//     the full 41-feature RAW_COLS set -- see TrainMuonID.py). <<<
static const char *LOWP_ONNX_PATH  = "../ToF/ONNX/xgb_muonID.onnx";
static const char *HIGHP_ONNX_PATH = "../CalorimetryHits/ONNX/xgb_muonID.onnx";

static const float MUON_ID_CUT_HIGHP = 0.5f; // from TestingMacro_Podio.cxx
static const float MUON_ID_CUT_LOWP  = 0.5f;   // placeholder, retune for your own model

static const double TIMING_CUT_NS = 20.0;

static const float MISSING_SENTINEL = -999.f;

static const int NUMBEROFEVENTS = 300000;

// =====================================================================
// Small helper: expand a shell-style wildcard (e.g. "reco_*.root") into a
// list of real file paths. Replaces TChain::Add(pattern).
// =====================================================================
vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i)
        result.push_back(string(glob_result.gl_pathv[i]));
    globfree(&glob_result);
    return result;
}

// =====================================================================
// Shared: detector -> truth-association-collection lookup.
// For a cluster collection "XClusters" the truth-matching collection is
// assumed to be named "XClusterAssociations"
// (edm4eic::MCRecoClusterParticleAssociation).
// =====================================================================
struct DetectorAssoc
{
    string clusterName;
    string assocCollName;
};

DetectorAssoc MakeDetectorAssoc(const string &clusterName)
{
    DetectorAssoc d;
    d.clusterName = clusterName;
    d.assocCollName = clusterName.substr(0, clusterName.size() - 1) + "Associations";
    cout << "  Detector '" << clusterName << "' -> Association collection '"
         << d.assocCollName << "'" << endl;
    return d;
}

// Generic ONNX runner - works for both models since both take a single
// "raw_features" float input and return "probabilities", we just read
// P(muon) = probs[1].
float run_muon_id(Ort::Session &session, Ort::MemoryInfo &mem, const std::vector<float> &raw)
{
    int64_t shape[2] = {1, (int64_t)raw.size()};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float *>(raw.data()), raw.size(), shape, 2);

    const char *input_names[]  = {"raw_features"};
    const char *output_names[] = {"probabilities"};

    auto output = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    float *probs = output[0].GetTensorMutableData<float>();
    return probs[1]; // P(muon)
}

// =====================================================================
// ==================  SHARED CALO FEATURES (full shower-shape)  =======
// Used by BOTH the HighP path (29-feature model, no ToF) and the LowP
// path (41-feature model, calo + ToF). Renamed from "HighP_*" to make
// clear it's shared, but kept the same names/behaviour so nothing about
// the HighP branch changes.
// =====================================================================

struct HighP_TrackCalHits
{
    double sumEnergy = 0.0;
    double phiMin = 1e9, phiMax = -1e9;
    double etaMin = 1e9, etaMax = -1e9;
    double Rmin = 1e9, Rmax = -1e9;
    double maxHitE = -1.0;
    vector<double> R, dEta, dPhi, E;
};

std::vector<float> HighP_build_raw_features(
    float ECalEnergy, float HCalEnergy,
    float ECalNumber, float HCalNumber,
    float ECalEoverP, float HCalEoverP,
    float ECalAvgHitEnergy, float HCalAvgHitEnergy,
    float ECalSpreadPhi, float ECalSpreadEta, float ECalSpreadR,
    float HCalSpreadPhi, float HCalSpreadEta, float HCalSpreadR,
    float ECalMaxHitFrac, float HCalMaxHitFrac,
    float ECalEnergyStdDev, float HCalEnergyStdDev,
    float ECalEnergyConcentration, float HCalEnergyConcentration,
    float ECalR_Disp, float ECalR_DispWeighted,
    float ECalEta_DispWeighted, float ECalPhi_DispWeighted,
    float HCalR_Disp, float HCalR_DispWeighted,
    float HCalEta_DispWeighted, float HCalPhi_DispWeighted,
    float TrackMomentum, float TrackEta)
{
    return {
        ECalEnergy, HCalEnergy,
        ECalNumber, HCalNumber,
        ECalEoverP, HCalEoverP,
        ECalAvgHitEnergy, HCalAvgHitEnergy,
        ECalSpreadPhi, ECalSpreadEta, ECalSpreadR,
        HCalSpreadPhi, HCalSpreadEta, HCalSpreadR,
        ECalMaxHitFrac, HCalMaxHitFrac,
        ECalEnergyStdDev, HCalEnergyStdDev,
        ECalEnergyConcentration, HCalEnergyConcentration,
        ECalR_Disp, ECalR_DispWeighted,
        ECalEta_DispWeighted, ECalPhi_DispWeighted,
        HCalR_Disp, HCalR_DispWeighted,
        HCalEta_DispWeighted, HCalPhi_DispWeighted,
        TrackMomentum, TrackEta
    };
}

HighP_TrackCalHits HighP_CollectHits(const edm4hep::MCParticle &simPart,
                                      const vector<DetectorAssoc> &detectors,
                                      const podio::Frame &frame,
                                      const TLorentzVector &Partic,
                                      double timingCutNs)
{
    HighP_TrackCalHits result;

    for (const auto &det : detectors)
    {
        const auto &assocColl =
            frame.get<edm4eic::MCRecoClusterParticleAssociationCollection>(det.assocCollName);

        for (const auto &assoc : assocColl)
        {
            if (assoc.getSim() != simPart) continue;

            const auto cluster = assoc.getRec();

            for (const auto &hit : cluster.getHits())
            {
                float HitE = hit.getEnergy();
                float HitT = hit.getTime();
                auto pos = hit.getPosition();

                if (HitT > timingCutNs) continue;

                TVector3 hitVec(pos.x, pos.y, pos.z);
                double hitEta = hitVec.Eta();
                double hitPhi = hitVec.Phi();
                double R = sqrt(pos.x * pos.x + pos.y * pos.y);

                double dEta = hitEta - Partic.Eta();
                double dPhi = TVector2::Phi_mpi_pi(hitPhi - Partic.Phi());

                result.sumEnergy += HitE;
                result.R.push_back(R);
                result.dEta.push_back(dEta);
                result.dPhi.push_back(dPhi);
                result.E.push_back(HitE);

                result.phiMin = min(result.phiMin, hitPhi); result.phiMax = max(result.phiMax, hitPhi);
                result.etaMin = min(result.etaMin, hitEta); result.etaMax = max(result.etaMax, hitEta);
                result.Rmin   = min(result.Rmin, R);        result.Rmax   = max(result.Rmax, R);

                if (HitE > result.maxHitE) result.maxHitE = HitE;
            }
        }
    }

    return result;
}

struct HighP_TrackFeatures
{
    float Energy = 0.f;
    float Number = 0.f;
    float EoverP = 0.f;
    float AvgHitEnergy = 0.f;
    float SpreadPhi = 0.f;
    float SpreadEta = 0.f;
    float SpreadR = 0.f;
    float MaxHitFrac = 0.f;
    float EnergyStdDev = 0.f;
    float EnergyConcentration = 0.f;
    float R_Disp = 0.f;
    float R_DispWeighted = 0.f;
    float Eta_DispWeighted = 0.f;
    float Phi_DispWeighted = 0.f;
};

HighP_TrackFeatures HighP_ComputeTrackFeatures(const HighP_TrackCalHits &hits, double trackP)
{
    HighP_TrackFeatures f;

    if (hits.sumEnergy <= 0) return f;

    int n = static_cast<int>(hits.E.size());
    if (n == 0) return f;

    double sumEnergy = hits.sumEnergy;
    double meanE = sumEnergy / n;

    double sumSqDevE = 0.0, sumE2 = 0.0;
    for (double e : hits.E)
    {
        double devE = e - meanE;
        sumSqDevE += devE * devE;
        sumE2 += e * e;
    }
    double energyStdDev = sqrt(sumSqDevE / n);
    double energyConcentration = sumE2 / (sumEnergy * sumEnergy);

    double spreadPhi = hits.phiMax - hits.phiMin;
    double spreadEta = hits.etaMax - hits.etaMin;
    double spreadR   = hits.Rmax - hits.Rmin;

    double sumRw = 0.0;
    for (int k = 0; k < n; ++k) sumRw += hits.R[k] * hits.E[k];
    double meanR_w = sumRw / sumEnergy;

    double sumR2diff = 0.0, sumR2diffW = 0.0;
    double sumDEta2diffW = 0.0, sumDPhi2diffW = 0.0;
    for (int k = 0; k < n; ++k)
    {
        double dR = hits.R[k] - meanR_w;
        sumR2diff   += dR * dR;
        sumR2diffW  += dR * dR * hits.E[k];

        sumDEta2diffW += hits.dEta[k] * hits.dEta[k] * hits.E[k];
        sumDPhi2diffW += hits.dPhi[k] * hits.dPhi[k] * hits.E[k];
    }

    double R_disp_unweighted = (n > 1) ? sqrt(sumR2diff / (n - 1)) : 0.0;
    double R_disp_weighted   = sqrt(sumR2diffW / sumEnergy);
    double Eta_disp_weighted = sqrt(sumDEta2diffW / sumEnergy);
    double Phi_disp_weighted = sqrt(sumDPhi2diffW / sumEnergy);

    f.Energy               = static_cast<float>(sumEnergy);
    f.Number               = static_cast<float>(n);
    f.EoverP               = static_cast<float>(sumEnergy / trackP);
    f.AvgHitEnergy         = static_cast<float>(meanE);
    f.SpreadPhi            = static_cast<float>(spreadPhi);
    f.SpreadEta            = static_cast<float>(spreadEta);
    f.SpreadR              = static_cast<float>(spreadR);
    f.MaxHitFrac           = static_cast<float>(hits.maxHitE / sumEnergy);
    f.EnergyStdDev         = static_cast<float>(energyStdDev);
    f.EnergyConcentration  = static_cast<float>(energyConcentration);
    f.R_Disp               = static_cast<float>(R_disp_unweighted);
    f.R_DispWeighted       = static_cast<float>(R_disp_weighted);
    f.Eta_DispWeighted     = static_cast<float>(Eta_disp_weighted);
    f.Phi_DispWeighted     = static_cast<float>(Phi_disp_weighted);

    return f;
}

// =====================================================================
// ==============================  ToF  =================================
// Used only by the LowP path. Unchanged from TestingMacro_Fixed_Podio.cxx.
// =====================================================================

struct BetaEstimate { double beta; bool valid; };

BetaEstimate CombineBeta(const std::vector<std::pair<double,double>> &hits)
{
    double sumXX = 0.0, sumXT = 0.0;
    for (auto &h : hits) {
        double t = h.first;
        double L = h.second;
        double x = L / c_light;
        sumXX += x * x;
        sumXT += x * t;
    }
    if (sumXX <= 0.0) return {0.0, false};
    double u_hat = sumXT / sumXX;
    if (u_hat <= 0.0) return {0.0, false};
    return {1.0 / u_hat, true};
}

struct LowP_TrackToFFeatures
{
    float Beta         = -999.f;
    float MassSq       = -999.f;
    float NHitsBarrel  = 0.f;
    float NHitsEndcap  = 0.f;
    float NHitsTotal   = 0.f;
    float MinDistBarrel= -999.f;
    float MinDistEndcap= -999.f;
    float AvgLenBarrel = -999.f;
    float AvgLenEndcap = -999.f;
    float HasToF       = 0.f;
};

template <typename HitCollection>
LowP_TrackToFFeatures LowP_ComputeToFFeatures(
    const TLorentzVector &Partic, int charge,
    const HitCollection &barrelHits, const HitCollection &endcapHits,
    double dR_cut_barrel, double dR_cut_endcap,
    double dist_cut_barrel, double dist_cut_endcap)
{
    LowP_TrackToFFeatures f;

    double trackEta = Partic.Eta();
    double trackPhi = Partic.Phi();

    std::vector<std::pair<double,double>> matchedBarrel; // {time, length}
    std::vector<std::pair<double,double>> matchedEndcap;

    double sumLenBarrel = 0.0, minDistBarrel = 1e18;
    double sumLenEndcap = 0.0, minDistEndcap = 1e18;

    for (const auto &hit : barrelHits) {
        auto pos = hit.getPosition();
        TVector3 ToFPos(pos.x, pos.y, pos.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);

        if (dR < dR_cut_barrel && charge * dPhi < 0) {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);
            if (tof.distance_to_TOF < dist_cut_barrel) {
                double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                matchedBarrel.push_back({hit.getTime(), length});
                sumLenBarrel += length;
                if (tof.distance_to_TOF < minDistBarrel) minDistBarrel = tof.distance_to_TOF;
            }
        }
    }

    for (const auto &hit : endcapHits) {
        auto pos = hit.getPosition();
        TVector3 ToFPos(pos.x, pos.y, pos.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);

        if (dR < dR_cut_endcap && charge * dPhi < 0) {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);
            if (tof.distance_to_TOF < dist_cut_endcap) {
                double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                matchedEndcap.push_back({hit.getTime(), length});
                sumLenEndcap += length;
                if (tof.distance_to_TOF < minDistEndcap) minDistEndcap = tof.distance_to_TOF;
            }
        }
    }

    f.NHitsBarrel = (float)matchedBarrel.size();
    f.NHitsEndcap = (float)matchedEndcap.size();
    f.NHitsTotal  = f.NHitsBarrel + f.NHitsEndcap;

    f.MinDistBarrel = matchedBarrel.empty() ? -999.f : (float)minDistBarrel;
    f.MinDistEndcap = matchedEndcap.empty() ? -999.f : (float)minDistEndcap;
    f.AvgLenBarrel  = matchedBarrel.empty() ? -999.f : (float)(sumLenBarrel / matchedBarrel.size());
    f.AvgLenEndcap  = matchedEndcap.empty() ? -999.f : (float)(sumLenEndcap / matchedEndcap.size());

    std::vector<std::pair<double,double>> allMatched;
    allMatched.insert(allMatched.end(), matchedBarrel.begin(), matchedBarrel.end());
    allMatched.insert(allMatched.end(), matchedEndcap.begin(), matchedEndcap.end());

    if (!allMatched.empty()) {
        BetaEstimate be = CombineBeta(allMatched);
        if (be.valid) {
            double p = Partic.P();
            double msq = p*p * (1.0/(be.beta*be.beta) - 1.0);
            f.Beta   = (float)be.beta;
            f.MassSq = (float)msq;
            f.HasToF = 1.f;
        }
    }

    return f;
}

// =====================================================================
// ============================  LOW-P PATH  ============================
// Now uses the FULL calo shower-shape feature set (via the shared
// HighP_CollectHits / HighP_ComputeTrackFeatures above) for both ECal and
// HCal, plus ToF. 41-feature vector -> session_LowP, in exactly the
// RAW_COLS order used by CombinedCaloToFAnalysis.cxx / TrainMuonID.py:
//
//   ECalEnergy, ECalNumber, ECalEoverP, ECalAvgHitEnergy,
//   ECalSpreadPhi, ECalSpreadEta, ECalSpreadR, ECalMaxHitFrac,
//   ECalEnergyStdDev, ECalEnergyConcentration,
//   ECalR_Disp, ECalR_DispWeighted, ECalEta_DispWeighted, ECalPhi_DispWeighted,
//   HCalEnergy, HCalNumber, HCalEoverP, HCalAvgHitEnergy,
//   HCalSpreadPhi, HCalSpreadEta, HCalSpreadR, HCalMaxHitFrac,
//   HCalEnergyStdDev, HCalEnergyConcentration,
//   HCalR_Disp, HCalR_DispWeighted, HCalEta_DispWeighted, HCalPhi_DispWeighted,
//   ToFBeta, ToFMassSq, ToFNHitsBarrel, ToFNHitsEndcap, ToFNHitsTotal,
//   ToFMinDistBarrel, ToFMinDistEndcap, ToFAvgLenBarrel, ToFAvgLenEndcap, ToFHasToF,
//   TrackMomentum, TrackEta, TrackPhi
//
// Sentinel convention (must match the C++ training macro / Python exactly):
//   Energy/Number are always real (0 when the track left no hits).
//   Every OTHER calo shower-shape field is set to MISSING_SENTINEL when
//   Number<=0 for that subsystem, since e.g. SpreadPhi=0 would otherwise
//   look like a real, tiny spread rather than "undefined".
//   The ONNX preprocessing node imputes these sentinels internally, so we
//   must NOT pre-impute them here -- just pass the raw sentinel through.
// =====================================================================

std::vector<float> LowP_build_raw_features(
    float ECalEnergy, float ECalNumber, float ECalEoverP, float ECalAvgHitEnergy,
    float ECalSpreadPhi, float ECalSpreadEta, float ECalSpreadR, float ECalMaxHitFrac,
    float ECalEnergyStdDev, float ECalEnergyConcentration,
    float ECalR_Disp, float ECalR_DispWeighted, float ECalEta_DispWeighted, float ECalPhi_DispWeighted,
    float HCalEnergy, float HCalNumber, float HCalEoverP, float HCalAvgHitEnergy,
    float HCalSpreadPhi, float HCalSpreadEta, float HCalSpreadR, float HCalMaxHitFrac,
    float HCalEnergyStdDev, float HCalEnergyConcentration,
    float HCalR_Disp, float HCalR_DispWeighted, float HCalEta_DispWeighted, float HCalPhi_DispWeighted,
    float ToFBeta, float ToFMassSq, float ToFNHitsBarrel, float ToFNHitsEndcap, float ToFNHitsTotal,
    float ToFMinDistBarrel, float ToFMinDistEndcap, float ToFAvgLenBarrel, float ToFAvgLenEndcap, float ToFHasToF,
    float TrackMomentum, float TrackEta, float TrackPhi)
{
    return {
        ECalEnergy, ECalNumber, ECalEoverP, ECalAvgHitEnergy,
        ECalSpreadPhi, ECalSpreadEta, ECalSpreadR, ECalMaxHitFrac,
        ECalEnergyStdDev, ECalEnergyConcentration,
        ECalR_Disp, ECalR_DispWeighted, ECalEta_DispWeighted, ECalPhi_DispWeighted,

        HCalEnergy, HCalNumber, HCalEoverP, HCalAvgHitEnergy,
        HCalSpreadPhi, HCalSpreadEta, HCalSpreadR, HCalMaxHitFrac,
        HCalEnergyStdDev, HCalEnergyConcentration,
        HCalR_Disp, HCalR_DispWeighted, HCalEta_DispWeighted, HCalPhi_DispWeighted,

        ToFBeta, ToFMassSq, ToFNHitsBarrel, ToFNHitsEndcap, ToFNHitsTotal,
        ToFMinDistBarrel, ToFMinDistEndcap, ToFAvgLenBarrel, ToFAvgLenEndcap, ToFHasToF,

        TrackMomentum, TrackEta, TrackPhi
    };
}

// Applies the training-time sentinel convention to a full calo feature set:
// Energy/Number stay real; every other field becomes MISSING_SENTINEL when
// the subsystem had no hits (Number<=0).
struct LowP_CaloRaw
{
    float Energy, Number, EoverP, AvgHitEnergy, SpreadPhi, SpreadEta, SpreadR,
          MaxHitFrac, EnergyStdDev, EnergyConcentration, R_Disp, R_DispWeighted,
          Eta_DispWeighted, Phi_DispWeighted;
};

LowP_CaloRaw LowP_ApplySentinel(const HighP_TrackFeatures &f)
{
    bool hasHits = f.Number > 0;
    LowP_CaloRaw r;
    r.Energy = f.Energy;
    r.Number = f.Number;
    r.EoverP               = hasHits ? f.EoverP : MISSING_SENTINEL;
    r.AvgHitEnergy          = hasHits ? f.AvgHitEnergy : MISSING_SENTINEL;
    r.SpreadPhi             = hasHits ? f.SpreadPhi : MISSING_SENTINEL;
    r.SpreadEta             = hasHits ? f.SpreadEta : MISSING_SENTINEL;
    r.SpreadR               = hasHits ? f.SpreadR : MISSING_SENTINEL;
    r.MaxHitFrac            = hasHits ? f.MaxHitFrac : MISSING_SENTINEL;
    r.EnergyStdDev          = hasHits ? f.EnergyStdDev : MISSING_SENTINEL;
    r.EnergyConcentration   = hasHits ? f.EnergyConcentration : MISSING_SENTINEL;
    r.R_Disp                = hasHits ? f.R_Disp : MISSING_SENTINEL;
    r.R_DispWeighted         = hasHits ? f.R_DispWeighted : MISSING_SENTINEL;
    r.Eta_DispWeighted       = hasHits ? f.Eta_DispWeighted : MISSING_SENTINEL;
    r.Phi_DispWeighted       = hasHits ? f.Phi_DispWeighted : MISSING_SENTINEL;
    return r;
}

// =====================================================================
// ================================ MAIN ================================
// =====================================================================
void TestingMuonID()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    // ---- ONNX Runtime: two independent sessions ----
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID_Combined");

    Ort::SessionOptions opts_HighP;
    Ort::Session session_HighP(env, HIGHP_ONNX_PATH, opts_HighP);

    Ort::SessionOptions opts_LowP;
    opts_LowP.SetIntraOpNumThreads(1);
    opts_LowP.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session_LowP(env, LOWP_ONNX_PATH, opts_LowP);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    static constexpr int NumOfFiles = 2;
    vector<TString> filePatterns(NumOfFiles);
    filePatterns.at(0) = "/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
    //filePatterns.at(0) = "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_1*.root";
    filePatterns.at(1) = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";

    vector<string> ecalClusterNames = {"EcalBarrelImagingClusters", "EcalBarrelScFiClusters", "EcalEndcapPClusters", "EcalEndcapNClusters"};
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    TFile *outfile = new TFile("Plots/TestingMuonID.root", "RECREATE");

    // ---- Histograms (same layout/binning as TestingMacro_Podio.cxx, wide
    //      enough to cover both the LowP and HighP regimes) ----


    // =====================================================================
    // ---- NEW: Low-p / High-p split histograms -----------------------
    //
    //   * P(muon) response separately for Low-p (p < P_SPLIT_GEV) and High-p
    //     (p >= P_SPLIT_GEV), for both muons and pions.
    //   * Muon efficiency eps_mu(p) and pion rejection eps_pi_rej(p)
    //     separately for both momentum regimes, with binning matched to
    //     each range (Low-p: fine bins below P_SPLIT_GEV; High-p: bins
    //     above P_SPLIT_GEV, up to PHIGH_MAX_GEV).
    //   * For Low-p, additionally calculate muon efficiency split into
    //     events with a ToF signal (tofF.HasToF>0.5) and without one.
    // =====================================================================

    // Momentum ranges and binning for the Low-p / High-p histograms.
    static const int    NBINS_LOWP   = 40;                 // Low-p: 0 - P_SPLIT_GEV
    static const int    NBINS_HIGHP  = 38;                 // High-p: P_SPLIT_GEV - PHIGH_MAX_GEV
    static const double PHIGH_MAX_GEV = 20.0;               // upper range of the High-p histograms

    // -- P(muon) response, separately for Low-p / High-p --
    TH1F *h_Response_Muon_LowP  = new TH1F("h_Response_Muon_LowP",
        "P(muon) Response for True Muons, p < 1 GeV/c;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Muon_HighP = new TH1F("h_Response_Muon_HighP",
        "P(muon) Response for True Muons, p #geq 1 GeV/c;P(muon);Counts", 100, 0, 1);

    TH1F *h_Response_Pion_LowP  = new TH1F("h_Response_Pion_LowP",
        "P(muon) Response for True Pions, p < 1 GeV/c;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Pion_HighP = new TH1F("h_Response_Pion_HighP",
        "P(muon) Response for True Pions, p #geq 1 GeV/c;P(muon);Counts", 100, 0, 1);

    TH1F *h_Response_Muon = new TH1F("h_Response_Muon", "P(muon) Response for True Muons;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Pion = new TH1F("h_Response_Pion", "P(muon) Response for True Pions;P(muon);Counts", 100, 0, 1);

    TH1F *h_Muon_Total_vs_Pt = new TH1F("h_Muon_Total_vs_Pt", "Muon Total vs pT;p_{T} [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);
    TH1F *h_Muon_Passed_vs_Pt = new TH1F("h_Muon_Passed_vs_Pt", "Muon Passed vs pT;p_{T} [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);

    TH1F *h_Muon_Total_vs_P  = new TH1F("h_Muon_Total_vs_P", "Muon Total vs p;p [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);
    TH1F *h_Muon_Passed_vs_P = new TH1F("h_Muon_Passed_vs_P", "Muon Passed vs p;p [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);

    TH1F *h_Muon_Total_vs_Eta  = new TH1F("h_Muon_Total_vs_Eta", "Muon Total vs #eta;#eta;Counts", 50, -1.2, 3.4);
    TH1F *h_Muon_Passed_vs_Eta = new TH1F("h_Muon_Passed_vs_Eta", "Muon Passed vs #eta;#eta;Counts", 50, -1.2, 3.4);

    TH1F *h_Pion_Total_vs_Pt   = new TH1F("h_Pion_Total_vs_Pt", "Pion Total vs pT;p_{T} [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);
    TH1F *h_Pion_Rejected_vs_Pt = new TH1F("h_Pion_Rejected_vs_Pt", "Pion Rejected vs pT;p_{T} [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);

    TH1F *h_Pion_Total_vs_P    = new TH1F("h_Pion_Total_vs_P", "Pion Total vs p;p [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);
    TH1F *h_Pion_Rejected_vs_P = new TH1F("h_Pion_Rejected_vs_P", "Pion Rejected vs p;p [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);

    TH1F *h_Pion_Total_vs_Eta  = new TH1F("h_Pion_Total_vs_Eta", "Pion Total vs #eta;#eta;Counts", 50, -1.2, 3.4);
    TH1F *h_Pion_Rejected_vs_Eta = new TH1F("h_Pion_Rejected_vs_Eta", "Pion Rejected vs #eta;#eta;Counts", 50, -1.2, 3.4);

    // -- Muon efficiency eps_mu(p), separately for Low-p / High-p --
    TH1F *h_Muon_Total_vs_P_LowP  = new TH1F("h_Muon_Total_vs_P_LowP",
        "Muon Total vs p (Low-p);p [GeV/c];Counts", NBINS_LOWP, 0, P_SPLIT_GEV);
    TH1F *h_Muon_Passed_vs_P_LowP = new TH1F("h_Muon_Passed_vs_P_LowP",
        "Muon Passed vs p (Low-p);p [GeV/c];Counts", NBINS_LOWP, 0, P_SPLIT_GEV);

    TH1F *h_Muon_Total_vs_P_HighP  = new TH1F("h_Muon_Total_vs_P_HighP",
        "Muon Total vs p (High-p);p [GeV/c];Counts", NBINS_HIGHP, P_SPLIT_GEV, PHIGH_MAX_GEV);
    TH1F *h_Muon_Passed_vs_P_HighP = new TH1F("h_Muon_Passed_vs_P_HighP",
        "Muon Passed vs p (High-p);p [GeV/c];Counts", NBINS_HIGHP, P_SPLIT_GEV, PHIGH_MAX_GEV);

    // -- Pion rejection eps_pi_rej(p), separately for Low-p / High-p --
    TH1F *h_Pion_Total_vs_P_LowP    = new TH1F("h_Pion_Total_vs_P_LowP",
        "Pion Total vs p (Low-p);p [GeV/c];Counts", NBINS_LOWP, 0, P_SPLIT_GEV);
    TH1F *h_Pion_Rejected_vs_P_LowP = new TH1F("h_Pion_Rejected_vs_P_LowP",
        "Pion Rejected vs p (Low-p);p [GeV/c];Counts", NBINS_LOWP, 0, P_SPLIT_GEV);

    TH1F *h_Pion_Total_vs_P_HighP    = new TH1F("h_Pion_Total_vs_P_HighP",
        "Pion Total vs p (High-p);p [GeV/c];Counts", NBINS_HIGHP, P_SPLIT_GEV, PHIGH_MAX_GEV);
    TH1F *h_Pion_Rejected_vs_P_HighP = new TH1F("h_Pion_Rejected_vs_P_HighP",
        "Pion Rejected vs p (High-p);p [GeV/c];Counts", NBINS_HIGHP, P_SPLIT_GEV, PHIGH_MAX_GEV);

    // -- Muon efficiency for Low-p, with / without a ToF signal --
    TH1F *h_Muon_Total_vs_P_LowP_withToF  = new TH1F("h_Muon_Total_vs_P_LowP_withToF",
        "Muon Total vs p (Low-p, with ToF hit);p [GeV/c];Counts", NBINS_LOWP, 0, P_SPLIT_GEV);
    TH1F *h_Muon_Passed_vs_P_LowP_withToF = new TH1F("h_Muon_Passed_vs_P_LowP_withToF",
        "Muon Passed vs p (Low-p, with ToF hit);p [GeV/c];Counts", NBINS_LOWP, 0, P_SPLIT_GEV);

    TH1F *h_Muon_Total_vs_P_LowP_noToF  = new TH1F("h_Muon_Total_vs_P_LowP_noToF",
        "Muon Total vs p (Low-p, no ToF hit);p [GeV/c];Counts", NBINS_LOWP, 0, P_SPLIT_GEV);
    TH1F *h_Muon_Passed_vs_P_LowP_noToF = new TH1F("h_Muon_Passed_vs_P_LowP_noToF",
        "Muon Passed vs p (Low-p, no ToF hit);p [GeV/c];Counts", NBINS_LOWP, 0, P_SPLIT_GEV);

    for (int File = 0; File < NumOfFiles; File++)
    {
        bool isMuonFile = (File == 0);
        string name = isMuonFile ? "Muons" : "Pions";

        vector<string> fileList = ExpandGlob(string(filePatterns.at(File)));

        podio::ROOTReader reader;
        reader.openFiles(fileList);

        unsigned nEvents = reader.getEntries("events");
        cout << "Total events in chain (" << name << "): " << nEvents << endl;

        vector<DetectorAssoc> ecalDetectors;
        for (auto &n : ecalClusterNames) ecalDetectors.push_back(MakeDetectorAssoc(n));

        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

        int eventID = 0;

        for (unsigned entry = 0; entry < nEvents; ++entry)
        {
            eventID++;
            if (eventID > NUMBEROFEVENTS) break;
            if (eventID % 50000 == 0) cout << "File " << name << " event: " << eventID << endl;

            podio::Frame frame(reader.readEntry("events", entry));

            const auto &tracks =
                frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedChargedParticles");
            const auto &trackAssocs =
                frame.get<edm4eic::MCRecoParticleAssociationCollection>("ReconstructedChargedParticleAssociations");

            // ToF hit collections - only used by the LowP branch, but cheap
            // to fetch once per event regardless.
            const auto &tofBarrelHits =
                frame.get<edm4eic::TrackerHitCollection>("TOFBarrelRecHits");
            const auto &tofEndcapHits =
                frame.get<edm4eic::TrackerHitCollection>("TOFEndcapRecHits");

            for (size_t particle = 0; particle < tracks.size(); ++particle)
            {
                const auto rcp = tracks[particle];

                auto mom = rcp.getMomentum();
                TLorentzVector Partic;
                Partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
                double trackP   = Partic.P();
                double trackPt  = Partic.Pt();
                double trackEta = Partic.Eta();
                double trackPhi = Partic.Phi();

                if (trackEta >= 1 && trackEta <= 1.3) continue;
                if (trackEta <= -1.0) continue;

                if (particle >= trackAssocs.size()) continue;
                const auto simPart = trackAssocs[particle].getSim();

                float Pmu = MISSING_SENTINEL;
                bool keep = true;
                bool isLowP = (trackP < P_SPLIT_GEV);
                // Filled only on the Low-p path, for the
                // "with ToF / without ToF" histograms.
                bool lowP_hasToF = false;

                if (isLowP)
                {
                    // ---------------- LOW-P PATH ----------------
                    // Full calo shower-shape features (same computation as
                    // HighP) for both ECal and HCal, plus ToF.
                    int charge = static_cast<int>(rcp.getCharge());

                    HighP_TrackCalHits ecalHits = HighP_CollectHits(simPart, ecalDetectors, frame, Partic, TIMING_CUT_NS);
                    HighP_TrackFeatures ecalF = HighP_ComputeTrackFeatures(ecalHits, trackP);

                    HighP_TrackCalHits hcalHits = HighP_CollectHits(simPart, hcalDetectors, frame, Partic, TIMING_CUT_NS);
                    HighP_TrackFeatures hcalF = HighP_ComputeTrackFeatures(hcalHits, trackP);

                    LowP_TrackToFFeatures tofF = LowP_ComputeToFFeatures(
                        Partic, charge,
                        tofBarrelHits, tofEndcapHits,
                        /*dR_cut_barrel*/0.8, /*dR_cut_endcap*/0.8,
                        /*dist_cut_barrel*/6.0, /*dist_cut_endcap*/6.0
                    );

                    lowP_hasToF = (tofF.HasToF > 0.5f);

                    if (ecalF.Number <= 0 && hcalF.Number <= 0 && tofF.HasToF < 0.5) { keep = false; }

                    if (keep)
                    {
                        LowP_CaloRaw ecalR = LowP_ApplySentinel(ecalF);
                        LowP_CaloRaw hcalR = LowP_ApplySentinel(hcalF);

                        std::vector<float> raw = LowP_build_raw_features(
                            ecalR.Energy, ecalR.Number, ecalR.EoverP, ecalR.AvgHitEnergy,
                            ecalR.SpreadPhi, ecalR.SpreadEta, ecalR.SpreadR, ecalR.MaxHitFrac,
                            ecalR.EnergyStdDev, ecalR.EnergyConcentration,
                            ecalR.R_Disp, ecalR.R_DispWeighted, ecalR.Eta_DispWeighted, ecalR.Phi_DispWeighted,

                            hcalR.Energy, hcalR.Number, hcalR.EoverP, hcalR.AvgHitEnergy,
                            hcalR.SpreadPhi, hcalR.SpreadEta, hcalR.SpreadR, hcalR.MaxHitFrac,
                            hcalR.EnergyStdDev, hcalR.EnergyConcentration,
                            hcalR.R_Disp, hcalR.R_DispWeighted, hcalR.Eta_DispWeighted, hcalR.Phi_DispWeighted,

                            tofF.Beta, tofF.MassSq, tofF.NHitsBarrel, tofF.NHitsEndcap, tofF.NHitsTotal,
                            tofF.MinDistBarrel, tofF.MinDistEndcap, tofF.AvgLenBarrel, tofF.AvgLenEndcap, tofF.HasToF,

                            static_cast<float>(trackP), static_cast<float>(trackEta), static_cast<float>(trackPhi)
                        );

                        Pmu = run_muon_id(session_LowP, memory_info, raw);
                    }
                }
                else
                {
                    // ---------------- HIGH-P PATH ---------------- (unchanged)
                    HighP_TrackCalHits ecalHits = HighP_CollectHits(simPart, ecalDetectors, frame, Partic, TIMING_CUT_NS);
                    HighP_TrackFeatures ecalF = HighP_ComputeTrackFeatures(ecalHits, trackP);

                    HighP_TrackCalHits hcalHits = HighP_CollectHits(simPart, hcalDetectors, frame, Partic, TIMING_CUT_NS);
                    HighP_TrackFeatures hcalF = HighP_ComputeTrackFeatures(hcalHits, trackP);

                    if (ecalF.Number <= 0 && hcalF.Number <= 0) { keep = false; }

                    if (keep)
                    {
                        std::vector<float> raw = HighP_build_raw_features(
                            ecalF.Energy, hcalF.Energy,
                            ecalF.Number, hcalF.Number,
                            ecalF.EoverP, hcalF.EoverP,
                            ecalF.AvgHitEnergy, hcalF.AvgHitEnergy,
                            ecalF.SpreadPhi, ecalF.SpreadEta, ecalF.SpreadR,
                            hcalF.SpreadPhi, hcalF.SpreadEta, hcalF.SpreadR,
                            ecalF.MaxHitFrac, hcalF.MaxHitFrac,
                            ecalF.EnergyStdDev, hcalF.EnergyStdDev,
                            ecalF.EnergyConcentration, hcalF.EnergyConcentration,
                            ecalF.R_Disp, ecalF.R_DispWeighted,
                            ecalF.Eta_DispWeighted, ecalF.Phi_DispWeighted,
                            hcalF.R_Disp, hcalF.R_DispWeighted,
                            hcalF.Eta_DispWeighted, hcalF.Phi_DispWeighted,
                            static_cast<float>(trackP), static_cast<float>(trackEta)
                        );

                        Pmu = run_muon_id(session_HighP, memory_info, raw);
                    }
                }

                if (!keep) continue;

                // Use the model-appropriate working point depending on
                // which branch this track went through.
                float cut = isLowP ? MUON_ID_CUT_LOWP : MUON_ID_CUT_HIGHP;
                bool passed = (Pmu > cut);

                if (isMuonFile)
                {
                    h_Response_Muon->Fill(Pmu);

                    h_Muon_Total_vs_Pt->Fill(trackPt);
                    h_Muon_Total_vs_P->Fill(trackP);
                    h_Muon_Total_vs_Eta->Fill(trackEta);

                    if (passed)
                    {
                        h_Muon_Passed_vs_Pt->Fill(trackPt);
                        h_Muon_Passed_vs_P->Fill(trackP);
                        h_Muon_Passed_vs_Eta->Fill(trackEta);
                    }

                    // ---- NEW: Low-p / High-p split fills ----
                    if (isLowP)
                    {
                        h_Response_Muon_LowP->Fill(Pmu);
                        h_Muon_Total_vs_P_LowP->Fill(trackP);
                        if (passed) h_Muon_Passed_vs_P_LowP->Fill(trackP);

                        if (lowP_hasToF)
                        {
                            h_Muon_Total_vs_P_LowP_withToF->Fill(trackP);
                            if (passed) h_Muon_Passed_vs_P_LowP_withToF->Fill(trackP);
                        }
                        else
                        {
                            h_Muon_Total_vs_P_LowP_noToF->Fill(trackP);
                            if (passed) h_Muon_Passed_vs_P_LowP_noToF->Fill(trackP);
                        }
                    }
                    else
                    {
                        h_Response_Muon_HighP->Fill(Pmu);
                        h_Muon_Total_vs_P_HighP->Fill(trackP);
                        if (passed) h_Muon_Passed_vs_P_HighP->Fill(trackP);
                    }
                }
                else // Pion file
                {
                    h_Response_Pion->Fill(Pmu);

                    h_Pion_Total_vs_Pt->Fill(trackPt);
                    h_Pion_Total_vs_P->Fill(trackP);
                    h_Pion_Total_vs_Eta->Fill(trackEta);

                    bool rejected = !passed; // Pmu <= cut -> correctly rejected
                    if (rejected) // Pion correctly rejected
                    {
                        h_Pion_Rejected_vs_Pt->Fill(trackPt);
                        h_Pion_Rejected_vs_P->Fill(trackP);
                        h_Pion_Rejected_vs_Eta->Fill(trackEta);
                    }

                    // ---- NEW: Low-p / High-p split fills ----
                    if (isLowP)
                    {
                        h_Response_Pion_LowP->Fill(Pmu);
                        h_Pion_Total_vs_P_LowP->Fill(trackP);
                        if (rejected) h_Pion_Rejected_vs_P_LowP->Fill(trackP);
                    }
                    else
                    {
                        h_Response_Pion_HighP->Fill(Pmu);
                        h_Pion_Total_vs_P_HighP->Fill(trackP);
                        if (rejected) h_Pion_Rejected_vs_P_HighP->Fill(trackP);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Final efficiency / rejection histograms (ratios) - unchanged
    // -----------------------------------------------------------------
    TH1F *h_Muon_Efficiency_vs_Pt  = (TH1F*)h_Muon_Passed_vs_Pt->Clone("h_Muon_Efficiency_vs_Pt");
    h_Muon_Efficiency_vs_Pt->SetTitle("Muon Efficiency vs p_{T};p_{T} [GeV/c];Efficiency");
    h_Muon_Efficiency_vs_Pt->Divide(h_Muon_Passed_vs_Pt, h_Muon_Total_vs_Pt, 1.0, 1.0, "B");

    TH1F *h_Muon_Efficiency_vs_P   = (TH1F*)h_Muon_Passed_vs_P->Clone("h_Muon_Efficiency_vs_P");
    h_Muon_Efficiency_vs_P->SetTitle("Muon Efficiency vs p;p [GeV/c];Efficiency");
    h_Muon_Efficiency_vs_P->Divide(h_Muon_Passed_vs_P, h_Muon_Total_vs_P, 1.0, 1.0, "B");

    TH1F *h_Muon_Efficiency_vs_Eta = (TH1F*)h_Muon_Passed_vs_Eta->Clone("h_Muon_Efficiency_vs_Eta");
    h_Muon_Efficiency_vs_Eta->SetTitle("Muon Efficiency vs #eta;#eta;Efficiency");
    h_Muon_Efficiency_vs_Eta->Divide(h_Muon_Passed_vs_Eta, h_Muon_Total_vs_Eta, 1.0, 1.0, "B");

    TH1F *h_Pion_Rejection_vs_Pt   = (TH1F*)h_Pion_Rejected_vs_Pt->Clone("h_Pion_Rejection_vs_Pt");
    h_Pion_Rejection_vs_Pt->SetTitle("Pion Rejection vs p_{T};p_{T} [GeV/c];Rejection Fraction");
    h_Pion_Rejection_vs_Pt->Divide(h_Pion_Rejected_vs_Pt, h_Pion_Total_vs_Pt, 1.0, 1.0, "B");

    TH1F *h_Pion_Rejection_vs_P    = (TH1F*)h_Pion_Rejected_vs_P->Clone("h_Pion_Rejection_vs_P");
    h_Pion_Rejection_vs_P->SetTitle("Pion Rejection vs p;p [GeV/c];Rejection Fraction");
    h_Pion_Rejection_vs_P->Divide(h_Pion_Rejected_vs_P, h_Pion_Total_vs_P, 1.0, 1.0, "B");

    TH1F *h_Pion_Rejection_vs_Eta  = (TH1F*)h_Pion_Rejected_vs_Eta->Clone("h_Pion_Rejection_vs_Eta");
    h_Pion_Rejection_vs_Eta->SetTitle("Pion Rejection vs #eta;#eta;Rejection Fraction");
    h_Pion_Rejection_vs_Eta->Divide(h_Pion_Rejected_vs_Eta, h_Pion_Total_vs_Eta, 1.0, 1.0, "B");

    // -----------------------------------------------------------------
    // ---- NEW: eps_mu(p) and eps_pi_rej(p), separately for Low-p / High-p ------
    // -----------------------------------------------------------------
    TH1F *h_Muon_Efficiency_vs_P_LowP = (TH1F*)h_Muon_Passed_vs_P_LowP->Clone("h_Muon_Efficiency_vs_P_LowP");
    h_Muon_Efficiency_vs_P_LowP->SetTitle("Muon Efficiency vs p, Low-p (p < 1 GeV/c) #varepsilon_{#mu}(p);p [GeV/c];Efficiency");
    h_Muon_Efficiency_vs_P_LowP->Divide(h_Muon_Passed_vs_P_LowP, h_Muon_Total_vs_P_LowP, 1.0, 1.0, "B");

    TH1F *h_Muon_Efficiency_vs_P_HighP = (TH1F*)h_Muon_Passed_vs_P_HighP->Clone("h_Muon_Efficiency_vs_P_HighP");
    h_Muon_Efficiency_vs_P_HighP->SetTitle("Muon Efficiency vs p, High-p (p #geq 1 GeV/c) #varepsilon_{#mu}(p);p [GeV/c];Efficiency");
    h_Muon_Efficiency_vs_P_HighP->Divide(h_Muon_Passed_vs_P_HighP, h_Muon_Total_vs_P_HighP, 1.0, 1.0, "B");

    TH1F *h_Pion_Rejection_vs_P_LowP = (TH1F*)h_Pion_Rejected_vs_P_LowP->Clone("h_Pion_Rejection_vs_P_LowP");
    h_Pion_Rejection_vs_P_LowP->SetTitle("Pion Rejection vs p, Low-p (p < 1 GeV/c) #varepsilon_{#pi}^{rej}(p);p [GeV/c];Rejection Fraction");
    h_Pion_Rejection_vs_P_LowP->Divide(h_Pion_Rejected_vs_P_LowP, h_Pion_Total_vs_P_LowP, 1.0, 1.0, "B");

    TH1F *h_Pion_Rejection_vs_P_HighP = (TH1F*)h_Pion_Rejected_vs_P_HighP->Clone("h_Pion_Rejection_vs_P_HighP");
    h_Pion_Rejection_vs_P_HighP->SetTitle("Pion Rejection vs p, High-p (p #geq 1 GeV/c) #varepsilon_{#pi}^{rej}(p);p [GeV/c];Rejection Fraction");
    h_Pion_Rejection_vs_P_HighP->Divide(h_Pion_Rejected_vs_P_HighP, h_Pion_Total_vs_P_HighP, 1.0, 1.0, "B");

    // -----------------------------------------------------------------
    // ---- NEW: Low-p muon efficiency with / without a ToF signal -------------
    // -----------------------------------------------------------------
    TH1F *h_Muon_Efficiency_vs_P_LowP_withToF =
        (TH1F*)h_Muon_Passed_vs_P_LowP_withToF->Clone("h_Muon_Efficiency_vs_P_LowP_withToF");
    h_Muon_Efficiency_vs_P_LowP_withToF->SetTitle(
        "Muon Efficiency vs p, Low-p, with ToF signal;p [GeV/c];Efficiency");
    h_Muon_Efficiency_vs_P_LowP_withToF->Divide(
        h_Muon_Passed_vs_P_LowP_withToF, h_Muon_Total_vs_P_LowP_withToF, 1.0, 1.0, "B");

    TH1F *h_Muon_Efficiency_vs_P_LowP_noToF =
        (TH1F*)h_Muon_Passed_vs_P_LowP_noToF->Clone("h_Muon_Efficiency_vs_P_LowP_noToF");
    h_Muon_Efficiency_vs_P_LowP_noToF->SetTitle(
        "Muon Efficiency vs p, Low-p, without ToF signal;p [GeV/c];Efficiency");
    h_Muon_Efficiency_vs_P_LowP_noToF->Divide(
        h_Muon_Passed_vs_P_LowP_noToF, h_Muon_Total_vs_P_LowP_noToF, 1.0, 1.0, "B");

    outfile->Write();
    outfile->Close();

    cout << "Efficiency and rejection histograms have been saved to Plots/MuonID_Performance_Combined.root" << endl;
}

#ifndef __CLING__
int main(int argc, char** argv)
{
    TestingMuonID();
    return 0;
}
#endif