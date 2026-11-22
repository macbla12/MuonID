// TestingMacro_Combined_Podio_TCM.cxx
//
// Combined podio/EDM4eic testing macro (HighP calo-only model + LowP calo+ToF model),
// po zmianach takich samych jak w TrainingMacroCaloToF_Podio_TCM.cxx / TestingMacroCaloToF_Podio_TCM.cxx:
//
// 1) Zamiast asocjacji MC (MCRecoClusterParticleAssociation) uzywamy TCM, dla OBU sciezek (HighP i LowP):
//      - kolekcja czastek: "ReconstructedParticles"
//      - ECal: rcp.getClusters()
//      - HCal: klastry z <Det>TrackClusterMatches dla rcp.getTracks().at(0)
//    Hity z cluster.getHits(). Zadna informacja MC nie jest uzywana.
//
// 2) (LowP) Dlugosc toru ToF z CentralTrackSegments (pt.pathlength), jak w TrackLength.cxx:
//    segment dopasowany po seg.getTrack() == rcp.getTracks().at(0), punkty w oknach R/z ToF,
//    dla kazdego hitu ToF najblizszy punkt segmentu. ToFSim.cxx NIE jest juz potrzebny.
//
// ROUTING
//   trackP >= P_SPLIT_GEV -> HighP: 29 cech, session_HighP
//   trackP <  P_SPLIT_GEV -> LowP : 41 cech (RAW_COLS), session_LowP
//
// Uruchom: root -l -q TestingMacro_Combined_Podio_TCM.cxx++

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

#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/TrackClusterMatchCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <edm4eic/TrackerHitCollection.h>
#include <edm4eic/TrackSegmentCollection.h>

using namespace std;

// Predkosc swiatla w mm/ns (wczesniej z ToFSim.cxx). Czasy hitow ToF w ns, dlugosci w mm.
static constexpr double c_light = 299.792458;

// =====================================================================
// Okna geometryczne ToF [mm] (jak w TrackLength.cxx) - musza byc IDENTYCZNE jak w treningu LowP!
// =====================================================================
const double BTOF_RMIN = 600,  BTOF_RMAX = 720,  BTOF_ZMAX = 1200;                 // barrel
const double FTOF_ZMIN = 1750, FTOF_ZMAX = 1950, FTOF_RMIN = 100, FTOF_RMAX = 650; // endcap (|z|)

static bool InBarrelTOF(double x, double y, double z)
{
    const double R = std::hypot(x, y), az = std::fabs(z);
    return (R >= BTOF_RMIN && R <= BTOF_RMAX && az <= BTOF_ZMAX);
}

static bool InEndcapTOF(double x, double y, double z)
{
    const double R = std::hypot(x, y), az = std::fabs(z);
    return (az >= FTOF_ZMIN && az <= FTOF_ZMAX && R >= FTOF_RMIN && R <= FTOF_RMAX);
}

// =====================================================================
// Momentum split point between the two models.
// =====================================================================
static const double P_SPLIT_GEV = 1.0;

static const char *LOWP_ONNX_PATH  = "../ToF/ONNX/xgb_muonID.onnx";
static const char *HIGHP_ONNX_PATH = "../CalorimetryHits/ONNX/xgb_muonID.onnx";

static const float MUON_ID_CUT_HIGHP = 0.5f;
static const float MUON_ID_CUT_LOWP  = 0.5f;   // placeholder, retune for your own model

static const double TIMING_CUT_NS = 20.0;

static const float MISSING_SENTINEL = -999.f;

static const int NUMBEROFEVENTS = 200000;

static const string SEG_COLL = "CentralTrackSegments";

// =====================================================================
// glob (zastepuje TChain::Add(pattern))
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

bool HasCollection(const vector<string> &names, const string &n)
{
    return find(names.begin(), names.end(), n) != names.end();
}

// =====================================================================
// HCal: "XClusters" -> "XTrackClusterMatches" (edm4eic::TrackClusterMatch).
// ECal uzywa rcp.getClusters(), wiec nie potrzebuje kolekcji match.
// =====================================================================
struct DetectorAssoc
{
    string clusterName;
    string matchCollName;
    bool   available = true;
};

DetectorAssoc MakeDetectorAssoc(const string &clusterName)
{
    DetectorAssoc d;
    d.clusterName   = clusterName;
    d.matchCollName = clusterName.substr(0, clusterName.size() - string("Clusters").size()) + "TrackClusterMatches";
    cout << "  Detector '" << clusterName << "' -> Match collection '" << d.matchCollName << "'" << endl;
    return d;
}

void CheckAvailability(vector<DetectorAssoc> &dets, const podio::Frame &frame)
{
    auto names = frame.getAvailableCollections();
    for (auto &d : dets) {
        d.available = HasCollection(names, d.matchCollName);
        cout << "  " << d.clusterName << " -> '" << d.matchCollName << "' : "
             << (d.available ? "OK" : "MISSING (detektor pominiety)") << endl;
    }
}

// Generic ONNX runner: input "raw_features", output "probabilities", P(muon) = probs[1].
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

// CollectHits - wersja TCM:
//  fromRcpClusters = true  -> ECal: rcp.getClusters()
//  fromRcpClusters = false -> HCal: klastry z TrackClusterMatch dla tracka tej czastki
HighP_TrackCalHits HighP_CollectHits(const edm4eic::ReconstructedParticle &rcp,
                                     const vector<DetectorAssoc> &detectors,
                                     const podio::Frame &frame,
                                     bool fromRcpClusters,
                                     const TLorentzVector &Partic,
                                     double timingCutNs)
{
    HighP_TrackCalHits result;

    vector<edm4eic::Cluster> clusters;

    if (fromRcpClusters)
    {
        for (const auto &c : rcp.getClusters()) clusters.push_back(c);
    }
    else
    {
        if (rcp.getTracks().empty()) return result;
        const auto trk = rcp.getTracks().at(0);

        for (const auto &det : detectors)
        {
            if (!det.available) continue;
            for (const auto &m : frame.get<edm4eic::TrackClusterMatchCollection>(det.matchCollName))
                if (m.getTrack() == trk) clusters.push_back(m.getCluster());
        }
    }

    for (const auto &cluster : clusters)
    {
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
// Tylko sciezka LowP. Dlugosc toru z CentralTrackSegments.
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

// Punkt segmentu tracka w oknie ToF: pozycja [mm] + pathlength [mm]
struct TofPoint { double x, y, z, path; };

static void CollectSegmentTofPoints(const edm4eic::TrackSegmentCollection &segs,
                                    const edm4eic::Track &trk,
                                    vector<TofPoint> &barrelPts,
                                    vector<TofPoint> &endcapPts)
{
    for (const auto seg : segs) {
        if (seg.getTrack() != trk) continue;
        for (const auto pt : seg.getPoints()) {
            const double x = pt.position.x, y = pt.position.y, z = pt.position.z;
            if (InBarrelTOF(x, y, z))      barrelPts.push_back({x, y, z, (double)pt.pathlength});
            else if (InEndcapTOF(x, y, z)) endcapPts.push_back({x, y, z, (double)pt.pathlength});
        }
    }
}

static const TofPoint *NearestPoint(const vector<TofPoint> &pts, const TVector3 &hit, double &dist)
{
    const TofPoint *best = nullptr;
    double bestD = 1e18;
    for (const auto &p : pts) {
        double d = std::sqrt((p.x - hit.X())*(p.x - hit.X()) +
                             (p.y - hit.Y())*(p.y - hit.Y()) +
                             (p.z - hit.Z())*(p.z - hit.Z()));
        if (d < bestD) { bestD = d; best = &p; }
    }
    dist = bestD;
    return best;
}

LowP_TrackToFFeatures LowP_ComputeToFFeatures(
    const TLorentzVector &Partic,
    const edm4eic::Track &trk,
    const edm4eic::TrackSegmentCollection &segments,
    const edm4eic::TrackerHitCollection &barrelHits,
    const edm4eic::TrackerHitCollection &endcapHits,
    double dR_cut_barrel, double dR_cut_endcap,
    double dist_cut_barrel, double dist_cut_endcap)
{
    LowP_TrackToFFeatures f;

    vector<TofPoint> barrelPts, endcapPts;
    CollectSegmentTofPoints(segments, trk, barrelPts, endcapPts);

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

        if (dR < dR_cut_barrel) {
            double dist = 0.0;
            const TofPoint *pt = NearestPoint(barrelPts, ToFPos, dist);
            if (!pt) continue;
            if (dist < dist_cut_barrel) {
                matchedBarrel.push_back({hit.getTime(), pt->path});
                sumLenBarrel += pt->path;
                if (dist < minDistBarrel) minDistBarrel = dist;
            }
        }
    }

    for (const auto &hit : endcapHits) {
        auto pos = hit.getPosition();
        TVector3 ToFPos(pos.x, pos.y, pos.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);

        if (dR < dR_cut_endcap) {
            double dist = 0.0;
            const TofPoint *pt = NearestPoint(endcapPts, ToFPos, dist);
            if (!pt) continue;
            if (dist < dist_cut_endcap) {
                matchedEndcap.push_back({hit.getTime(), pt->path});
                sumLenEndcap += pt->path;
                if (dist < minDistEndcap) minDistEndcap = dist;
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
// 41 cech w kolejnosci RAW_COLS (jak w treningu / TrainMuonID.py).
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

// Energy/Number zostaja prawdziwe; pozostale pola -> MISSING_SENTINEL gdy Number<=0.
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
    r.AvgHitEnergy         = hasHits ? f.AvgHitEnergy : MISSING_SENTINEL;
    r.SpreadPhi            = hasHits ? f.SpreadPhi : MISSING_SENTINEL;
    r.SpreadEta            = hasHits ? f.SpreadEta : MISSING_SENTINEL;
    r.SpreadR              = hasHits ? f.SpreadR : MISSING_SENTINEL;
    r.MaxHitFrac           = hasHits ? f.MaxHitFrac : MISSING_SENTINEL;
    r.EnergyStdDev         = hasHits ? f.EnergyStdDev : MISSING_SENTINEL;
    r.EnergyConcentration  = hasHits ? f.EnergyConcentration : MISSING_SENTINEL;
    r.R_Disp               = hasHits ? f.R_Disp : MISSING_SENTINEL;
    r.R_DispWeighted       = hasHits ? f.R_DispWeighted : MISSING_SENTINEL;
    r.Eta_DispWeighted     = hasHits ? f.Eta_DispWeighted : MISSING_SENTINEL;
    r.Phi_DispWeighted     = hasHits ? f.Phi_DispWeighted : MISSING_SENTINEL;
    return r;
}

// =====================================================================
// ================================ MAIN ================================
// =====================================================================
void TestingMacro_Combined_Podio_TCM()
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

    // ECal: rcp.getClusters(); HCal: TrackClusterMatch
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    TFile *outfile = new TFile("Plots/TestingMuonIDTCM.root", "RECREATE");

    // Momentum ranges and binning for the Low-p / High-p histograms.
    static const int    NBINS_LOWP   = 40;
    static const int    NBINS_HIGHP  = 38;
    static const double PHIGH_MAX_GEV = 20.0;

    // -- P(muon) response, separately for Low-p / High-p --
    TH1F *h_Response_Muon_LowP  = new TH1F("h_Response_Muon_LowP",
        "P(muon) Response for True Muons, p < 1 GeV/c;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Muon_HighP = new TH1F("h_Response_Muon_HighP",
        "P(muon) Response for True Muons, p #geq 1 GeV/c;P(muon);Counts", 100, 0, 1);

    TH1F *h_Response_Pion_LowP  = new TH1F("h_Response_Pion_LowP",
        "P(pion) Response for True Pions, p < 1 GeV/c;P(pion);Counts", 100, 0, 1);
    TH1F *h_Response_Pion_HighP = new TH1F("h_Response_Pion_HighP",
        "P(pion) Response for True Pions, p #geq 1 GeV/c;P(pion);Counts", 100, 0, 1);

    TH1F *h_Response_Muon = new TH1F("h_Response_Muon", "P(muon) Response for True Muons;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Pion = new TH1F("h_Response_Pion", "P(pion) Response for True Pions;P(pion);Counts", 100, 0, 1);

    TH1F *h_Muon_Total_vs_Pt = new TH1F("h_Muon_Total_vs_Pt", "Muon Total vs pT;p_{T} [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);
    TH1F *h_Muon_Passed_vs_Pt = new TH1F("h_Muon_Passed_vs_Pt", "Muon Passed vs pT;p_{T} [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);

    TH1F *h_Muon_Total_vs_P  = new TH1F("h_Muon_Total_vs_P", "Muon Total vs p;p [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);
    TH1F *h_Muon_Passed_vs_P = new TH1F("h_Muon_Passed_vs_P", "Muon Passed vs p;p [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);

    TH1F *h_Muon_Total_vs_Eta  = new TH1F("h_Muon_Total_vs_Eta", "Muon Total vs #eta;#eta;Counts", 50, -3.4, 3.4);
    TH1F *h_Muon_Passed_vs_Eta = new TH1F("h_Muon_Passed_vs_Eta", "Muon Passed vs #eta;#eta;Counts", 50, -3.4, 3.4);

    TH1F *h_Pion_Total_vs_Pt   = new TH1F("h_Pion_Total_vs_Pt", "Pion Total vs pT;p_{T} [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);
    TH1F *h_Pion_Rejected_vs_Pt = new TH1F("h_Pion_Rejected_vs_Pt", "Pion Rejected vs pT;p_{T} [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);

    TH1F *h_Pion_Total_vs_P    = new TH1F("h_Pion_Total_vs_P", "Pion Total vs p;p [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);
    TH1F *h_Pion_Rejected_vs_P = new TH1F("h_Pion_Rejected_vs_P", "Pion Rejected vs p;p [GeV/c];Counts", 40, 0, PHIGH_MAX_GEV);

    TH1F *h_Pion_Total_vs_Eta  = new TH1F("h_Pion_Total_vs_Eta", "Pion Total vs #eta;#eta;Counts", 50, -3.4, 3.4);
    TH1F *h_Pion_Rejected_vs_Eta = new TH1F("h_Pion_Rejected_vs_Eta", "Pion Rejected vs #eta;#eta;Counts", 50, -3.4, 3.4);

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

    // Puste kolekcje uzywane, gdy kolekcji nie ma w pliku.
    static const edm4eic::TrackerHitCollection   emptyToF;
    static const edm4eic::TrackSegmentCollection emptySeg;

    for (int File = 0; File < NumOfFiles; File++)
    {
        bool isMuonFile = (File == 0);
        string name = isMuonFile ? "Muons" : "Pions";

        vector<string> fileList = ExpandGlob(string(filePatterns.at(File)));
        if (fileList.empty()) {
            cerr << "[ERROR] No files match: " << filePatterns.at(File) << endl;
            continue;
        }

        podio::ROOTReader reader;
        reader.openFiles(fileList);

        unsigned nEvents = reader.getEntries("events");
        cout << "Total events in chain (" << name << "): " << nEvents << endl;

        // ECal: rcp.getClusters() -> pusta lista detektorow (nieuzywana)
        vector<DetectorAssoc> ecalDetectors;

        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

        bool hasBToF = true, hasEToF = true, hasSeg = true;

        int eventID = 0;

        for (unsigned entry = 0; entry < nEvents; ++entry)
        {
            eventID++;
            if (eventID > NUMBEROFEVENTS) break;
            if (eventID % 50000 == 0) cout << "File " << name << " event: " << eventID << endl;

            podio::Frame frame(reader.readEntry("events", entry));

            if (entry == 0) {
                cout << "Checking collections in first event:" << endl;
                CheckAvailability(hcalDetectors, frame);
                auto names = frame.getAvailableCollections();
                hasBToF = HasCollection(names, "TOFBarrelRecHits");
                hasEToF = HasCollection(names, "TOFEndcapRecHits");
                hasSeg  = HasCollection(names, SEG_COLL);
                cout << "  TOFBarrelRecHits: " << (hasBToF ? "OK" : "MISSING") << endl;
                cout << "  TOFEndcapRecHits: " << (hasEToF ? "OK" : "MISSING") << endl;
                cout << "  " << SEG_COLL << ": " << (hasSeg ? "OK" : "MISSING (ToF bez dlugosci toru)") << endl;
            }

            const auto &tracks =
                frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");

            // ToF + segmenty - uzywane tylko przez galaz LowP
            const auto &tofBarrelHits = hasBToF ? frame.get<edm4eic::TrackerHitCollection>("TOFBarrelRecHits") : emptyToF;
            const auto &tofEndcapHits = hasEToF ? frame.get<edm4eic::TrackerHitCollection>("TOFEndcapRecHits") : emptyToF;
            const auto &segments      = hasSeg  ? frame.get<edm4eic::TrackSegmentCollection>(SEG_COLL)         : emptySeg;

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

                if (trackP <= 0) continue;

                //if (trackEta >= 1 && trackEta <= 1.3) continue;
                //if (trackEta <= -1.0) continue;

                // track potrzebny dla HCal (TCM) i ToF (CentralTrackSegments)
                if (rcp.getTracks().empty()) continue;
                const auto trk = rcp.getTracks().at(0);

                float Pmu = MISSING_SENTINEL;
                bool keep = true;
                bool isLowP = (trackP < P_SPLIT_GEV);
                bool lowP_hasToF = false;

                if (isLowP)
                {
                    // ---------------- LOW-P PATH ----------------
                    HighP_TrackCalHits ecalHits = HighP_CollectHits(rcp, ecalDetectors, frame, true,  Partic, TIMING_CUT_NS);
                    HighP_TrackFeatures ecalF = HighP_ComputeTrackFeatures(ecalHits, trackP);

                    HighP_TrackCalHits hcalHits = HighP_CollectHits(rcp, hcalDetectors, frame, false, Partic, TIMING_CUT_NS);
                    HighP_TrackFeatures hcalF = HighP_ComputeTrackFeatures(hcalHits, trackP);

                    LowP_TrackToFFeatures tofF = LowP_ComputeToFFeatures(
                        Partic, trk, segments,
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
                    // ---------------- HIGH-P PATH ----------------
                    HighP_TrackCalHits ecalHits = HighP_CollectHits(rcp, ecalDetectors, frame, true,  Partic, TIMING_CUT_NS);
                    HighP_TrackFeatures ecalF = HighP_ComputeTrackFeatures(ecalHits, trackP);

                    HighP_TrackCalHits hcalHits = HighP_CollectHits(rcp, hcalDetectors, frame, false, Partic, TIMING_CUT_NS);
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

                    bool rejected = !passed;
                    if (rejected)
                    {
                        h_Pion_Rejected_vs_Pt->Fill(trackPt);
                        h_Pion_Rejected_vs_P->Fill(trackP);
                        h_Pion_Rejected_vs_Eta->Fill(trackEta);
                    }

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
    // Final efficiency / rejection histograms (ratios)
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

    // eps_mu(p) and eps_pi_rej(p), separately for Low-p / High-p
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

    // Low-p muon efficiency with / without a ToF signal
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

    cout << "Efficiency and rejection histograms have been saved to Plots/TestingMuonID.root" << endl;
}

#ifndef __CLING__
int main(int argc, char** argv)
{
    TestingMacro_Combined_Podio_TCM();
    return 0;
}
#endif