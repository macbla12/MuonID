// TestingMacro_MuonID_LowHigh.cxx
//
// Polaczone makro testowe Muon ID: LowP (Calo + ToF, 41 cech) + HighP (Calo, 30 cech).
//
// - Dla KAZDEGO gatunku (Muon, Pion, Electron, Kaon, Proton) przetwarzane sa:
//     * pliki  reco_*LowP*.root          (Low-owe pliki)
//     * pliki  reco_*.root BEZ "LowP"    (High-owe pliki)
// - Model wybierany PER SLAD na podstawie pedu:
//     p <= P_SWITCH  -> xgb_muonID_LowP.onnx   (cechy Calo + ToF, 41 cech, jak w treningu LowP)
//     p >  P_SWITCH  -> xgb_muonID_HighP.onnx  (cechy Calo, 30 cech, jak w treningu HighP)
//   Zakresy treningowe: Low: 0 < p <= 1.0, High: p > 0.7  =>  P_SWITCH w oknie 0.7-1.0.
// - Histogramy: efficiency / rejection / mis-ID w zakresie 0-20 GeV ORAZ dodatkowo
//   w przyblizeniu 0-2 GeV (histogramy z sufiksem "Zoom").
// - Dodatkowo zapisywane sa gotowe wykresy w pliku Plots/MuonID_Plots.root.
//
// Uruchom: root -l -q TestingMacro_MuonID_LowHigh.cxx++

#include <TH1.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TROOT.h>
#include <TString.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TMath.h>
#include <iostream>
#include <string>
#include <vector>
#include <set>
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

// =============================================================================
// Ustawienia
// =============================================================================
static const float  MUON_ID_CUT = 0.5f;
static const double P_SWITCH    = 0.7;   // p <= P_SWITCH -> model Low, p > P_SWITCH -> model High
static const long   MAX_EVENTS  = 100000;     // -1 = wszystkie

static const string SEG_COLL = "CentralTrackSegments";

// Predkosc swiatla [mm/ns]
static constexpr double c_light = 299.792458;

// Korekta vz dla ToF (jak w treningu LowP): t_corr = t_hit - T0_PER_MM * vz
static const bool   CORRECT_T0   = true;
static const double T0_PER_MM    = -1.0 / c_light;
static const bool   CUT_BAD_PATH = true;
static const double L_TOL        = 10.0;   // [mm]

// Pomijanie hitow z EcalBarrelImagingRecHits w ECal (tak jak w TestingLowP / treningu LowP).
// Jesli trening HighP tego NIE robil, ustaw false.
static const bool SKIP_IMAGING_HITS = true;

// Okna geometryczne ToF [mm] - IDENTYCZNE jak w treningu LowP
const double BTOF_RMIN = 600,  BTOF_RMAX = 720,  BTOF_ZMAX = 1200;
const double FTOF_ZMIN = 1750, FTOF_ZMAX = 1950, FTOF_RMIN = 100, FTOF_RMAX = 650;

static const float MISSING_SENTINEL = -999.f;

// =============================================================================
// Narzedzia
// =============================================================================
static TFile *gPlotFile = nullptr;   // plik tylko z wykresami (kanwami)

vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t g;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) result.push_back(g.gl_pathv[i]);
    globfree(&g);
    return result;
}

bool HasCollection(const vector<string> &names, const string &n)
{
    return find(names.begin(), names.end(), n) != names.end();
}

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

// =============================================================================
// Kalorymetry (TCM) - wspolne dla Low i High
// =============================================================================
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

struct TrackCalHits
{
    double sumEnergy = 0.0;
    double phiMin = 1e9, phiMax = -1e9;
    double etaMin = 1e9, etaMax = -1e9;
    double Rmin = 1e9, Rmax = -1e9;
    double maxHitE = -1.0;
    vector<double> R, dEta, dPhi, E;
};

// ECal: rcp.getClusters(); HCal: klastry z TrackClusterMatch dla tracka tej czastki.
// (Kierunek hitu liczony od poczatku ukladu - tak jak w obu treningach.)
TrackCalHits CollectHits(const edm4eic::ReconstructedParticle &rcp,
                         const vector<DetectorAssoc> &detectors,
                         const podio::Frame &frame,
                         bool fromRcpClusters,
                         const TLorentzVector &Partic)
{
    TrackCalHits result;
    vector<edm4eic::Cluster> clusters;
    set<podio::ObjectID> imagingHitIDs;

    if (fromRcpClusters)
    {
        for (const auto &c : rcp.getClusters()) clusters.push_back(c);

        if (SKIP_IMAGING_HITS && !clusters.empty())
        {
            auto avail = frame.getAvailableCollections();
            if (HasCollection(avail, "EcalBarrelImagingRecHits"))
            {
                for (const auto &ih : frame.get<edm4eic::CalorimeterHitCollection>("EcalBarrelImagingRecHits"))
                    imagingHitIDs.insert(ih.getObjectID());
            }
        }
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
            if (imagingHitIDs.count(hit.getObjectID())) continue;

            float HitE = hit.getEnergy();
            auto  pos  = hit.getPosition();

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

struct TrackFeatures
{
    float Energy = 0.f, Number = 0.f, EoverP = 0.f, AvgHitEnergy = 0.f;
    float SpreadPhi = 0.f, SpreadEta = 0.f, SpreadR = 0.f, MaxHitFrac = 0.f;
    float EnergyStdDev = 0.f, EnergyConcentration = 0.f;
    float R_Disp = 0.f, R_DispWeighted = 0.f, Eta_DispWeighted = 0.f, Phi_DispWeighted = 0.f;
};

TrackFeatures ComputeTrackFeatures(const TrackCalHits &hits, double trackP)
{
    TrackFeatures f;
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

    double spreadPhi = acos(cos(hits.phiMax - hits.phiMin));
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
        sumR2diff  += dR * dR;
        sumR2diffW += dR * dR * hits.E[k];

        sumDEta2diffW += hits.dEta[k] * hits.dEta[k] * hits.E[k];
        sumDPhi2diffW += hits.dPhi[k] * hits.dPhi[k] * hits.E[k];
    }

    double R_disp_unweighted = (n > 1) ? sqrt(sumR2diff / (n - 1)) : 0.0;
    double R_disp_weighted   = sqrt(sumR2diffW / sumEnergy);
    double Eta_disp_weighted = sqrt(sumDEta2diffW / sumEnergy);
    double Phi_disp_weighted = sqrt(sumDPhi2diffW / sumEnergy);

    f.Energy              = static_cast<float>(sumEnergy);
    f.Number              = static_cast<float>(n);
    f.EoverP              = static_cast<float>(sumEnergy / trackP);
    f.AvgHitEnergy        = static_cast<float>(meanE);
    f.SpreadPhi           = static_cast<float>(spreadPhi);
    f.SpreadEta           = static_cast<float>(spreadEta);
    f.SpreadR             = static_cast<float>(spreadR);
    f.MaxHitFrac          = static_cast<float>(hits.maxHitE / sumEnergy);
    f.EnergyStdDev        = static_cast<float>(energyStdDev);
    f.EnergyConcentration = static_cast<float>(energyConcentration);
    f.R_Disp              = static_cast<float>(R_disp_unweighted);
    f.R_DispWeighted      = static_cast<float>(R_disp_weighted);
    f.Eta_DispWeighted    = static_cast<float>(Eta_disp_weighted);
    f.Phi_DispWeighted    = static_cast<float>(Phi_disp_weighted);
    return f;
}

// =============================================================================
// ToF (tylko model Low)
// =============================================================================
struct BetaEstimate { double beta; bool valid; };

BetaEstimate CombineBeta(const std::vector<std::pair<double,double>> &hits)
{
    double sumXX = 0.0, sumXT = 0.0;
    for (auto &h : hits) {
        double t = h.first;
        double x = h.second / c_light;
        sumXX += x * x;
        sumXT += x * t;
    }
    if (sumXX <= 0.0) return {0.0, false};
    double u_hat = sumXT / sumXX;
    if (u_hat <= 0.0) return {0.0, false};
    return {1.0 / u_hat, true};
}

struct TrackToFFeatures
{
    float Beta          = -999.f;
    float MassSq        = -999.f;
    float NHitsBarrel   = 0.f;
    float NHitsEndcap   = 0.f;
    float NHitsTotal    = 0.f;
    float MinDistBarrel = -999.f;
    float MinDistEndcap = -999.f;
    float AvgLenBarrel  = -999.f;
    float AvgLenEndcap  = -999.f;
    float HasToF        = 0.f;
};

struct TofPoint { double x, y, z, path; };

static void CollectSegmentTofPoints(const edm4eic::TrackSegmentCollection &segs,
                                    const edm4eic::Track &trk,
                                    const TVector3 &ref,
                                    vector<TofPoint> &barrelPts,
                                    vector<TofPoint> &endcapPts)
{
    for (const auto seg : segs) {
        if (seg.getTrack() != trk) continue;
        for (const auto pt : seg.getPoints()) {
            const double x = pt.position.x, y = pt.position.y, z = pt.position.z;
            if (CUT_BAD_PATH &&
                pt.pathlength < (TVector3(x, y, z) - ref).Mag() - L_TOL) continue;
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

TrackToFFeatures ComputeToFFeatures(
    const TLorentzVector &Partic,
    const edm4eic::Track &trk,
    const TVector3 &ref,
    const edm4eic::TrackSegmentCollection &segments,
    const edm4eic::TrackerHitCollection &barrelHits,
    const edm4eic::TrackerHitCollection &endcapHits,
    double dR_cut_barrel, double dR_cut_endcap,
    double dist_cut_barrel, double dist_cut_endcap)
{
    TrackToFFeatures f;

    const double vz     = ref.Z();
    const double tShift = CORRECT_T0 ? T0_PER_MM * vz : 0.0;

    vector<TofPoint> barrelPts, endcapPts;
    CollectSegmentTofPoints(segments, trk, ref, barrelPts, endcapPts);

    double trackEta = Partic.Eta();
    double trackPhi = Partic.Phi();

    std::vector<std::pair<double,double>> matchedBarrel, matchedEndcap; // {time, length}

    double sumLenBarrel = 0.0, minDistBarrel = 1e18;
    double sumLenEndcap = 0.0, minDistEndcap = 1e18;

    for (const auto &hit : barrelHits) {
        auto p = hit.getPosition();
        TVector3 ToFPos(p.x, p.y, p.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);

        if (dR < dR_cut_barrel) {
            double dist = 0.0;
            const TofPoint *pt = NearestPoint(barrelPts, ToFPos, dist);
            if (!pt) continue;
            if (dist < dist_cut_barrel) {
                matchedBarrel.push_back({hit.getTime() - tShift, pt->path});
                sumLenBarrel += pt->path;
                if (dist < minDistBarrel) minDistBarrel = dist;
            }
        }
    }

    for (const auto &hit : endcapHits) {
        auto p = hit.getPosition();
        TVector3 ToFPos(p.x, p.y, p.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);

        if (dR < dR_cut_endcap) {
            double dist = 0.0;
            const TofPoint *pt = NearestPoint(endcapPts, ToFPos, dist);
            if (!pt) continue;
            if (dist < dist_cut_endcap) {
                matchedEndcap.push_back({hit.getTime() - tShift, pt->path});
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

// =============================================================================
// ONNX
// =============================================================================
float run_muon_id(Ort::Session &session, Ort::MemoryInfo &mem, const std::vector<float> &raw)
{
    int64_t shape[2] = {1, static_cast<int64_t>(raw.size())};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(raw.data()), raw.size(), shape, 2);

    const char* input_names[]  = {"raw_features"};
    const char* output_names[] = {"probabilities"};

    auto output = session.Run(Ort::RunOptions{nullptr},
                              input_names, &input_tensor, 1,
                              output_names, 1);

    float* probs = output[0].GetTensorMutableData<float>();
    return probs[1];   // P(muon)
}

// 41 cech (model LowP), kolejnosc RAW_COLS
std::vector<float> build_raw_low(const TrackFeatures &e, const TrackFeatures &h,
                                 const TrackToFFeatures &t,
                                 double trackP, double trackEta, double trackPhi)
{
    auto E = [&](float v) { return (e.Number > 0) ? v : MISSING_SENTINEL; };
    auto H = [&](float v) { return (h.Number > 0) ? v : MISSING_SENTINEL; };
    auto T = [&](float v) { return (t.HasToF > 0.5f) ? v : MISSING_SENTINEL; };

    return {
        e.Energy, e.Number, E(e.EoverP), E(e.AvgHitEnergy),
        E(e.SpreadPhi), E(e.SpreadEta), E(e.SpreadR), E(e.MaxHitFrac),
        E(e.EnergyStdDev), E(e.EnergyConcentration),
        E(e.R_Disp), E(e.R_DispWeighted), E(e.Eta_DispWeighted), E(e.Phi_DispWeighted),

        h.Energy, h.Number, H(h.EoverP), H(h.AvgHitEnergy),
        H(h.SpreadPhi), H(h.SpreadEta), H(h.SpreadR), H(h.MaxHitFrac),
        H(h.EnergyStdDev), H(h.EnergyConcentration),
        H(h.R_Disp), H(h.R_DispWeighted), H(h.Eta_DispWeighted), H(h.Phi_DispWeighted),

        T(t.Beta), T(t.MassSq), t.NHitsBarrel, t.NHitsEndcap, t.NHitsTotal,
        T(t.MinDistBarrel), T(t.MinDistEndcap), T(t.AvgLenBarrel), T(t.AvgLenEndcap), t.HasToF,

        static_cast<float>(trackP), static_cast<float>(trackEta), static_cast<float>(trackPhi)
    };
}

// 30 cech (model HighP) - bez sentineli, jak w treningu HighP
std::vector<float> build_raw_high(const TrackFeatures &e, const TrackFeatures &h,
                                  double trackP, double trackEta)
{
    return {
        e.Energy, h.Energy,
        e.Number, h.Number,
        e.EoverP, h.EoverP,
        e.AvgHitEnergy, h.AvgHitEnergy,
        e.SpreadPhi, e.SpreadEta, e.SpreadR,
        h.SpreadPhi, h.SpreadEta, h.SpreadR,
        e.MaxHitFrac, h.MaxHitFrac,
        e.EnergyStdDev, h.EnergyStdDev,
        e.EnergyConcentration, h.EnergyConcentration,
        e.R_Disp, e.R_DispWeighted,
        e.Eta_DispWeighted, e.Phi_DispWeighted,
        h.R_Disp, h.R_DispWeighted,
        h.Eta_DispWeighted, h.Phi_DispWeighted,
        static_cast<float>(trackP), static_cast<float>(trackEta)
    };
}

// =============================================================================
// Histogramy
// Zmienne: 0=Pt(0-20), 1=P(0-20), 2=Eta, 3=Pt(0-2, zoom), 4=P(0-2, zoom)
// =============================================================================
static const int NV = 5;
static const char  *vName[NV]  = {"Pt", "P", "Eta", "PtZoom", "PZoom"};
static const char  *vTitle[NV] = {"p_{T} [GeV/c]", "p [GeV/c]", "#eta", "p_{T} [GeV/c]", "p [GeV/c]"};
static const int    vBins[NV]  = {30, 40, 30, 40, 40};
static const double vLo[NV]    = {0, 0, -3.4, 0, 0};
static const double vHi[NV]    = {20, 20, 3.4, 2, 2};

struct Group
{
    string name;
    TH1F *resp = nullptr;
    TH1F *tot[NV], *rej[NV], *pas[NV];
};

Group MakeGroup(const string &name)
{
    Group g;
    g.name = name;
    g.resp = new TH1F(("h_Response_" + name).c_str(),
                      (name + " P(muon) Response;P(muon);Counts").c_str(), 100, 0, 1);
    for (int v = 0; v < NV; ++v) {
        const string s = string("_vs_") + vName[v];
        const string ax = string(";") + vTitle[v] + ";Counts";
        g.tot[v] = new TH1F(("h_" + name + "_Total"    + s).c_str(), (name + " Total"          + ax).c_str(), vBins[v], vLo[v], vHi[v]);
        g.rej[v] = new TH1F(("h_" + name + "_Rejected" + s).c_str(), (name + " Rejected"       + ax).c_str(), vBins[v], vLo[v], vHi[v]);
        g.pas[v] = new TH1F(("h_" + name + "_Passed"   + s).c_str(), (name + " Passed (Pmu>cut)" + ax).c_str(), vBins[v], vLo[v], vHi[v]);
    }
    return g;
}

void FillGroup(Group &g, double pt, double p, double eta, float Pmu)
{
    const double val[NV] = {pt, p, eta, pt, p};
    const bool passed = (Pmu > MUON_ID_CUT);
    g.resp->Fill(Pmu);
    for (int v = 0; v < NV; ++v) {
        g.tot[v]->Fill(val[v]);
        if (passed) g.pas[v]->Fill(val[v]);
        else        g.rej[v]->Fill(val[v]);
    }
}

TH1F *Ratio(TH1F *num, TH1F *den, const string &name, const string &title)
{
    TH1F *r = (TH1F*)num->Clone(name.c_str());
    r->SetTitle(title.c_str());
    r->Divide(num, den, 1.0, 1.0, "B");
    return r;
}

// =============================================================================
// Przetwarzanie jednego zestawu plikow
// =============================================================================
struct Models
{
    Ort::Session    *low;
    Ort::Session    *high;
    Ort::MemoryInfo *mem;
};

struct Hists
{
    Group muon, bkgAll;
    vector<Group> bkg;   // Pion, Electron, Kaon, Proton
};

static long gNLow = 0, gNHigh = 0;   // ile sladow poszlo do ktorego modelu

void ProcessFiles(const vector<string> &fileList, int species, const string &tag,
                  Models &M, Hists &H)
{
    if (fileList.empty()) return;

    // species: 0 = Muon, 1..4 = Pion, Electron, Kaon, Proton
    Group *gSpecies = (species == 0) ? &H.muon : &H.bkg.at(species - 1);

    vector<string> collections = {
        "ReconstructedParticles", "MCParticles", "ReconstructedParticleAssociations",
        "HcalBarrelTrackClusterMatches", "HcalEndcapNTrackClusterMatches", "LFHCALTrackClusterMatches",
        "EcalBarrelClusters", "EcalBarrelImagingRecHits", "EcalBarrelScFiRecHits",
        "EcalEndcapNClusters", "EcalEndcapNRecHits", "EcalEndcapPClusters", "EcalEndcapPRecHits",
        "HcalBarrelClusters", "HcalBarrelRecHits", "HcalEndcapNClusters", "HcalEndcapNMergedHits",
        "LFHCALClusters", "LFHCALRecHits",
        "TOFBarrelSharedRecHits", "TOFEndcapSharedRecHits", "CentralTrackSegments"
    };

    podio::ROOTReader reader;
    reader.openFiles(fileList);

    unsigned nEntries = reader.getEntries("events");
    cout << "Total events (" << tag << "): " << nEntries << " (" << fileList.size() << " files)" << endl;

    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};
    vector<DetectorAssoc> hcalDetectors, ecalDetectors;   // ECal z rcp.getClusters() -> pusta lista
    for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

    bool hasBToF = true, hasEToF = true, hasSeg = true;

    static const edm4eic::TrackerHitCollection   emptyToF;
    static const edm4eic::TrackSegmentCollection emptySeg;

    long nToProcess = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : nEntries;

    for (long entry = 0; entry < nToProcess; ++entry)
    {
        if ((entry + 1) % 50000 == 0) cout << tag << " event: " << entry + 1 << endl;

        podio::Frame frame(reader.readEntry("events", entry, collections));

        if (entry == 0) {
            cout << "Checking collections in first event (" << tag << "):" << endl;
            CheckAvailability(hcalDetectors, frame);
            auto names = frame.getAvailableCollections();
            hasBToF = HasCollection(names, "TOFBarrelSharedRecHits");
            hasEToF = HasCollection(names, "TOFEndcapSharedRecHits");
            hasSeg  = HasCollection(names, SEG_COLL);
            cout << "  TOFBarrelSharedRecHits: " << (hasBToF ? "OK" : "MISSING") << endl;
            cout << "  TOFEndcapSharedRecHits: " << (hasEToF ? "OK" : "MISSING") << endl;
            cout << "  " << SEG_COLL << ": " << (hasSeg ? "OK" : "MISSING") << endl;
        }

        const auto &tracks = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");

        const auto &barrelToF = hasBToF ? frame.get<edm4eic::TrackerHitCollection>("TOFBarrelSharedRecHits") : emptyToF;
        const auto &endcapToF = hasEToF ? frame.get<edm4eic::TrackerHitCollection>("TOFEndcapSharedRecHits") : emptyToF;
        const auto &segments  = hasSeg  ? frame.get<edm4eic::TrackSegmentCollection>(SEG_COLL)         : emptySeg;

        for (size_t particle = 0; particle < tracks.size(); ++particle)
        {
            const auto rcp = tracks[particle];

            auto mom = rcp.getMomentum();
            TLorentzVector Partic;
            Partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
            const double trackP   = Partic.P();
            const double trackPt  = Partic.Pt();
            const double trackEta = Partic.Eta();
            const double trackPhi = Partic.Phi();

            if (trackP <= 0) continue;

            const bool useLow = (trackP <= P_SWITCH);

            // ECal / HCal (wspolne)
            TrackCalHits ecalHits = CollectHits(rcp, ecalDetectors, frame, true,  Partic);
            TrackCalHits hcalHits = CollectHits(rcp, hcalDetectors, frame, false, Partic);
            TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP);
            TrackFeatures hcalF = ComputeTrackFeatures(hcalHits, trackP);

            float Pmu = 0.f;

            if (useLow)
            {
                // --- model LowP: wymaga tracka (HCal TCM + ToF) ---
                if (rcp.getTracks().empty()) continue;
                const auto trk = rcp.getTracks().at(0);

                const auto rp = rcp.getReferencePoint();
                TVector3 refPoint(rp.x, rp.y, rp.z);

                TrackToFFeatures tofF = ComputeToFFeatures(
                    Partic, trk, refPoint, segments, barrelToF, endcapToF,
                    0.8, 0.8, 6.0, 6.0);

                if (ecalF.Number <= 0 && hcalF.Number <= 0 && tofF.HasToF < 0.5) continue;

                std::vector<float> raw = build_raw_low(ecalF, hcalF, tofF, trackP, trackEta, trackPhi);
                Pmu = run_muon_id(*M.low, *M.mem, raw);
                ++gNLow;
            }
            else
            {
                // --- model HighP: tylko Calo ---
                if (ecalF.Number <= 0 && hcalF.Number <= 0) continue;

                std::vector<float> raw = build_raw_high(ecalF, hcalF, trackP, trackEta);
                Pmu = run_muon_id(*M.high, *M.mem, raw);
                ++gNHigh;
            }

            FillGroup(*gSpecies, trackPt, trackP, trackEta, Pmu);
            if (species != 0) FillGroup(H.bkgAll, trackPt, trackP, trackEta, Pmu);
        }
    }

    cout << "End of " << tag << ": " << nToProcess << " events" << endl;
}

// =============================================================================
// Rysowanie
// =============================================================================
static void StyleH(TH1F *h, int color, int marker, int lstyle = 1)
{
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetLineStyle(lstyle);
    h->SetMarkerSize(0.9);
    h->SetMinimum(0.0);
    h->SetMaximum(1.15);
}

static void DrawCanvas(const vector<TH1F*> &hs, const vector<string> &labels,
                       const string &title, const string &fileName)
{
    static const int colors[]  = {kBlue + 1, kRed + 1, kGreen + 2, kMagenta + 1, kOrange + 1, kCyan + 2};
    static const int markers[] = {20, 21, 22, 23, 33, 34};

    TCanvas c(("c_" + fileName).c_str(), title.c_str(), 900, 650);
    c.SetGrid();
    TLegend leg(0.62, 0.15, 0.89, 0.15 + 0.055 * hs.size());
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);

    for (size_t i = 0; i < hs.size(); ++i) {
        StyleH(hs[i], colors[i % 6], markers[i % 6]);
        hs[i]->SetTitle(title.c_str());
        hs[i]->Draw(i == 0 ? "E1" : "E1 SAME");
        leg.AddEntry(hs[i], labels[i].c_str(), "lp");
    }
    leg.Draw();

    // zapis do pliku z wykresami, potem powrot do poprzedniego katalogu
    if (gPlotFile) {
        TDirectory::TContext ctx(gPlotFile);
        c.Write(fileName.c_str());
    }
}

// =============================================================================
// Main
// =============================================================================
void TestingMuonIDFive()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");
    gStyle->SetOptStat(0);
    gSystem->mkdir("Plots", kTRUE);

    // ONNX Runtime: dwa modele
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // UWAGA: wczesniej sciezki byly zamienione miejscami (Low <- HighP.onnx, High <- LowP.onnx),
    // przez co 41 cech trafialo do modelu oczekujacego 30 -> Ort::Exception.
    Ort::Session sessionLow (env, "ONNX/xgb_muonID_LowP.onnx",  so);
    Ort::Session sessionHigh(env, "ONNX/xgb_muonID_HighP.onnx", so);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Models M{&sessionLow, &sessionHigh, &memory_info};

    // Gatunki: 0 = Muon (sygnal), 1..4 = background
    const vector<string> speciesNames = {"Muon", "Pion", "Electron", "Kaon", "Proton"};
    const string baseDir = "/run/media/epic/Data/Background/";
    const string subDir  = "/Podio-1.7.99/";

    TFile *outfile = new TFile("Plots/MuonID_Performance_Full.root", "RECREATE");

    gPlotFile = new TFile("Plots/MuonID_Plots.root", "RECREATE");
    outfile->cd();   // histogramy dalej do glownego pliku

    Hists H;
    H.muon   = MakeGroup("Muon");
    H.bkgAll = MakeGroup("Background");
    for (size_t i = 1; i < speciesNames.size(); ++i) H.bkg.push_back(MakeGroup("Bkg" + speciesNames[i]));

    for (int s = 0; s < (int)speciesNames.size(); ++s)
    {
        const string &sp = speciesNames[s];

        vector<string> lowFiles = ExpandGlob(baseDir + sp + subDir + "reco_*LowP*.root");
        vector<string> allFiles = ExpandGlob(baseDir + sp + subDir + "reco_*.root");

        // pliki "High" = wszystkie reco_*.root bez tych z LowP
        set<string> lowSet(lowFiles.begin(), lowFiles.end());
        vector<string> highFiles;
        for (const auto &f : allFiles) if (!lowSet.count(f)) highFiles.push_back(f);

        if (lowFiles.empty())  cerr << "[WARN] No LowP files for "  << sp << endl;
        if (highFiles.empty()) cerr << "[WARN] No High files for " << sp << endl;

        ProcessFiles(lowFiles,  s, sp + " [LowP files]",  M, H);
        ProcessFiles(highFiles, s, sp + " [High files]", M, H);
    }

    cout << "Tracks -> model Low: " << gNLow << ", model High: " << gNHigh
         << " (P_SWITCH = " << P_SWITCH << " GeV)" << endl;

    // -------------------------------------------------------------------------
    // Efficiency / Rejection / MisID (binomial errors)
    // -------------------------------------------------------------------------
    TH1F *effMuon[NV], *rejAll[NV], *rejSp[4][NV];

    for (int v = 0; v < NV; ++v)
    {
        const string vn = vName[v];
        const string ax = string(";") + vTitle[v];

        effMuon[v] = Ratio(H.muon.pas[v], H.muon.tot[v], "h_Muon_Efficiency_vs_" + vn,
                           "Muon Efficiency" + ax + ";Efficiency");

        rejAll[v] = Ratio(H.bkgAll.rej[v], H.bkgAll.tot[v], "h_Background_Rejection_vs_" + vn,
                          "Background Rejection" + ax + ";Rejection Fraction");
        Ratio(H.bkgAll.pas[v], H.bkgAll.tot[v], "h_Background_MisID_vs_" + vn,
              "Background MisID (inefficiency)" + ax + ";MisID Fraction");

        for (int i = 0; i < 4; ++i)
        {
            const string &n = speciesNames[i + 1];
            Group &g = H.bkg[i];
            rejSp[i][v] = Ratio(g.rej[v], g.tot[v], "h_Bkg" + n + "_Rejection_vs_" + vn,
                                n + " Rejection" + ax + ";Rejection Fraction");
            Ratio(g.pas[v], g.tot[v], "h_Bkg" + n + "_MisID_vs_" + vn,
                  n + " MisID (inefficiency)" + ax + ";MisID Fraction");
        }
    }

    // -------------------------------------------------------------------------
    // Wykresy: efficiency/rejection 0-20 (Pt, P, Eta) oraz zoom 0-2 (PtZoom, PZoom)
    // -------------------------------------------------------------------------
    const char *plotTag[NV]  = {"vs_Pt_0-20", "vs_P_0-20", "vs_Eta", "vs_Pt_0-2_zoom", "vs_P_0-2_zoom"};
    for (int v = 0; v < NV; ++v)
    {
        const string ax = string(";") + vTitle[v];

        // Muon efficiency + caly background rejection
        DrawCanvas({effMuon[v], rejAll[v]},
                   {"Muon efficiency", "Background rejection"},
                   "Muon ID performance" + ax + ";Fraction",
                   string("Performance_") + plotTag[v]);

        // Rejection osobno dla gatunkow + muon efficiency
        vector<TH1F*> hs = {effMuon[v]};
        vector<string> lb = {"Muon efficiency"};
        for (int i = 0; i < 4; ++i) {
            hs.push_back(rejSp[i][v]);
            lb.push_back(speciesNames[i + 1] + " rejection");
        }
        DrawCanvas(hs, lb, "Muon efficiency & rejection per species" + ax + ";Fraction",
                   string("PerSpecies_") + plotTag[v]);
    }

    outfile->Write();
    outfile->Close();

    gPlotFile->Close();
    delete gPlotFile;
    gPlotFile = nullptr;

    cout << "Histograms saved to Plots/MuonID_Performance_Full.root, plots saved to Plots/MuonID_Plots.root" << endl;
}

#ifndef __CLING__
int main(int argc, char** argv)
{
    TestingMuonIDFive();
    return 0;
}
#endif