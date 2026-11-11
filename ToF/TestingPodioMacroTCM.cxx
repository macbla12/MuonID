// TestingMacroCaloToF_Podio_TCM.cxx
//
// Wersja makra testowego (Calo + ToF + ONNX muon ID) z tymi samymi zmianami co w treningu
// (TrainingMacroCaloToF_Podio_TCM.cxx), zeby cechy w testingu byly liczone identycznie jak w treningu:
//
// 1) Zamiast asocjacji MC uzywamy TCM:
//      - kolekcja czastek: "ReconstructedParticles"
//      - ECal: rcp.getClusters()
//      - HCal: klastry z <Det>TrackClusterMatches dla rcp.getTracks().at(0)
//    Hity z cluster.getHits(). Zadna informacja MC nie jest uzywana.
//
// 2) Dlugosc toru ToF z CentralTrackSegments (pt.pathlength), jak w TrackLength.cxx:
//    segment dopasowany po seg.getTrack() == rcp.getTracks().at(0), punkty w oknach R/z ToF,
//    dla kazdego hitu ToF najblizszy punkt segmentu. ToFSim.cxx NIE jest juz potrzebny.
//
// 41 cech w kolejnosci RAW_COLS, MUON_ID_CUT, binning histogramow i logika
// wydajnosci/odrzucania bez zmian.
//
// Uruchom: root -l -q TestingMacroCaloToF_Podio_TCM.cxx++

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TROOT.h>
#include <TString.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TMath.h>
#include <iostream>
#include <string>
#include <vector>
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

// -----------------------------------------------------------------------------
// Okna geometryczne ToF [mm] (jak w TrackLength.cxx i w treningu) - musza byc IDENTYCZNE jak w treningu!
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Combine ToF hits into one beta estimate. Bez zmian.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Calorimeter side (TCM)
// -----------------------------------------------------------------------------
// "XClusters" -> "XTrackClusterMatches"
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

struct HistSet
{
    TH1F *HitR, *HitPhi, *HitZ, *HitEnergy, *HitTime;
    TH1F *DeltaEta, *DeltaPhi;
    TH2F *HitEtaR, *HitPhiR, *HitEtaPhi;
    TH1F *h_EoverP, *h_NHits;
    TH1F *h_AvgHitEnergy, *h_EnergyStdDev, *h_EnergyConcentration;
    TH1F *h_SpreadPhi, *h_SpreadEta, *h_SpreadR, *h_MaxHitFrac;
    TH1F *h_R_Disp, *h_R_DispWeighted, *h_Eta_DispWeighted, *h_Phi_DispWeighted;
};

struct TrackCalHits
{
    double sumEnergy = 0.0;
    double phiMin = 1e9, phiMax = -1e9;
    double etaMin = 1e9, etaMax = -1e9;
    double Rmin = 1e9, Rmax = -1e9;
    double maxHitE = -1.0;
    vector<double> R, dEta, dPhi, E;
};

// CollectHits - wersja TCM:
//  fromRcpClusters = true  -> ECal: rcp.getClusters()
//  fromRcpClusters = false -> HCal: klastry z TrackClusterMatch dla tracka tej czastki
TrackCalHits CollectHits(const edm4eic::ReconstructedParticle &rcp,
                         const vector<DetectorAssoc> &detectors,
                         const podio::Frame &frame,
                         bool fromRcpClusters,
                         const TLorentzVector &Partic,
                         double timingCutNs,
                         TH1F *hR, TH2F *hEtaR, TH2F *hPhiR, TH2F *hEtaPhi,
                         TH1F *hPhi, TH1F *hZ, TH1F *hE, TH1F *hT,
                         TH1F *hDEta, TH1F *hDPhi)
{
    TrackCalHits result;

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
            auto  pos  = hit.getPosition();

            if (hT) hT->Fill(HitT);
            if (HitT > timingCutNs) continue;

            TVector3 hitVec(pos.x, pos.y, pos.z);
            double hitEta = hitVec.Eta();
            double hitPhi = hitVec.Phi();
            double R = sqrt(pos.x * pos.x + pos.y * pos.y);

            if (hR)      hR->Fill(R);
            if (hEtaR)   hEtaR->Fill(hitEta, R);
            if (hPhiR)   hPhiR->Fill(hitPhi, R);
            if (hEtaPhi) hEtaPhi->Fill(hitEta, hitPhi);
            if (hPhi)    hPhi->Fill(hitPhi);
            if (hZ)      hZ->Fill(pos.z);
            if (hE)      hE->Fill(HitE);

            double dEta = hitEta - Partic.Eta();
            double dPhi = TVector2::Phi_mpi_pi(hitPhi - Partic.Phi());
            if (hDEta) hDEta->Fill(dEta);
            if (hDPhi) hDPhi->Fill(dPhi);

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

// Bez zmian (czysta matematyka).
TrackFeatures ComputeTrackFeatures(const TrackCalHits &hits, double trackP, HistSet &h)
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

    h.h_NHits->Fill(n);
    h.h_AvgHitEnergy->Fill(meanE);
    h.h_EnergyStdDev->Fill(energyStdDev);
    h.h_EnergyConcentration->Fill(energyConcentration);

    double spreadPhi = hits.phiMax - hits.phiMin;
    double spreadEta = hits.etaMax - hits.etaMin;
    double spreadR   = hits.Rmax - hits.Rmin;

    h.h_EoverP->Fill(sumEnergy / trackP);
    h.h_SpreadPhi->Fill(spreadPhi);
    h.h_SpreadEta->Fill(spreadEta);
    h.h_SpreadR->Fill(spreadR);
    h.h_MaxHitFrac->Fill(hits.maxHitE / sumEnergy);

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

    h.h_R_Disp->Fill(R_disp_unweighted);
    h.h_R_DispWeighted->Fill(R_disp_weighted);
    h.h_Eta_DispWeighted->Fill(Eta_disp_weighted);
    h.h_Phi_DispWeighted->Fill(Phi_disp_weighted);

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

HistSet MakeHistSet(const string &prefix)
{
    HistSet h;
    h.HitR      = new TH1F((prefix + "_HitR").c_str(), (prefix + " Hit R;R [mm];Counts").c_str(), 200, 0, 3200);
    h.HitEtaR   = new TH2F((prefix + "_HitEtaR").c_str(), (prefix + " HitEta vs R;#eta;R [mm]").c_str(), 200, -3.5, 3.5, 200, 0, 3200);
    h.HitPhiR   = new TH2F((prefix + "_HitPhiR").c_str(), (prefix + " HitPhi vs R;#phi;R [mm]").c_str(), 200, -3.15, 3.15, 200, 0, 3200);
    h.HitEtaPhi = new TH2F((prefix + "_HitEtaPhi").c_str(), (prefix + " HitEta vs Phi;#eta;#phi").c_str(), 200, -3.5, 3.5, 200, -3.15, 3.15);
    h.HitPhi    = new TH1F((prefix + "_HitPhi").c_str(), (prefix + " Hit Phi;#phi;Counts").c_str(), 200, -3.15, 3.15);
    h.HitZ      = new TH1F((prefix + "_HitZ").c_str(), (prefix + " Hit Z;Z [mm];Counts").c_str(), 200, -3000, 3000);
    h.HitEnergy = new TH1F((prefix + "_HitEnergy").c_str(), (prefix + " Hit Energy;E [GeV];Counts").c_str(), 200, 0, 1);
    h.HitTime   = new TH1F((prefix + "_HitTime").c_str(), (prefix + " Hit Time;t [ns];Counts").c_str(), 200, 0, 200);
    h.DeltaEta  = new TH1F((prefix + "_DeltaEta").c_str(), (prefix + " #Delta#eta;#Delta#eta;Counts").c_str(), 200, -1, 1);
    h.DeltaPhi  = new TH1F((prefix + "_DeltaPhi").c_str(), (prefix + " #Delta#phi;#Delta#phi;Counts").c_str(), 200, -1, 1);
    h.h_EoverP  = new TH1F((prefix + "_h_EoverP").c_str(), (prefix + " E/p;E/p;Counts").c_str(), 150, 0, 3);
    h.h_NHits   = new TH1F((prefix + "_h_NHits").c_str(), (prefix + " N hits;N;Counts").c_str(), 60, 0, 60);
    h.h_AvgHitEnergy = new TH1F((prefix + "_h_AvgHitEnergy").c_str(), (prefix + " AvgHitEnergy;E [GeV];Counts").c_str(), 150, 0, 1);
    h.h_EnergyStdDev = new TH1F((prefix + "_h_EnergyStdDev").c_str(), (prefix + " Energy StdDev;E [GeV];Counts").c_str(), 150, 0, 0.5);
    h.h_EnergyConcentration = new TH1F((prefix + "_h_EnergyConcentration").c_str(), (prefix + " Energy Concentration;.;Counts").c_str(), 150, 0, 1);
    h.h_SpreadPhi = new TH1F((prefix + "_h_SpreadPhi").c_str(), (prefix + " Spread Phi;#phi;Counts").c_str(), 150, 0, 3.2);
    h.h_SpreadEta = new TH1F((prefix + "_h_SpreadEta").c_str(), (prefix + " Spread Eta;#eta;Counts").c_str(), 150, 0, 3.2);
    h.h_SpreadR = new TH1F((prefix + "_h_SpreadR").c_str(), (prefix + " Spread R;R [mm];Counts").c_str(), 150, 0, 3200);
    h.h_MaxHitFrac = new TH1F((prefix + "_h_MaxHitFrac").c_str(), (prefix + " MaxHitFrac;Frac;Counts").c_str(), 150, 0, 1);
    h.h_R_Disp = new TH1F((prefix + "_h_R_Disp").c_str(), (prefix + " R Disp;R [mm];Counts").c_str(), 150, 0, 1000);
    h.h_R_DispWeighted = new TH1F((prefix + "_h_R_DispWeighted").c_str(), (prefix + " R Disp Weighted;R [mm];Counts").c_str(), 150, 0, 1000);
    h.h_Eta_DispWeighted = new TH1F((prefix + "_h_Eta_DispWeighted").c_str(), (prefix + " Eta Disp Weighted;#eta;Counts").c_str(), 150, 0, 1);
    h.h_Phi_DispWeighted = new TH1F((prefix + "_h_Phi_DispWeighted").c_str(), (prefix + " Phi Disp Weighted;#phi;Counts").c_str(), 150, 0, 3.2);
    return h;
}

// -----------------------------------------------------------------------------
// ToF side (dlugosc toru z CentralTrackSegments)
// -----------------------------------------------------------------------------
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

TrackToFFeatures ComputeToFFeatures(
    const TLorentzVector &Partic,
    const edm4eic::Track &trk,
    const edm4eic::TrackSegmentCollection &segments,
    const edm4eic::TrackerHitCollection &barrelHits,
    const edm4eic::TrackerHitCollection &endcapHits,
    double dR_cut_barrel, double dR_cut_endcap,
    double dist_cut_barrel, double dist_cut_endcap)
{
    TrackToFFeatures f;

    vector<TofPoint> barrelPts, endcapPts;
    CollectSegmentTofPoints(segments, trk, barrelPts, endcapPts);

    double trackEta = Partic.Eta();
    double trackPhi = Partic.Phi();

    std::vector<std::pair<double,double>> matchedBarrel; // {time, length}
    std::vector<std::pair<double,double>> matchedEndcap;

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
                matchedBarrel.push_back({hit.getTime(), pt->path});
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

// -----------------------------------------------------------------------------
// ONNX runtime helper (bez zmian): input "raw_features" (1,41), output "probabilities".
// -----------------------------------------------------------------------------
float run_muon_id(
    Ort::Session& session,
    Ort::MemoryInfo& mem,
    const std::vector<float>& raw)
{
    int64_t shape[2] = {1, static_cast<int64_t>(raw.size())};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(raw.data()), raw.size(), shape, 2);

    const char* input_names[] = {"raw_features"};
    const char* output_names[] = {"probabilities"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    float* probs = output_tensors[0].GetTensorMutableData<float>();
    return probs[1];   // P(muon)
}

// -----------------------------------------------------------------------------
// 41-element raw feature vector in RAW_COLS order (bez zmian).
// -----------------------------------------------------------------------------
static const int N_RAW = 41;
static const float MISSING_SENTINEL = -999.f;

std::vector<float> build_raw_features_from_components(
    // ECal (14)
    float ECalEnergy, float ECalNumber, float ECalEoverP, float ECalAvgHitEnergy,
    float ECalSpreadPhi, float ECalSpreadEta, float ECalSpreadR, float ECalMaxHitFrac,
    float ECalEnergyStdDev, float ECalEnergyConcentration,
    float ECalR_Disp, float ECalR_DispWeighted, float ECalEta_DispWeighted, float ECalPhi_DispWeighted,
    // HCal (14)
    float HCalEnergy, float HCalNumber, float HCalEoverP, float HCalAvgHitEnergy,
    float HCalSpreadPhi, float HCalSpreadEta, float HCalSpreadR, float HCalMaxHitFrac,
    float HCalEnergyStdDev, float HCalEnergyConcentration,
    float HCalR_Disp, float HCalR_DispWeighted, float HCalEta_DispWeighted, float HCalPhi_DispWeighted,
    // ToF (10)
    float ToFBeta, float ToFMassSq, float ToFNHitsBarrel, float ToFNHitsEndcap, float ToFNHitsTotal,
    float ToFMinDistBarrel, float ToFMinDistEndcap, float ToFAvgLenBarrel, float ToFAvgLenEndcap, float ToFHasToF,
    // Track (3)
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

// -----------------------------------------------------------------------------
// Main testing macro
// -----------------------------------------------------------------------------
void TestingPodioMacroTCM()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    const double TIMING_CUT_NS = 20.0;
    const float  MUON_ID_CUT   = 0.5f;
    const long   MAX_EVENTS    = 5000;   // -1 = wszystkie

    const string SEG_COLL = "CentralTrackSegments";

    // ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, "ONNX/xgb_muonID.onnx", session_options);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    static constexpr int NumOfFiles = 2;
    vector<string> filePatterns(NumOfFiles);
    filePatterns.at(0) = "/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
    //filePatterns.at(0) = "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_100.root";
    filePatterns.at(1) = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
    //filePatterns.at(1) = "/run/media/epic/Data/Tau/reco/Energy_10x275/old/double_pi/recoDoublePi.root";

    // ECal: rcp.getClusters() (kolekcje match niepotrzebne), HCal: TrackClusterMatch
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    TFile *outfile = new TFile("Plots/MuonID_Performance_Podio.root", "RECREATE");

    // Response histograms
    TH1F *h_Response_Muon = new TH1F("h_Response_Muon", "P(muon) Response for True Muons;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Pion = new TH1F("h_Response_Pion", "P(muon) Response for True Pions;P(muon);Counts", 100, 0, 1);

    // Efficiency / rejection counters
    TH1F *h_Muon_Total_vs_Pt  = new TH1F("h_Muon_Total_vs_Pt", "Muon Total vs pT;p_{T} [GeV/c];Counts", 40, 0, 2);
    TH1F *h_Muon_Passed_vs_Pt = new TH1F("h_Muon_Passed_vs_Pt", "Muon Passed vs pT;p_{T} [GeV/c];Counts", 40, 0, 2);

    TH1F *h_Muon_Total_vs_P   = new TH1F("h_Muon_Total_vs_P", "Muon Total vs p;p [GeV/c];Counts", 40, 0, 2);
    TH1F *h_Muon_Passed_vs_P  = new TH1F("h_Muon_Passed_vs_P", "Muon Passed vs p;p [GeV/c];Counts", 40, 0, 2);

    TH1F *h_Muon_Total_vs_Eta  = new TH1F("h_Muon_Total_vs_Eta", "Muon Total vs #eta;#eta;Counts", 30, -3.4, 3.4);
    TH1F *h_Muon_Passed_vs_Eta = new TH1F("h_Muon_Passed_vs_Eta", "Muon Passed vs #eta;#eta;Counts", 30, -3.4, 3.4);

    TH1F *h_Pion_Total_vs_Pt    = new TH1F("h_Pion_Total_vs_Pt", "Pion Total vs pT;p_{T} [GeV/c];Counts", 40, 0, 2);
    TH1F *h_Pion_Rejected_vs_Pt = new TH1F("h_Pion_Rejected_vs_Pt", "Pion Rejected vs pT;p_{T} [GeV/c];Counts", 40, 0, 2);

    TH1F *h_Pion_Total_vs_P     = new TH1F("h_Pion_Total_vs_P", "Pion Total vs p;p [GeV/c];Counts", 40, 0, 2);
    TH1F *h_Pion_Rejected_vs_P  = new TH1F("h_Pion_Rejected_vs_P", "Pion Rejected vs p;p [GeV/c];Counts", 40, 0, 2);

    TH1F *h_Pion_Total_vs_Eta    = new TH1F("h_Pion_Total_vs_Eta", "Pion Total vs #eta;#eta;Counts", 30, -3.4, 3.4);
    TH1F *h_Pion_Rejected_vs_Eta = new TH1F("h_Pion_Rejected_vs_Eta", "Pion Rejected vs #eta;#eta;Counts", 30, -3.4, 3.4);

    // Puste kolekcje uzywane, gdy kolekcji nie ma w pliku.
    static const edm4eic::TrackerHitCollection   emptyToF;
    static const edm4eic::TrackSegmentCollection emptySeg;

    // Pomocnicze histogramy dla ComputeTrackFeatures - tworzone RAZ, odlaczone od pliku wyjsciowego.
    TH1::AddDirectory(kFALSE);
    HistSet ecalHists = MakeHistSet("ECal_tmp");
    HistSet hcalHists = MakeHistSet("HCal_tmp");
    TH1::AddDirectory(kTRUE);

    for (int File = 0; File < NumOfFiles; File++)
    {
        bool isMuonFile = (File == 0);
        string name = isMuonFile ? "Muons" : "Pions";

        vector<string> fileList = ExpandGlob(filePatterns.at(File));
        if (fileList.empty()) {
            cerr << "[ERROR] No files match: " << filePatterns.at(File) << endl;
            continue;
        }

        podio::ROOTReader reader;
        reader.openFiles(fileList);

        unsigned nEntries = reader.getEntries("events");
        cout << "Total events in chain (" << name << "): " << nEntries << " (" << fileList.size() << " files)" << endl;

        // ECal: rcp.getClusters() -> pusta lista detektorow (nieuzywana)
        vector<DetectorAssoc> ecalDetectors;

        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

        bool hasBToF = true, hasEToF = true, hasSeg = true;

        long nToProcess = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : nEntries;

        for (long entry = 0; entry < nToProcess; ++entry)
        {
            if ((entry + 1) % 50000 == 0) cout << "File " << name << " event: " << entry + 1 << endl;

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

            const auto &barrelToF = hasBToF ? frame.get<edm4eic::TrackerHitCollection>("TOFBarrelRecHits") : emptyToF;
            const auto &endcapToF = hasEToF ? frame.get<edm4eic::TrackerHitCollection>("TOFEndcapRecHits") : emptyToF;
            const auto &segments  = hasSeg  ? frame.get<edm4eic::TrackSegmentCollection>(SEG_COLL)         : emptySeg;

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

                // Selekcja zgodna z treningiem
                if (trackP <= 0 || trackP > 1.0) continue;
                //if (trackEta >= 1.0 && trackEta <= 1.3) continue;
                //if (trackEta <= -1.25) continue;

                // track potrzebny dla HCal (TCM) i ToF (CentralTrackSegments), jak w treningu
                if (rcp.getTracks().empty()) continue;
                const auto trk = rcp.getTracks().at(0);

                int charge = static_cast<int>(rcp.getCharge());
                (void)charge;

                // -------- ECal: rcp.getClusters() --------
                TrackCalHits ecalHits = CollectHits(rcp, ecalDetectors, frame, true, Partic, TIMING_CUT_NS,
                    ecalHists.HitR, ecalHists.HitEtaR, ecalHists.HitPhiR, ecalHists.HitEtaPhi,
                    ecalHists.HitPhi, ecalHists.HitZ, ecalHists.HitEnergy, ecalHists.HitTime,
                    ecalHists.DeltaEta, ecalHists.DeltaPhi);
                TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP, ecalHists);

                // -------- HCal: TrackClusterMatch --------
                TrackCalHits hcalHits = CollectHits(rcp, hcalDetectors, frame, false, Partic, TIMING_CUT_NS,
                    hcalHists.HitR, hcalHists.HitEtaR, hcalHists.HitPhiR, hcalHists.HitEtaPhi,
                    hcalHists.HitPhi, hcalHists.HitZ, hcalHists.HitEnergy, hcalHists.HitTime,
                    hcalHists.DeltaEta, hcalHists.DeltaPhi);
                TrackFeatures hcalF = ComputeTrackFeatures(hcalHits, trackP, hcalHists);

                // -------- ToF (dlugosc z CentralTrackSegments) --------
                TrackToFFeatures tofF = ComputeToFFeatures(
                    Partic, trk, segments, barrelToF, endcapToF,
                    /*dR_cut_barrel*/0.8, /*dR_cut_endcap*/0.8,
                    /*dist_cut_barrel*/6.0, /*dist_cut_endcap*/6.0);

                // Drop tracks invisible everywhere (jak w treningu)
                if (ecalF.Number <= 0 && hcalF.Number <= 0 && tofF.HasToF < 0.5) continue;

                // Map computed features -> raw fields expected by ONNX preprocessor
                float ECalEnergy              = ecalF.Energy;
                float ECalNumber              = ecalF.Number;
                float ECalEoverP              = (ecalF.Number > 0) ? ecalF.EoverP : MISSING_SENTINEL;
                float ECalAvgHitEnergy        = (ecalF.Number > 0) ? ecalF.AvgHitEnergy : MISSING_SENTINEL;
                float ECalSpreadPhi           = (ecalF.Number > 0) ? ecalF.SpreadPhi : MISSING_SENTINEL;
                float ECalSpreadEta           = (ecalF.Number > 0) ? ecalF.SpreadEta : MISSING_SENTINEL;
                float ECalSpreadR             = (ecalF.Number > 0) ? ecalF.SpreadR : MISSING_SENTINEL;
                float ECalMaxHitFrac          = (ecalF.Number > 0) ? ecalF.MaxHitFrac : MISSING_SENTINEL;
                float ECalEnergyStdDev        = (ecalF.Number > 0) ? ecalF.EnergyStdDev : MISSING_SENTINEL;
                float ECalEnergyConcentration = (ecalF.Number > 0) ? ecalF.EnergyConcentration : MISSING_SENTINEL;
                float ECalR_Disp              = (ecalF.Number > 0) ? ecalF.R_Disp : MISSING_SENTINEL;
                float ECalR_DispWeighted      = (ecalF.Number > 0) ? ecalF.R_DispWeighted : MISSING_SENTINEL;
                float ECalEta_DispWeighted    = (ecalF.Number > 0) ? ecalF.Eta_DispWeighted : MISSING_SENTINEL;
                float ECalPhi_DispWeighted    = (ecalF.Number > 0) ? ecalF.Phi_DispWeighted : MISSING_SENTINEL;

                float HCalEnergy              = hcalF.Energy;
                float HCalNumber              = hcalF.Number;
                float HCalEoverP              = (hcalF.Number > 0) ? hcalF.EoverP : MISSING_SENTINEL;
                float HCalAvgHitEnergy        = (hcalF.Number > 0) ? hcalF.AvgHitEnergy : MISSING_SENTINEL;
                float HCalSpreadPhi           = (hcalF.Number > 0) ? hcalF.SpreadPhi : MISSING_SENTINEL;
                float HCalSpreadEta           = (hcalF.Number > 0) ? hcalF.SpreadEta : MISSING_SENTINEL;
                float HCalSpreadR             = (hcalF.Number > 0) ? hcalF.SpreadR : MISSING_SENTINEL;
                float HCalMaxHitFrac          = (hcalF.Number > 0) ? hcalF.MaxHitFrac : MISSING_SENTINEL;
                float HCalEnergyStdDev        = (hcalF.Number > 0) ? hcalF.EnergyStdDev : MISSING_SENTINEL;
                float HCalEnergyConcentration = (hcalF.Number > 0) ? hcalF.EnergyConcentration : MISSING_SENTINEL;
                float HCalR_Disp              = (hcalF.Number > 0) ? hcalF.R_Disp : MISSING_SENTINEL;
                float HCalR_DispWeighted      = (hcalF.Number > 0) ? hcalF.R_DispWeighted : MISSING_SENTINEL;
                float HCalEta_DispWeighted    = (hcalF.Number > 0) ? hcalF.Eta_DispWeighted : MISSING_SENTINEL;
                float HCalPhi_DispWeighted    = (hcalF.Number > 0) ? hcalF.Phi_DispWeighted : MISSING_SENTINEL;

                float ToFBeta          = (tofF.HasToF > 0.5f) ? tofF.Beta : MISSING_SENTINEL;
                float ToFMassSq        = (tofF.HasToF > 0.5f) ? tofF.MassSq : MISSING_SENTINEL;
                float ToFNHitsBarrel   = tofF.NHitsBarrel;
                float ToFNHitsEndcap   = tofF.NHitsEndcap;
                float ToFNHitsTotal    = tofF.NHitsTotal;
                float ToFMinDistBarrel = (tofF.HasToF > 0.5f) ? tofF.MinDistBarrel : MISSING_SENTINEL;
                float ToFMinDistEndcap = (tofF.HasToF > 0.5f) ? tofF.MinDistEndcap : MISSING_SENTINEL;
                float ToFAvgLenBarrel  = (tofF.HasToF > 0.5f) ? tofF.AvgLenBarrel : MISSING_SENTINEL;
                float ToFAvgLenEndcap  = (tofF.HasToF > 0.5f) ? tofF.AvgLenEndcap : MISSING_SENTINEL;
                float ToFHasToF        = tofF.HasToF;

                std::vector<float> raw = build_raw_features_from_components(
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
                    static_cast<float>(trackP), static_cast<float>(trackEta), static_cast<float>(trackPhi));

                float Pmu = run_muon_id(session, memory_info, raw);

                if (isMuonFile)
                {
                    h_Response_Muon->Fill(Pmu);

                    h_Muon_Total_vs_Pt->Fill(trackPt);
                    h_Muon_Total_vs_P->Fill(trackP);
                    h_Muon_Total_vs_Eta->Fill(trackEta);

                    if (Pmu > MUON_ID_CUT)
                    {
                        h_Muon_Passed_vs_Pt->Fill(trackPt);
                        h_Muon_Passed_vs_P->Fill(trackP);
                        h_Muon_Passed_vs_Eta->Fill(trackEta);
                    }
                }
                else // Pion file
                {
                    h_Response_Pion->Fill(Pmu);

                    h_Pion_Total_vs_Pt->Fill(trackPt);
                    h_Pion_Total_vs_P->Fill(trackP);
                    h_Pion_Total_vs_Eta->Fill(trackEta);

                    if (Pmu <= MUON_ID_CUT) // pion correctly rejected
                    {
                        h_Pion_Rejected_vs_Pt->Fill(trackPt);
                        h_Pion_Rejected_vs_P->Fill(trackP);
                        h_Pion_Rejected_vs_Eta->Fill(trackEta);
                    }
                }
            } // tracks
        } // events

        cout << "End of " << name << ": " << nToProcess << " events" << endl;
    } // files

    // Efficiency / rejection (binomial errors)
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

    outfile->Write();
    outfile->Close();

    cout << "Histograms saved to Plots/MuonID_Performance_Podio_TCM.root" << endl;
}

// Standalone program entry point (poza ROOT/Cling).
#ifndef __CLING__
int main(int argc, char** argv)
{
    TestingPodioMacroTCM();
    return 0;
}
#endif