// TrainingMacroCaloToF_Podio.cxx
//
// Wersja podio / EDM4eic makra Calo+ToF (TrainingMacro z ToF).
// Fizyka, cechy, cieta, wartosci -999, binning histogramow, drzewo ML i pliki wyjsciowe
// (ONNX/MLDataCaloToF.root, Plots/CaloToF/Diag_<Muons|Pions>.root) sa takie same.
// Zmienia sie tylko odczyt: podio::Frame zamiast TTreeReader na flat branchach.
//
// - Hity kalorymetrow: cluster.getHits() (wlasciwa kolekcja, np. HcalEndcapNMergedHits).
// - Hity ToF: TOFBarrelRecHits / TOFEndcapRecHits jako edm4eic::TrackerHit.
//   (Jesli kompilator zglosi inny typ, sprawdz: frame.get("TOFBarrelRecHits")->getValueTypeName())
// - ToFSim.cxx (ToFSim(), ToFResults, c_light) jest niezmienione i musi byc w tym samym katalogu.
//
// Uruchom: root -l -q TrainingMacroCaloToF_Podio.cxx++

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TTree.h>
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

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include <edm4hep/MCParticleCollection.h>

#include <edm4eic/MCRecoParticleAssociationCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/MCRecoClusterParticleAssociationCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <edm4eic/TrackerHitCollection.h>

using namespace std;

// Brings in ToFSim(), ToFResults, c_light, ... (ToFSim.cxx must be in the same directory)
#include "ToFSim.cxx"

// =============================================================================
// Combine several (time, length) ToF hits of the SAME track into one beta estimate
// (least squares through the origin). Bez zmian.
// =============================================================================
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

// =============================================================================
// glob (zastepuje TChain::Add(pattern))
// =============================================================================
vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t g;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) result.push_back(g.gl_pathv[i]);
    globfree(&g);
    return result;
}

// =============================================================================
// ====================  CALORIMETER SIDE  ====================================
// =============================================================================

// Per-detector: nazwa kolekcji asocjacji klaster<->sim. "XClusters" -> "XClusterAssociations"
struct DetectorAssoc
{
    string clusterName;
    string assocCollName;
    bool   available = true;
};

DetectorAssoc MakeDetectorAssoc(const string &clusterName)
{
    DetectorAssoc d;
    d.clusterName   = clusterName;
    d.assocCollName = clusterName.substr(0, clusterName.size() - 1) + "Associations";
    return d;
}

bool HasCollection(const vector<string> &names, const string &n)
{
    return find(names.begin(), names.end(), n) != names.end();
}

void CheckAvailability(vector<DetectorAssoc> &dets, const podio::Frame &frame)
{
    auto names = frame.getAvailableCollections();
    for (auto &d : dets) {
        d.available = HasCollection(names, d.assocCollName);
        cout << "  " << d.clusterName << " -> '" << d.assocCollName << "' : "
             << (d.available ? "OK" : "MISSING (detektor pominiety)") << endl;
    }
}

struct HistSet
{
    TH1F *HitR, *HitPhi, *HitZ, *HitEnergy, *HitTime;
    TH1F *DeltaEta, *DeltaPhi;
    TH2F *hHitEtaR, *hHitPhiR, *hHitEtaPhi;

    TH1F *h_Energy, *h_EoverP, *h_AvgHitEnergy, *h_NHits;
    TH1F *h_SpreadPhi, *h_SpreadEta, *h_SpreadR;
    TH1F *h_MaxHitFrac, *h_EnergyStdDev;

    TH1F *h_R_Disp, *h_R_DispWeighted, *h_EnergyConcentration;
    TH1F *h_Eta_DispWeighted, *h_Phi_DispWeighted;
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

// CollectHits - wersja podio: asocjacje z Frame, dopasowanie po sim, hity z cluster.getHits().
TrackCalHits CollectHits(const edm4hep::MCParticle &simPart,
                         const vector<DetectorAssoc> &detectors,
                         const podio::Frame &frame,
                         const TLorentzVector &Partic,
                         double timingCutNs,
                         TH1F *hR, TH2F *hEtaR, TH2F *hPhiR, TH2F *hEtaPhi,
                         TH1F *hPhi, TH1F *hZ, TH1F *hE, TH1F *hT,
                         TH1F *hDEta, TH1F *hDPhi)
{
    TrackCalHits result;

    for (const auto &det : detectors)
    {
        if (!det.available) continue;

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

// Bez zmian (czysta matematyka na TrackCalHits).
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
    h.h_Energy->Fill(sumEnergy);
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

    h.HitR       = new TH1F((prefix + "_HitR").c_str(), (prefix + " Hit R position;R [mm];Counts").c_str(), 2000, 0, 3200);
    h.HitPhi     = new TH1F((prefix + "_HitPhi").c_str(), (prefix + " Hit Phi position;#phi;Counts").c_str(), 200, -3.15, 3.15);
    h.HitZ       = new TH1F((prefix + "_HitZ").c_str(), (prefix + " Hit Z position;Z [mm];Counts").c_str(), 200, -3000, 3000);
    h.HitEnergy  = new TH1F((prefix + "_HitEnergy").c_str(), (prefix + " Hit Energy;Energy [GeV];Counts").c_str(), 200, 0, 1);
    h.HitTime    = new TH1F((prefix + "_HitTime").c_str(), (prefix + " Hit Time;Time [ns];Counts").c_str(), 200, 0, 200);

    h.DeltaEta   = new TH1F((prefix + "_DeltaEta").c_str(), (prefix + " Delta Eta (hit - track);#Delta#eta;Counts").c_str(), 200, -1, 1);
    h.DeltaPhi   = new TH1F((prefix + "_DeltaPhi").c_str(), (prefix + " Delta Phi (hit - track);#Delta#phi;Counts").c_str(), 200, -1, 1);

    h.hHitEtaR   = new TH2F((prefix + "_hHitEtaR").c_str(), (prefix + " Hit map: #eta vs R;#eta;R [mm]").c_str(), 120, -3.5, 3.5, 200, 0, 3200);
    h.hHitPhiR   = new TH2F((prefix + "_hHitPhiR").c_str(), (prefix + " Hit map: #phi vs R;#phi [rad];R [mm]").c_str(), 180, -TMath::Pi(), TMath::Pi(), 200, 0, 3200);
    h.hHitEtaPhi = new TH2F((prefix + "_hHitEtaPhi").c_str(), (prefix + " Hit map: #eta vs #phi;#eta;#phi [rad]").c_str(), 120, -3.5, 3.5, 180, -TMath::Pi(), TMath::Pi());

    h.h_Energy       = new TH1F((prefix + "_h_Energy").c_str(), (prefix + " Total Energy;E_{calo} [GeV];Counts").c_str(), 50, 0, 1);
    h.h_EoverP       = new TH1F((prefix + "_h_EoverP").c_str(), (prefix + " E/p_{track};E/p;Counts").c_str(), 60, 0, 3);
    h.h_AvgHitEnergy = new TH1F((prefix + "_h_AvgHitEnergy").c_str(), (prefix + " Average Hit Energy per Track;E_{avg} [GeV];Counts").c_str(), 50, 0, 1);
    h.h_NHits        = new TH1F((prefix + "_h_NHits").c_str(), (prefix + " Number of Hits per Track;N_{hits};Counts").c_str(), 50, -0.5, 99.5);
    h.h_SpreadPhi    = new TH1F((prefix + "_h_SpreadPhi").c_str(), (prefix + " Spread in #phi (max-min);#phi_{Spread} [rad];Counts").c_str(), 50, 0, 1.0);
    h.h_SpreadEta    = new TH1F((prefix + "_h_SpreadEta").c_str(), (prefix + " Spread in #eta (max-min);#eta_{Spread} ;Counts").c_str(), 50, 0, 1.0);
    h.h_SpreadR      = new TH1F((prefix + "_h_SpreadR").c_str(), (prefix + " Radial spread of hits (max-min);R_{Spread} [mm];Counts").c_str(), 50, 0, 2000);
    h.h_MaxHitFrac   = new TH1F((prefix + "_h_MaxHitFrac").c_str(), (prefix + " Max hit energy / total;f_{max};Counts").c_str(), 50, 0, 1.05);
    h.h_EnergyStdDev = new TH1F((prefix + "_h_EnergyStdDev").c_str(), (prefix + " Std Dev of hit energies;#sigma_{E} [GeV];Counts").c_str(), 60, 0, 0.3);

    h.h_R_Disp              = new TH1F((prefix + "_h_R_Disp").c_str(), (prefix + " R dispersion (unweighted);D_{R} [mm];Counts").c_str(), 50, 0, 400);
    h.h_R_DispWeighted      = new TH1F((prefix + "_h_R_DispWeighted").c_str(), (prefix + " R dispersion (energy-weighted);D_{R}^{w} [mm];Counts").c_str(), 50, 0, 400);
    h.h_EnergyConcentration = new TH1F((prefix + "_h_EnergyConcentration").c_str(), (prefix + " Energy concentration #Sigma E_{i}^{2}/(#Sigma E_{i})^{2};C_{E};Counts").c_str(), 50, 0, 1.05);
    h.h_Eta_DispWeighted    = new TH1F((prefix + "_h_Eta_DispWeighted").c_str(), (prefix + " #eta dispersion (energy-weighted);D_{#eta}^{w};Counts").c_str(), 60, 0, 0.6);
    h.h_Phi_DispWeighted    = new TH1F((prefix + "_h_Phi_DispWeighted").c_str(), (prefix + " #phi dispersion (energy-weighted);D_{#phi}^{w} [rad];Counts").c_str(), 60, 0, 3);

    return h;
}

// =============================================================================
// =========================  ToF SIDE  =======================================
// =============================================================================

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

// Ta sama logika dR + charge*dPhi + RK4 distance_to_TOF co wczesniej;
// zamiast TTreeReaderArray dostaje kolekcje edm4eic::TrackerHit.
TrackToFFeatures ComputeToFFeatures(
    const TLorentzVector &Partic, int charge,
    const edm4eic::TrackerHitCollection &barrelHits,
    const edm4eic::TrackerHitCollection &endcapHits,
    double dR_cut_barrel, double dR_cut_endcap,
    double dist_cut_barrel, double dist_cut_endcap,
    TH1D *hDRBarrel = nullptr, TH1D *hDistBarrel = nullptr,
    TH1D *hDREndcap = nullptr, TH1D *hDistEndcap = nullptr)
{
    TrackToFFeatures f;

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
        if (hDRBarrel) hDRBarrel->Fill(dR);

        if (dR < dR_cut_barrel && charge * dPhi < 0) {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);
            if (hDistBarrel) hDistBarrel->Fill(tof.distance_to_TOF);
            if (tof.distance_to_TOF < dist_cut_barrel) {
                double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                matchedBarrel.push_back({hit.getTime(), length});
                sumLenBarrel += length;
                if (tof.distance_to_TOF < minDistBarrel) minDistBarrel = tof.distance_to_TOF;
            }
        }
    }

    for (const auto &hit : endcapHits) {
        auto p = hit.getPosition();
        TVector3 ToFPos(p.x, p.y, p.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);
        if (hDREndcap) hDREndcap->Fill(dR);

        if (dR < dR_cut_endcap && charge * dPhi < 0) {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);
            if (hDistEndcap) hDistEndcap->Fill(tof.distance_to_TOF);
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

// =============================================================================
// ================================  MAIN  ====================================
// =============================================================================

void TrainingPodioMacro()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    static constexpr int NumOfFiles = 2;
    std::vector<string> filePatterns(NumOfFiles);
    std::vector<int>    filePDG   = {13, 211};
    std::vector<string> fileName  = {"Muons", "Pions"};

    filePatterns.at(0) = "/run/media/epic/Data/Background/Muons/Continuous/reco*.root";
    filePatterns.at(1) = "/run/media/epic/Data/Background/Pions/Continuous/reco*.root";

    // ToF matching cuts (bez zmian)
    double dR_cut_barrel   = 0.8;
    double dR_cut_endcap   = 0.8;
    double dist_cut_barrel = 6.0;
    double dist_cut_endcap = 6.0;
    double pMax            = 1.0;

    const double TIMING_CUT_NS = 20.0;
    const long   MAX_EVENTS    = -1;   // -1 = wszystkie

    vector<string> ecalClusterNames = {"EcalBarrelImagingClusters", "EcalBarrelScFiClusters", "EcalEndcapPClusters", "EcalEndcapNClusters"};
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    // ---- ML tree ----------------------------------------------------------
    float ECalEnergy, ECalNumber, ECalEoverP, ECalAvgHitEnergy, ECalSpreadPhi, ECalSpreadEta, ECalSpreadR,
          ECalMaxHitFrac, ECalEnergyStdDev, ECalEnergyConcentration, ECalR_Disp, ECalR_DispWeighted,
          ECalEta_DispWeighted, ECalPhi_DispWeighted;

    float HCalEnergy, HCalNumber, HCalEoverP, HCalAvgHitEnergy, HCalSpreadPhi, HCalSpreadEta, HCalSpreadR,
          HCalMaxHitFrac, HCalEnergyStdDev, HCalEnergyConcentration, HCalR_Disp, HCalR_DispWeighted,
          HCalEta_DispWeighted, HCalPhi_DispWeighted;

    float ToFBeta, ToFMassSq, ToFNHitsBarrel, ToFNHitsEndcap, ToFNHitsTotal;
    float ToFMinDistBarrel, ToFMinDistEndcap, ToFAvgLenBarrel, ToFAvgLenEndcap, ToFHasToF;

    float TrackMomentum, TrackEta, TrackPhi, TrackCharge;
    float TruePDG, FileIndex;

    TFile *mlFile = new TFile("ONNX/MLDataCaloToF.root", "RECREATE");
    TTree *MLDataTree = new TTree("MLDataTree", "MLDataTree");

    MLDataTree->Branch("ECalEnergy", &ECalEnergy, "ECalEnergy/F");
    MLDataTree->Branch("ECalNumber", &ECalNumber, "ECalNumber/F");
    MLDataTree->Branch("ECalEoverP", &ECalEoverP, "ECalEoverP/F");
    MLDataTree->Branch("ECalAvgHitEnergy", &ECalAvgHitEnergy, "ECalAvgHitEnergy/F");
    MLDataTree->Branch("ECalSpreadPhi", &ECalSpreadPhi, "ECalSpreadPhi/F");
    MLDataTree->Branch("ECalSpreadEta", &ECalSpreadEta, "ECalSpreadEta/F");
    MLDataTree->Branch("ECalSpreadR", &ECalSpreadR, "ECalSpreadR/F");
    MLDataTree->Branch("ECalMaxHitFrac", &ECalMaxHitFrac, "ECalMaxHitFrac/F");
    MLDataTree->Branch("ECalEnergyStdDev", &ECalEnergyStdDev, "ECalEnergyStdDev/F");
    MLDataTree->Branch("ECalEnergyConcentration", &ECalEnergyConcentration, "ECalEnergyConcentration/F");
    MLDataTree->Branch("ECalR_Disp", &ECalR_Disp, "ECalR_Disp/F");
    MLDataTree->Branch("ECalR_DispWeighted", &ECalR_DispWeighted, "ECalR_DispWeighted/F");
    MLDataTree->Branch("ECalEta_DispWeighted", &ECalEta_DispWeighted, "ECalEta_DispWeighted/F");
    MLDataTree->Branch("ECalPhi_DispWeighted", &ECalPhi_DispWeighted, "ECalPhi_DispWeighted/F");

    MLDataTree->Branch("HCalEnergy", &HCalEnergy, "HCalEnergy/F");
    MLDataTree->Branch("HCalNumber", &HCalNumber, "HCalNumber/F");
    MLDataTree->Branch("HCalEoverP", &HCalEoverP, "HCalEoverP/F");
    MLDataTree->Branch("HCalAvgHitEnergy", &HCalAvgHitEnergy, "HCalAvgHitEnergy/F");
    MLDataTree->Branch("HCalSpreadPhi", &HCalSpreadPhi, "HCalSpreadPhi/F");
    MLDataTree->Branch("HCalSpreadEta", &HCalSpreadEta, "HCalSpreadEta/F");
    MLDataTree->Branch("HCalSpreadR", &HCalSpreadR, "HCalSpreadR/F");
    MLDataTree->Branch("HCalMaxHitFrac", &HCalMaxHitFrac, "HCalMaxHitFrac/F");
    MLDataTree->Branch("HCalEnergyStdDev", &HCalEnergyStdDev, "HCalEnergyStdDev/F");
    MLDataTree->Branch("HCalEnergyConcentration", &HCalEnergyConcentration, "HCalEnergyConcentration/F");
    MLDataTree->Branch("HCalR_Disp", &HCalR_Disp, "HCalR_Disp/F");
    MLDataTree->Branch("HCalR_DispWeighted", &HCalR_DispWeighted, "HCalR_DispWeighted/F");
    MLDataTree->Branch("HCalEta_DispWeighted", &HCalEta_DispWeighted, "HCalEta_DispWeighted/F");
    MLDataTree->Branch("HCalPhi_DispWeighted", &HCalPhi_DispWeighted, "HCalPhi_DispWeighted/F");

    MLDataTree->Branch("ToFBeta", &ToFBeta, "ToFBeta/F");
    MLDataTree->Branch("ToFMassSq", &ToFMassSq, "ToFMassSq/F");
    MLDataTree->Branch("ToFNHitsBarrel", &ToFNHitsBarrel, "ToFNHitsBarrel/F");
    MLDataTree->Branch("ToFNHitsEndcap", &ToFNHitsEndcap, "ToFNHitsEndcap/F");
    MLDataTree->Branch("ToFNHitsTotal", &ToFNHitsTotal, "ToFNHitsTotal/F");
    MLDataTree->Branch("ToFMinDistBarrel", &ToFMinDistBarrel, "ToFMinDistBarrel/F");
    MLDataTree->Branch("ToFMinDistEndcap", &ToFMinDistEndcap, "ToFMinDistEndcap/F");
    MLDataTree->Branch("ToFAvgLenBarrel", &ToFAvgLenBarrel, "ToFAvgLenBarrel/F");
    MLDataTree->Branch("ToFAvgLenEndcap", &ToFAvgLenEndcap, "ToFAvgLenEndcap/F");
    MLDataTree->Branch("ToFHasToF", &ToFHasToF, "ToFHasToF/F");

    MLDataTree->Branch("TrackMomentum", &TrackMomentum, "TrackMomentum/F");
    MLDataTree->Branch("TrackEta", &TrackEta, "TrackEta/F");
    MLDataTree->Branch("TrackPhi", &TrackPhi, "TrackPhi/F");
    MLDataTree->Branch("TrackCharge", &TrackCharge, "TrackCharge/F");
    MLDataTree->Branch("TruePDG", &TruePDG, "TruePDG/F");
    MLDataTree->Branch("FileIndex", &FileIndex, "FileIndex/F");
    // XGBoost: -999 = brak danych (brak ToF / brak klastra) -> missing=-999 przy budowie DMatrix.

    // Pusta kolekcja uzywana, gdy kolekcji ToF nie ma w pliku.
    static const edm4eic::TrackerHitCollection emptyToF;

    for (int File = 0; File < NumOfFiles; File++)
    {
        const string &name = fileName[File];
        int truePDG = filePDG[File];

        vector<string> fileList = ExpandGlob(filePatterns.at(File));
        if (fileList.empty()) {
            cerr << "[ERROR] No files match: " << filePatterns.at(File) << endl;
            continue;
        }

        podio::ROOTReader reader;
        reader.openFiles(fileList);

        unsigned nEntries = reader.getEntries("events");
        cout << "Total events in chain (" << name << "): " << nEntries << " (" << fileList.size() << " files)" << endl;

        cout << "Building ECal detector group:" << endl;
        vector<DetectorAssoc> ecalDetectors;
        for (auto &n : ecalClusterNames) ecalDetectors.push_back(MakeDetectorAssoc(n));

        cout << "Building HCal detector group:" << endl;
        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

        bool hasBToF = true, hasEToF = true;

        // -------- per-file diagnostic output --------
        TFile *outfile = new TFile(Form("Plots/CaloToF/Diag_%s.root", name.c_str()), "RECREATE");

        HistSet ecalHists = MakeHistSet("ECal");
        HistSet hcalHists = MakeHistSet("HCal");

        TH1D *hMass       = new TH1D(Form("Mass_%s", name.c_str()), Form("Mass %s;m [GeV];Counts", name.c_str()), 50, 0, 0.2);
        TH1D *hMassSq     = new TH1D(Form("MassSq_%s", name.c_str()), Form("MassSq %s;m^{2} [GeV^{2}];Counts", name.c_str()), 50, -0.04, 0.08);
        TH2D *hBetaVsMom  = new TH2D(Form("BetaVsMom_%s", name.c_str()), Form("BetaVsMom %s;p [GeV];#beta", name.c_str()), 100, 0, 2, 100, 0.7, 1.1);
        TH1D *hDRBarrel   = new TH1D(Form("dRBarrel_%s", name.c_str()), Form("dR Barrel %s;#DeltaR;Counts", name.c_str()), 200, 0, 2);
        TH1D *hDistBarrel = new TH1D(Form("DistBarrel_%s", name.c_str()), Form("Dist Barrel %s;dist [mm];Counts", name.c_str()), 200, 0, 100);
        TH1D *hDREndcap   = new TH1D(Form("dREndcap_%s", name.c_str()), Form("dR Endcap %s;#DeltaR;Counts", name.c_str()), 200, 0, 20);
        TH1D *hDistEndcap = new TH1D(Form("DistEndcap_%s", name.c_str()), Form("Dist Endcap %s;dist [mm];Counts", name.c_str()), 200, 0, 100);

        long nToProcess  = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : nEntries;
        long nMisaligned = 0;   // kontrola: track[i] <-> trackAssoc[i]

        for (long entry = 0; entry < nToProcess; ++entry)
        {
            if ((entry + 1) % 50000 == 0) cout << "File " << name << " and event number... " << entry + 1 << endl;

            podio::Frame frame(reader.readEntry("events", entry));

            if (entry == 0) {
                cout << "Checking collections in first event:" << endl;
                CheckAvailability(ecalDetectors, frame);
                CheckAvailability(hcalDetectors, frame);
                auto names = frame.getAvailableCollections();
                hasBToF = HasCollection(names, "TOFBarrelRecHits");
                hasEToF = HasCollection(names, "TOFEndcapRecHits");
                cout << "  TOFBarrelRecHits: " << (hasBToF ? "OK" : "MISSING") << endl;
                cout << "  TOFEndcapRecHits: " << (hasEToF ? "OK" : "MISSING") << endl;
            }

            const auto &tracks =
                frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedChargedParticles");
            const auto &trackAssocs =
                frame.get<edm4eic::MCRecoParticleAssociationCollection>("ReconstructedChargedParticleAssociations");

            const auto &barrelToF = hasBToF ? frame.get<edm4eic::TrackerHitCollection>("TOFBarrelRecHits") : emptyToF;
            const auto &endcapToF = hasEToF ? frame.get<edm4eic::TrackerHitCollection>("TOFEndcapRecHits") : emptyToF;

            for (size_t particle = 0; particle < tracks.size(); ++particle)
            {
                const auto rcp = tracks[particle];

                auto mom = rcp.getMomentum();
                TLorentzVector Partic;
                Partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
                double trackP = Partic.P();

                if (trackP <= 0 || trackP > pMax) continue;

                int charge = (int)rcp.getCharge();

                if (particle >= trackAssocs.size()) continue;
                if (trackAssocs[particle].getRec() != rcp) nMisaligned++;
                const auto simPart = trackAssocs[particle].getSim();

                // -------- calorimeters --------
                TrackCalHits ecalHits = CollectHits(simPart, ecalDetectors, frame, Partic, TIMING_CUT_NS,
                    ecalHists.HitR, ecalHists.hHitEtaR, ecalHists.hHitPhiR, ecalHists.hHitEtaPhi,
                    ecalHists.HitPhi, ecalHists.HitZ, ecalHists.HitEnergy, ecalHists.HitTime,
                    ecalHists.DeltaEta, ecalHists.DeltaPhi);
                TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP, ecalHists);

                TrackCalHits hcalHits = CollectHits(simPart, hcalDetectors, frame, Partic, TIMING_CUT_NS,
                    hcalHists.HitR, hcalHists.hHitEtaR, hcalHists.hHitPhiR, hcalHists.hHitEtaPhi,
                    hcalHists.HitPhi, hcalHists.HitZ, hcalHists.HitEnergy, hcalHists.HitTime,
                    hcalHists.DeltaEta, hcalHists.DeltaPhi);
                TrackFeatures hcalF = ComputeTrackFeatures(hcalHits, trackP, hcalHists);

                // -------- ToF --------
                TrackToFFeatures tofF = ComputeToFFeatures(
                    Partic, charge, barrelToF, endcapToF,
                    dR_cut_barrel, dR_cut_endcap, dist_cut_barrel, dist_cut_endcap,
                    hDRBarrel, hDistBarrel, hDREndcap, hDistEndcap);

                if (tofF.HasToF > 0.5) {
                    hMassSq->Fill(tofF.MassSq);
                    if (trackP < 0.5) hMass->Fill(sqrt(tofF.MassSq));
                    hBetaVsMom->Fill(trackP, tofF.Beta);
                }

                if (ecalF.Number <= 0 && hcalF.Number <= 0 && tofF.HasToF < 0.5) continue;

                bool ecalHasHits = ecalF.Number > 0;
                bool hcalHasHits = hcalF.Number > 0;

                ECalEnergy              = ecalF.Energy;
                ECalNumber              = ecalF.Number;
                ECalEoverP              = ecalHasHits ? ecalF.EoverP : -999.f;
                ECalAvgHitEnergy        = ecalHasHits ? ecalF.AvgHitEnergy : -999.f;
                ECalSpreadPhi           = ecalHasHits ? ecalF.SpreadPhi : -999.f;
                ECalSpreadEta           = ecalHasHits ? ecalF.SpreadEta : -999.f;
                ECalSpreadR             = ecalHasHits ? ecalF.SpreadR : -999.f;
                ECalMaxHitFrac          = ecalHasHits ? ecalF.MaxHitFrac : -999.f;
                ECalEnergyStdDev        = ecalHasHits ? ecalF.EnergyStdDev : -999.f;
                ECalEnergyConcentration = ecalHasHits ? ecalF.EnergyConcentration : -999.f;
                ECalR_Disp              = ecalHasHits ? ecalF.R_Disp : -999.f;
                ECalR_DispWeighted      = ecalHasHits ? ecalF.R_DispWeighted : -999.f;
                ECalEta_DispWeighted    = ecalHasHits ? ecalF.Eta_DispWeighted : -999.f;
                ECalPhi_DispWeighted    = ecalHasHits ? ecalF.Phi_DispWeighted : -999.f;

                HCalEnergy              = hcalF.Energy;
                HCalNumber              = hcalF.Number;
                HCalEoverP              = hcalHasHits ? hcalF.EoverP : -999.f;
                HCalAvgHitEnergy        = hcalHasHits ? hcalF.AvgHitEnergy : -999.f;
                HCalSpreadPhi           = hcalHasHits ? hcalF.SpreadPhi : -999.f;
                HCalSpreadEta           = hcalHasHits ? hcalF.SpreadEta : -999.f;
                HCalSpreadR             = hcalHasHits ? hcalF.SpreadR : -999.f;
                HCalMaxHitFrac          = hcalHasHits ? hcalF.MaxHitFrac : -999.f;
                HCalEnergyStdDev        = hcalHasHits ? hcalF.EnergyStdDev : -999.f;
                HCalEnergyConcentration = hcalHasHits ? hcalF.EnergyConcentration : -999.f;
                HCalR_Disp              = hcalHasHits ? hcalF.R_Disp : -999.f;
                HCalR_DispWeighted      = hcalHasHits ? hcalF.R_DispWeighted : -999.f;
                HCalEta_DispWeighted    = hcalHasHits ? hcalF.Eta_DispWeighted : -999.f;
                HCalPhi_DispWeighted    = hcalHasHits ? hcalF.Phi_DispWeighted : -999.f;

                ToFBeta          = tofF.Beta;
                ToFMassSq        = tofF.MassSq;
                ToFNHitsBarrel   = tofF.NHitsBarrel;
                ToFNHitsEndcap   = tofF.NHitsEndcap;
                ToFNHitsTotal    = tofF.NHitsTotal;
                ToFMinDistBarrel = tofF.MinDistBarrel;
                ToFMinDistEndcap = tofF.MinDistEndcap;
                ToFAvgLenBarrel  = tofF.AvgLenBarrel;
                ToFAvgLenEndcap  = tofF.AvgLenEndcap;
                ToFHasToF        = tofF.HasToF;

                TrackMomentum = (float)trackP;
                TrackEta      = (float)Partic.Eta();
                TrackPhi      = (float)Partic.Phi();
                TrackCharge   = (float)charge;
                TruePDG       = (float)truePDG;
                FileIndex     = (float)File;

                MLDataTree->Fill();
            }
        }

        cout << "===========================" << endl;
        cout << "End of " << name << " file" << endl;
        cout << "Number of events processed: " << nToProcess << endl;
        cout << "Tracks with track<->assoc index misalignment: " << nMisaligned << " (powinno byc 0)" << endl;
        cout << "===========================" << endl;

        outfile->Write();
        outfile->Close();
    }

    mlFile->cd();
    MLDataTree->Write();
    mlFile->Close();
}