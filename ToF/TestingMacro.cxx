// TestingMacro_Fixed.cxx
//
// Self-contained TestingMacro that builds the SAME raw-feature input expected
// by the ONNX preprocessor used in the Python CombinedCaloToFAnalysis pipeline.
// This file merges the helper definitions (calorimeter and ToF feature
// computation) and the ONNX runtime inference loop into one translation unit
// so that types like TrackToFFeatures and ComputeToFFeatures are visible
// before they are used (fixes the "unknown type name 'TrackToFFeatures'"
// compile error).
//
// Usage: compile with ROOT and ONNX Runtime available on the include/link path.
// Example (rough):
//   g++ -O2 -std=c++17 TestingMacro_Fixed.cxx `root-config --cflags --libs` -lonnxruntime -o TestingMacro_Fixed
//
// Make sure ToFSim.cxx (and any field-map dependencies) are available and
// included in the same directory or adjust the include path accordingly.

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TROOT.h>
#include <TChain.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TMath.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>

using namespace std;

// -----------------------------------------------------------------------------
// Include ToFSim implementation (must be present in the same directory).
// This brings in ToFSim(), ToFResults, c_light, etc.
// -----------------------------------------------------------------------------
#include "ToFSim.cxx"

// -----------------------------------------------------------------------------
// Utility: combine multiple ToF hits into a single beta estimate.
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

// -----------------------------------------------------------------------------
// Calorimeter helper types and functions (from TrainingMacro.cxx).
// -----------------------------------------------------------------------------
struct DetectorHits
{
    string clusterName;
    unique_ptr<TTreeReaderArray<int>>          simuAssocCluster;
    unique_ptr<TTreeReaderArray<unsigned int>> hitsBegin;
    unique_ptr<TTreeReaderArray<unsigned int>> hitsEnd;
    unique_ptr<TTreeReaderArray<int>>          hitsIndex;
    unique_ptr<TTreeReaderArray<float>>        hitEnergy;
    unique_ptr<TTreeReaderArray<float>>        hitTime;
    unique_ptr<TTreeReaderArray<float>>        hitPosX;
    unique_ptr<TTreeReaderArray<float>>        hitPosY;
    unique_ptr<TTreeReaderArray<float>>        hitPosZ;
};

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

DetectorHits MakeDetectorHits(TTreeReader &reader, const string &clusterName)
{
    DetectorHits d;
    d.clusterName = clusterName;

    string recHitsName = clusterName;
    size_t pos = recHitsName.rfind("Clusters");
    if (pos != string::npos) recHitsName.replace(pos, string("Clusters").size(), "RecHits");

    string assocName     = "_" + clusterName.substr(0, clusterName.size() - 1) + "Associations_sim.index";
    string hitsIndexName = "_" + clusterName + "_hits.index";
    string hitsBeginName = clusterName + ".hits_begin";
    string hitsEndName   = clusterName + ".hits_end";

    d.simuAssocCluster = make_unique<TTreeReaderArray<int>>(reader, assocName.c_str());
    d.hitsBegin        = make_unique<TTreeReaderArray<unsigned int>>(reader, hitsBeginName.c_str());
    d.hitsEnd          = make_unique<TTreeReaderArray<unsigned int>>(reader, hitsEndName.c_str());
    d.hitsIndex        = make_unique<TTreeReaderArray<int>>(reader, hitsIndexName.c_str());
    d.hitEnergy        = make_unique<TTreeReaderArray<float>>(reader, (recHitsName + ".energy").c_str());
    d.hitTime          = make_unique<TTreeReaderArray<float>>(reader, (recHitsName + ".time").c_str());
    d.hitPosX          = make_unique<TTreeReaderArray<float>>(reader, (recHitsName + ".position.x").c_str());
    d.hitPosY          = make_unique<TTreeReaderArray<float>>(reader, (recHitsName + ".position.y").c_str());
    d.hitPosZ          = make_unique<TTreeReaderArray<float>>(reader, (recHitsName + ".position.z").c_str());

    cout << "  Detector '" << clusterName << "' -> RecHits '" << recHitsName
         << "', Assoc '" << assocName << "'" << endl;
    return d;
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

TrackCalHits CollectHits(int simuID, vector<DetectorHits> &detectors, const TLorentzVector &Partic,
                          double timingCutNs,
                          TH1F *hR, TH2F *hEtaR, TH2F *hPhiR, TH2F *hEtaPhi,
                          TH1F *hPhi, TH1F *hZ, TH1F *hE, TH1F *hT,
                          TH1F *hDEta, TH1F *hDPhi)
{
    TrackCalHits result;

    for (auto &det : detectors)
    {
        size_t nClusters = det.simuAssocCluster->GetSize();
        for (size_t iCluster = 0; iCluster < nClusters; ++iCluster)
        {
            if (simuID != (*det.simuAssocCluster)[iCluster]) continue;

            unsigned int begin = (*det.hitsBegin)[iCluster];
            unsigned int end   = (*det.hitsEnd)[iCluster];
            if (static_cast<int>(end - begin) <= 0) continue;

            for (unsigned int i = begin; i < end; ++i)
            {
                int hitIndex = (*det.hitsIndex)[i];

                float HitE = (*det.hitEnergy)[hitIndex];
                float Hitx = (*det.hitPosX)[hitIndex];
                float Hity = (*det.hitPosY)[hitIndex];
                float Hitz = (*det.hitPosZ)[hitIndex];
                float HitT = (*det.hitTime)[hitIndex];

                if (hT) hT->Fill(HitT);
                if (HitT > timingCutNs) continue;

                TVector3 hitVec(Hitx, Hity, Hitz);
                double hitEta = hitVec.Eta();
                double hitPhi = hitVec.Phi();
                double R = sqrt(Hitx * Hitx + Hity * Hity);

                if (hR)     hR->Fill(R);
                if (hEtaR)  hEtaR->Fill(hitEta, R);
                if (hPhiR)  hPhiR->Fill(hitPhi, R);
                if (hEtaPhi) hEtaPhi->Fill(hitEta, hitPhi);
                if (hPhi)   hPhi->Fill(hitPhi);
                if (hZ)     hZ->Fill(Hitz);
                if (hE)     hE->Fill(HitE);

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
    float Number = 0.f;          // N hits
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

    // --- R dispersion (unweighted + energy-weighted), around energy-weighted mean R ---
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

    // --- pack into the return struct for the TTree ---
    f.Energy               = static_cast<float>(sumEnergy);
    f.Number               = static_cast<float>(n);
    f.EoverP                = static_cast<float>(sumEnergy / trackP);
    f.AvgHitEnergy           = static_cast<float>(meanE);
    f.SpreadPhi              = static_cast<float>(spreadPhi);
    f.SpreadEta              = static_cast<float>(spreadEta);
    f.SpreadR                = static_cast<float>(spreadR);
    f.MaxHitFrac             = static_cast<float>(hits.maxHitE / sumEnergy);
    f.EnergyStdDev           = static_cast<float>(energyStdDev);
    f.EnergyConcentration    = static_cast<float>(energyConcentration);
    f.R_Disp                 = static_cast<float>(R_disp_unweighted);
    f.R_DispWeighted          = static_cast<float>(R_disp_weighted);
    f.Eta_DispWeighted        = static_cast<float>(Eta_disp_weighted);
    f.Phi_DispWeighted        = static_cast<float>(Phi_disp_weighted);

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
// ToF helper types and functions (from IDAnalysis.cxx). These are declared
// before use to avoid the compile error reported earlier.
// -----------------------------------------------------------------------------
struct TrackToFFeatures
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

TrackToFFeatures ComputeToFFeatures(
    const TLorentzVector &Partic, int charge,
    TTreeReaderArray<float> &BToFPosX, TTreeReaderArray<float> &BToFPosY,
    TTreeReaderArray<float> &BToFPosZ, TTreeReaderArray<float> &BToFTime,
    TTreeReaderArray<float> &EToFPosX, TTreeReaderArray<float> &EToFPosY,
    TTreeReaderArray<float> &EToFPosZ, TTreeReaderArray<float> &EToFTime,
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

    for (int i = 0; i < BToFTime.GetSize(); ++i) {
        TVector3 ToFPos(BToFPosX[i], BToFPosY[i], BToFPosZ[i]);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);
        if (hDRBarrel) hDRBarrel->Fill(dR);

        if (dR < dR_cut_barrel && charge * dPhi < 0) {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);
            if (hDistBarrel) hDistBarrel->Fill(tof.distance_to_TOF);
            if (tof.distance_to_TOF < dist_cut_barrel) {
                double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                matchedBarrel.push_back({BToFTime[i], length});
                sumLenBarrel += length;
                if (tof.distance_to_TOF < minDistBarrel) minDistBarrel = tof.distance_to_TOF;
            }
        }
    }

    for (int i = 0; i < EToFTime.GetSize(); ++i) {
        TVector3 ToFPos(EToFPosX[i], EToFPosY[i], EToFPosZ[i]);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);
        if (hDREndcap) hDREndcap->Fill(dR);

        if (dR < dR_cut_endcap && charge * dPhi < 0) {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);
            if (hDistEndcap) hDistEndcap->Fill(tof.distance_to_TOF);
            if (tof.distance_to_TOF < dist_cut_endcap) {
                double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                matchedEndcap.push_back({EToFTime[i], length});
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

// -----------------------------------------------------------------------------
// ONNX runtime helper: run the merged ONNX model (preprocessor + XGBoost).
// The ONNX graph expects a single input named "raw_features" with shape (N,41)
// (full raw feature set produced by the Python preprocessor) and returns
// "probabilities" (softmax) or similar; we read P(muon) = probs[1].
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
    // Defensive: if model returns a single score, user may need to adapt.
    return probs[1];
}

// -----------------------------------------------------------------------------
// Build raw feature vector in the exact order used by the Python RAW_COLS.
// -----------------------------------------------------------------------------
static const int N_RAW = 41;
static const float MISSING_SENTINEL = -999.f;

// Build the full 41-element raw feature vector in the exact order used by
// `PyGym.py` RAW_COLS. For calo-derived fields that are undefined when no
// cluster/hits exist we use MISSING_SENTINEL.
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
    std::vector<float> raw;
    raw.reserve(N_RAW);

    // ECal
    raw.push_back(ECalEnergy);
    raw.push_back(ECalNumber);
    raw.push_back(ECalEoverP);
    raw.push_back(ECalAvgHitEnergy);
    raw.push_back(ECalSpreadPhi);
    raw.push_back(ECalSpreadEta);
    raw.push_back(ECalSpreadR);
    raw.push_back(ECalMaxHitFrac);
    raw.push_back(ECalEnergyStdDev);
    raw.push_back(ECalEnergyConcentration);
    raw.push_back(ECalR_Disp);
    raw.push_back(ECalR_DispWeighted);
    raw.push_back(ECalEta_DispWeighted);
    raw.push_back(ECalPhi_DispWeighted);

    // HCal
    raw.push_back(HCalEnergy);
    raw.push_back(HCalNumber);
    raw.push_back(HCalEoverP);
    raw.push_back(HCalAvgHitEnergy);
    raw.push_back(HCalSpreadPhi);
    raw.push_back(HCalSpreadEta);
    raw.push_back(HCalSpreadR);
    raw.push_back(HCalMaxHitFrac);
    raw.push_back(HCalEnergyStdDev);
    raw.push_back(HCalEnergyConcentration);
    raw.push_back(HCalR_Disp);
    raw.push_back(HCalR_DispWeighted);
    raw.push_back(HCalEta_DispWeighted);
    raw.push_back(HCalPhi_DispWeighted);

    // ToF
    raw.push_back(ToFBeta);
    raw.push_back(ToFMassSq);
    raw.push_back(ToFNHitsBarrel);
    raw.push_back(ToFNHitsEndcap);
    raw.push_back(ToFNHitsTotal);
    raw.push_back(ToFMinDistBarrel);
    raw.push_back(ToFMinDistEndcap);
    raw.push_back(ToFAvgLenBarrel);
    raw.push_back(ToFAvgLenEndcap);
    raw.push_back(ToFHasToF);

    // Track
    raw.push_back(TrackMomentum);
    raw.push_back(TrackEta);
    raw.push_back(TrackPhi);

    return raw;
}

// -----------------------------------------------------------------------------
// Main testing macro: iterate over muon/pion files, compute features, run ONNX,
// fill histograms for response, efficiency and rejection vs p, pT, eta.
// -----------------------------------------------------------------------------
void TestingMacro()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    const double TIMING_CUT_NS = 20.0;
    const float MUON_ID_CUT = 0.5f;

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load merged ONNX model (preprocessor + XGBoost)
    Ort::Session session(env, "ONNX/xgb_muonID.onnx", session_options);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    static constexpr int NumOfFiles = 2;
    vector<TString> files(NumOfFiles);
    files.at(0) = "/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
    //files.at(0)="/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_1*.root";
    files.at(1) = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
    //files.at(1)="/run/media/epic/Data/Tau/reco/Energy_10x275/old/double_pi/recoDoublePi.root";

    vector<string> ecalClusterNames = {"EcalBarrelImagingClusters","EcalBarrelScFiClusters", "EcalEndcapPClusters", "EcalEndcapNClusters"};
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    // Output file for histograms
    TFile *outfile = new TFile("Plots/MuonID_Performance_fromPythonFeatures.root", "RECREATE");

    // Response histograms
    TH1F *h_Response_Muon = new TH1F("h_Response_Muon", "P(muon) Response for True Muons;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Pion = new TH1F("h_Response_Pion", "P(muon) Response for True Pions;P(muon);Counts", 100, 0, 1);

    // Efficiency / rejection counters vs pT, p, eta
    TH1F *h_Muon_Total_vs_Pt = new TH1F("h_Muon_Total_vs_Pt", "Muon Total vs pT;p_{T} [GeV/c];Counts",40,0,2);
    TH1F *h_Muon_Passed_vs_Pt = new TH1F("h_Muon_Passed_vs_Pt", "Muon Passed vs pT;p_{T} [GeV/c];Counts",40,0,2);

    TH1F *h_Muon_Total_vs_P  = new TH1F("h_Muon_Total_vs_P", "Muon Total vs p;p [GeV/c];Counts",40,0,2);
    TH1F *h_Muon_Passed_vs_P = new TH1F("h_Muon_Passed_vs_P", "Muon Passed vs p;p [GeV/c];Counts",40,0,2);

    TH1F *h_Muon_Total_vs_Eta  = new TH1F("h_Muon_Total_vs_Eta", "Muon Total vs #eta;#eta;Counts",30,-1.2,3.4);
    TH1F *h_Muon_Passed_vs_Eta = new TH1F("h_Muon_Passed_vs_Eta", "Muon Passed vs #eta;#eta;Counts",30,-1.2,3.4);

    TH1F *h_Pion_Total_vs_Pt   = new TH1F("h_Pion_Total_vs_Pt", "Pion Total vs pT;p_{T} [GeV/c];Counts",40,0,2);
    TH1F *h_Pion_Rejected_vs_Pt = new TH1F("h_Pion_Rejected_vs_Pt", "Pion Rejected vs pT;p_{T} [GeV/c];Counts",40,0,2);

    TH1F *h_Pion_Total_vs_P    = new TH1F("h_Pion_Total_vs_P", "Pion Total vs p;p [GeV/c];Counts",40,0,2);
    TH1F *h_Pion_Rejected_vs_P = new TH1F("h_Pion_Rejected_vs_P", "Pion Rejected vs p;p [GeV/c];Counts",40,0,2);

    TH1F *h_Pion_Total_vs_Eta  = new TH1F("h_Pion_Total_vs_Eta", "Pion Total vs #eta;#eta;Counts",30,-1.2,3.4);
    TH1F *h_Pion_Rejected_vs_Eta = new TH1F("h_Pion_Rejected_vs_Eta", "Pion Rejected vs #eta;#eta;Counts",30,-1.2,3.4);

    // Loop over files (muons then pions)
    for (int File = 0; File < NumOfFiles; File++)
    {
        bool isMuonFile = (File == 0);
        string name = isMuonFile ? "Muons" : "Pions";

        TChain *mychain = new TChain("events");
        mychain->Add(files.at(File));

        TTreeReader tree_reader(mychain);
        Long64_t nEvents = mychain->GetEntries();
        cout << "Total events in chain (" << name << "): " << nEvents << endl;

        // Tracking branches
        TTreeReaderArray<float> trackMomX(tree_reader, "ReconstructedChargedParticles.momentum.x");
        TTreeReaderArray<float> trackMomY(tree_reader, "ReconstructedChargedParticles.momentum.y");
        TTreeReaderArray<float> trackMomZ(tree_reader, "ReconstructedChargedParticles.momentum.z");
        TTreeReaderArray<float> trackEng(tree_reader, "ReconstructedChargedParticles.energy");
        TTreeReaderArray<float> trackCharge(tree_reader, "ReconstructedChargedParticles.charge");
        TTreeReaderArray<int>   simuAssoc(tree_reader, "_ReconstructedChargedParticleAssociations_sim.index");

        // ToF branches (if needed by ComputeToFFeatures)
        TTreeReaderArray<float> BToFPosX(tree_reader, "TOFBarrelRecHits.position.x");
        TTreeReaderArray<float> BToFPosY(tree_reader, "TOFBarrelRecHits.position.y");
        TTreeReaderArray<float> BToFPosZ(tree_reader, "TOFBarrelRecHits.position.z");
        TTreeReaderArray<float> BToFTime(tree_reader, "TOFBarrelRecHits.time");

        TTreeReaderArray<float> EToFPosX(tree_reader, "TOFEndcapRecHits.position.x");
        TTreeReaderArray<float> EToFPosY(tree_reader, "TOFEndcapRecHits.position.y");
        TTreeReaderArray<float> EToFPosZ(tree_reader, "TOFEndcapRecHits.position.z");
        TTreeReaderArray<float> EToFTime(tree_reader, "TOFEndcapRecHits.time");

        // Build detector groups for calo
        vector<DetectorHits> ecalDetectors;
        for (auto &n : ecalClusterNames) ecalDetectors.push_back(MakeDetectorHits(tree_reader, n));

        vector<DetectorHits> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorHits(tree_reader, n));

        int eventID = 0;

        while (tree_reader.Next())
        {
            eventID++;
            if (eventID % 50000 == 0) cout << "File " << name << " event: " << eventID << endl;

            for (size_t particle = 0; particle < trackMomX.GetSize(); particle++)
            {
                TLorentzVector Partic;
                Partic.SetPxPyPzE(trackMomX[particle], trackMomY[particle], trackMomZ[particle], trackEng[particle]);
                double trackP  = Partic.P();
                double trackPt = Partic.Pt();
                double trackEta = Partic.Eta();
                double trackPhi = Partic.Phi();

                // Keep selection consistent with training macro
                if (trackP > 1.0) continue;
                if (trackEta >= 1.0 && trackEta <= 1.3) continue;
                if (trackEta <= -1.25) continue;

                int charge = static_cast<int>(trackCharge[particle]);
                int simuID = simuAssoc[particle];

                // Collect calorimeter hits and compute features
                HistSet ecalHists = MakeHistSet("ECal_tmp");
                HistSet hcalHists = MakeHistSet("HCal_tmp");
                TrackCalHits ecalHits = CollectHits(simuID, ecalDetectors, Partic, TIMING_CUT_NS,
                    ecalHists.HitR, ecalHists.HitEtaR, ecalHists.HitPhiR, ecalHists.HitEtaPhi,
                    ecalHists.HitPhi, ecalHists.HitZ, ecalHists.HitEnergy, ecalHists.HitTime,
                    ecalHists.DeltaEta, ecalHists.DeltaPhi);
                TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP, ecalHists);

                TrackCalHits hcalHits = CollectHits(simuID, hcalDetectors, Partic, TIMING_CUT_NS,
                    hcalHists.HitR, hcalHists.HitEtaR, hcalHists.HitPhiR, hcalHists.HitEtaPhi,
                    hcalHists.HitPhi, hcalHists.HitZ, hcalHists.HitEnergy, hcalHists.HitTime,
                    hcalHists.DeltaEta, hcalHists.DeltaPhi);
                TrackFeatures hcalF = ComputeTrackFeatures(hcalHits, trackP, hcalHists);

                // Compute ToF features
                TrackToFFeatures tofF = ComputeToFFeatures(
                    Partic, charge,
                    BToFPosX, BToFPosY, BToFPosZ, BToFTime,
                    EToFPosX, EToFPosY, EToFPosZ, EToFTime,
                    /*dR_cut_barrel*/0.8, /*dR_cut_endcap*/0.8,
                    /*dist_cut_barrel*/6.0, /*dist_cut_endcap*/6.0
                );

                // Drop tracks invisible everywhere (same logic as training)
                if (ecalF.Number <= 0 && hcalF.Number <= 0 && tofF.HasToF < 0.5) continue;


                // Map computed features -> raw fields expected by ONNX preprocessor
                // ECal: energy and number are real; the rest use the sentinel when no hits.
                float ECalEnergy     = ecalF.Energy;
                float ECalNumber     = ecalF.Number;
                float ECalEoverP     = (ecalF.Number > 0) ? ecalF.EoverP : MISSING_SENTINEL;
                float ECalAvgHitEnergy = (ecalF.Number > 0) ? ecalF.AvgHitEnergy : MISSING_SENTINEL;
                float ECalSpreadPhi  = (ecalF.Number > 0) ? ecalF.SpreadPhi : MISSING_SENTINEL;
                float ECalSpreadEta  = (ecalF.Number > 0) ? ecalF.SpreadEta : MISSING_SENTINEL;
                float ECalSpreadR    = (ecalF.Number > 0) ? ecalF.SpreadR : MISSING_SENTINEL;
                float ECalMaxHitFrac = (ecalF.Number > 0) ? ecalF.MaxHitFrac : MISSING_SENTINEL;
                float ECalEnergyStdDev = (ecalF.Number > 0) ? ecalF.EnergyStdDev : MISSING_SENTINEL;
                float ECalEnergyConcentration = (ecalF.Number > 0) ? ecalF.EnergyConcentration : MISSING_SENTINEL;
                float ECalR_Disp = (ecalF.Number > 0) ? ecalF.R_Disp : MISSING_SENTINEL;
                float ECalR_DispWeighted = (ecalF.Number > 0) ? ecalF.R_DispWeighted : MISSING_SENTINEL;
                float ECalEta_DispWeighted = (ecalF.Number > 0) ? ecalF.Eta_DispWeighted : MISSING_SENTINEL;
                float ECalPhi_DispWeighted = (ecalF.Number > 0) ? ecalF.Phi_DispWeighted : MISSING_SENTINEL;

                float HCalEnergy     = hcalF.Energy;
                float HCalNumber     = hcalF.Number;
                float HCalEoverP     = (hcalF.Number > 0) ? hcalF.EoverP : MISSING_SENTINEL;
                float HCalAvgHitEnergy = (hcalF.Number > 0) ? hcalF.AvgHitEnergy : MISSING_SENTINEL;
                float HCalSpreadPhi  = (hcalF.Number > 0) ? hcalF.SpreadPhi : MISSING_SENTINEL;
                float HCalSpreadEta  = (hcalF.Number > 0) ? hcalF.SpreadEta : MISSING_SENTINEL;
                float HCalSpreadR    = (hcalF.Number > 0) ? hcalF.SpreadR : MISSING_SENTINEL;
                float HCalMaxHitFrac = (hcalF.Number > 0) ? hcalF.MaxHitFrac : MISSING_SENTINEL;
                float HCalEnergyStdDev = (hcalF.Number > 0) ? hcalF.EnergyStdDev : MISSING_SENTINEL;
                float HCalEnergyConcentration = (hcalF.Number > 0) ? hcalF.EnergyConcentration : MISSING_SENTINEL;
                float HCalR_Disp = (hcalF.Number > 0) ? hcalF.R_Disp : MISSING_SENTINEL;
                float HCalR_DispWeighted = (hcalF.Number > 0) ? hcalF.R_DispWeighted : MISSING_SENTINEL;
                float HCalEta_DispWeighted = (hcalF.Number > 0) ? hcalF.Eta_DispWeighted : MISSING_SENTINEL;
                float HCalPhi_DispWeighted = (hcalF.Number > 0) ? hcalF.Phi_DispWeighted : MISSING_SENTINEL;

                float ToFBeta           = (tofF.HasToF > 0.5f) ? tofF.Beta : MISSING_SENTINEL;
                float ToFMassSq         = (tofF.HasToF > 0.5f) ? tofF.MassSq : MISSING_SENTINEL;
                float ToFNHitsBarrel    = tofF.NHitsBarrel;
                float ToFNHitsEndcap    = tofF.NHitsEndcap;
                float ToFNHitsTotal     = tofF.NHitsTotal;
                float ToFMinDistBarrel  = (tofF.HasToF > 0.5f) ? tofF.MinDistBarrel : MISSING_SENTINEL;
                float ToFMinDistEndcap  = (tofF.HasToF > 0.5f) ? tofF.MinDistEndcap : MISSING_SENTINEL;
                float ToFAvgLenBarrel   = (tofF.HasToF > 0.5f) ? tofF.AvgLenBarrel : MISSING_SENTINEL;
                float ToFAvgLenEndcap   = (tofF.HasToF > 0.5f) ? tofF.AvgLenEndcap : MISSING_SENTINEL;
                float ToFHasToF         = tofF.HasToF;

                float TrackMomentum = static_cast<float>(trackP);
                float TrackEta_f    = static_cast<float>(trackEta);
                float TrackPhi_f    = static_cast<float>(trackPhi);

                // Build raw vector in the exact RAW_COLS order (41 features)
                std::vector<float> raw = build_raw_features_from_components(
                    // ECal (14)
                    ECalEnergy, ECalNumber, ECalEoverP, ECalAvgHitEnergy,
                    ECalSpreadPhi, ECalSpreadEta, ECalSpreadR, ECalMaxHitFrac,
                    ECalEnergyStdDev, ECalEnergyConcentration,
                    ECalR_Disp, ECalR_DispWeighted, ECalEta_DispWeighted, ECalPhi_DispWeighted,
                    // HCal (14)
                    HCalEnergy, HCalNumber, HCalEoverP, HCalAvgHitEnergy,
                    HCalSpreadPhi, HCalSpreadEta, HCalSpreadR, HCalMaxHitFrac,
                    HCalEnergyStdDev, HCalEnergyConcentration,
                    HCalR_Disp, HCalR_DispWeighted, HCalEta_DispWeighted, HCalPhi_DispWeighted,
                    // ToF (10)
                    ToFBeta, ToFMassSq, ToFNHitsBarrel, ToFNHitsEndcap, ToFNHitsTotal,
                    ToFMinDistBarrel, ToFMinDistEndcap, ToFAvgLenBarrel, ToFAvgLenEndcap, ToFHasToF,
                    // Track (3)
                    TrackMomentum, TrackEta_f, TrackPhi_f
                );

                // Run ONNX model (preprocessor + XGBoost) and get P(muon)
                float Pmu = run_muon_id(session, memory_info, raw);

                // Fill histograms and counters
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
            } // end loop over particles in event
        } // end event loop

        delete mychain;
    } // end file loop

    // Compute efficiency / rejection histograms (divide with binomial errors)
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

    // Write output
    outfile->Write();
    outfile->Close();

    cout << "Histograms saved to Plots/MuonID_Performance_fromPythonFeatures.root" << endl;
}

// If compiled as a standalone program, provide a main that calls TestingMacro().
#ifndef __CLING__
int main(int argc, char** argv)
{
    TestingMacro();
    return 0;
}
#endif
