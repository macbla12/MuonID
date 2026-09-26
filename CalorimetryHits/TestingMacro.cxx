#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TROOT.h>
#include <TChain.h>
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

// =====================================================================
// Generic per-detector reader bundle
// =====================================================================
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

DetectorHits MakeDetectorHits(TTreeReader &reader, const string &clusterName)
{
    DetectorHits d;
    d.clusterName = clusterName;

    string recHitsName = clusterName;
    size_t pos = recHitsName.rfind("Clusters");
    if (pos != string::npos) recHitsName.replace(pos, string("Clusters").size(), "RecHits");

    string assocName = "_" + clusterName.substr(0, clusterName.size() - 1) + "Associations_sim.index";
    string hitsIndexName = "_" + clusterName + "_hits.index";
    string hitsBeginName  = clusterName + ".hits_begin";
    string hitsEndName    = clusterName + ".hits_end";

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

std::vector<float> build_raw_features(
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

TrackCalHits CollectHits(int simuID, vector<DetectorHits> &detectors, const TLorentzVector &Partic, double timingCutNs)
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

                if (HitT > timingCutNs) continue;

                TVector3 hitVec(Hitx, Hity, Hitz);
                double hitEta = hitVec.Eta();
                double hitPhi = hitVec.Phi();
                double R = sqrt(Hitx * Hitx + Hity * Hity);

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

float run_muon_id(
    Ort::Session& session,
    Ort::MemoryInfo& mem,
    const std::vector<float>& raw)
{
    int64_t shape[2] = {1, (int64_t)raw.size()};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(raw.data()), raw.size(), shape, 2);

    const char* input_names[] = {"raw_features"};
    const char* output_names[] = {"probabilities"};

    auto output = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    float* probs = output[0].GetTensorMutableData<float>();
    return probs[1]; // P(muon)
}

void TestingMacro()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    const double TIMING_CUT_NS = 20.0;
    const float MUON_ID_CUT = 0.511f; 

    // Inicjalizacja ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "ONNX/xgb_muonID.onnx", session_options);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    static constexpr int NumOfFiles = 2;
    vector<TString> files(NumOfFiles);

    files.at(0) = "/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
    //files.at(0)="/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275*";
    files.at(1) = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
    //files.at(1)="/run/media/epic/Data/Tau/reco/Energy_10x275/old/double_pi/recoDoublePi.root";

    vector<string> ecalClusterNames = {"EcalBarrelImagingClusters","EcalBarrelScFiClusters", "EcalEndcapPClusters", "EcalEndcapNClusters"};
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    // Output file
    TFile *outfile = new TFile("Plots/MuonID_Performance.root", "RECREATE");

    // -----------------------------------------------------------------
    // Declare response and efficiency histograms
    // -----------------------------------------------------------------
    TH1F *h_Response_Muon = new TH1F("h_Response_Muon", "P(muon) Response for True Muons;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Pion = new TH1F("h_Response_Pion", "P(muon) Response for True Pions;P(muon);Counts", 100, 0, 1);

    // Counters for efficiency and rejection (numerator / denominator)
    TH1F *h_Muon_Total_vs_Pt = new TH1F("h_Muon_Total_vs_Pt", "Muon Total vs pT;p_{T} [GeV/c];Counts",30,0,20);
    TH1F *h_Muon_Passed_vs_Pt = new TH1F("h_Muon_Passed_vs_Pt", "Muon Passed vs pT;p_{T} [GeV/c];Counts",30,0,20);

    TH1F *h_Muon_Total_vs_P  = new TH1F("h_Muon_Total_vs_P", "Muon Total vs p;p [GeV/c];Counts",40,0,20);
    TH1F *h_Muon_Passed_vs_P = new TH1F("h_Muon_Passed_vs_P", "Muon Passed vs p;p [GeV/c];Counts",40,0,20);

    TH1F *h_Muon_Total_vs_Eta  = new TH1F("h_Muon_Total_vs_Eta", "Muon Total vs #eta;#eta;Counts",30,-1.2,3.4);
    TH1F *h_Muon_Passed_vs_Eta = new TH1F("h_Muon_Passed_vs_Eta", "Muon Passed vs #eta;#eta;Counts",30,-1.2,3.4);

    TH1F *h_Pion_Total_vs_Pt   = new TH1F("h_Pion_Total_vs_Pt", "Pion Total vs pT;p_{T} [GeV/c];Counts",30,0,20);
    TH1F *h_Pion_Rejected_vs_Pt = new TH1F("h_Pion_Rejected_vs_Pt", "Pion Rejected vs pT;p_{T} [GeV/c];Counts",30,0,20);

    TH1F *h_Pion_Total_vs_P    = new TH1F("h_Pion_Total_vs_P", "Pion Total vs p;p [GeV/c];Counts",40,0,20);
    TH1F *h_Pion_Rejected_vs_P = new TH1F("h_Pion_Rejected_vs_P", "Pion Rejected vs p;p [GeV/c];Counts",40,0,20);

    TH1F *h_Pion_Total_vs_Eta  = new TH1F("h_Pion_Total_vs_Eta", "Pion Total vs #eta;#eta;Counts",30,-1.2,3.4);
    TH1F *h_Pion_Rejected_vs_Eta = new TH1F("h_Pion_Rejected_vs_Eta", "Pion Rejected vs #eta;#eta;Counts",30,-1.2,3.4);

    for (int File = 0; File < NumOfFiles; File++)
    {
        bool isMuonFile = (File == 0);
        string name = isMuonFile ? "Muons" : "Pions";

        TChain *mychain = new TChain("events");
        mychain->Add(files.at(File));

        TTreeReader tree_reader(mychain);
        Long64_t nEvents = mychain->GetEntries();
        cout << "Total events in chain (" << name << "): " << nEvents << endl;

        TTreeReaderArray<float> trackMomX(tree_reader, "ReconstructedChargedParticles.momentum.x");
        TTreeReaderArray<float> trackMomY(tree_reader, "ReconstructedChargedParticles.momentum.y");
        TTreeReaderArray<float> trackMomZ(tree_reader, "ReconstructedChargedParticles.momentum.z");
        TTreeReaderArray<float> trackEng(tree_reader, "ReconstructedChargedParticles.energy");
        TTreeReaderArray<int> simuAssoc(tree_reader, "_ReconstructedChargedParticleAssociations_sim.index");

        vector<DetectorHits> ecalDetectors;
        for (auto &n : ecalClusterNames) ecalDetectors.push_back(MakeDetectorHits(tree_reader, n));

        vector<DetectorHits> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorHits(tree_reader, n));

        int eventID = 0;

        while (tree_reader.Next())
        {
            eventID++;
            if (eventID == 200000) break;
            if (eventID % 50000 == 0) cout << "File " << name << " event: " << eventID << endl;
            for (size_t particle = 0; particle < trackMomX.GetSize(); particle++)
            {
                TLorentzVector Partic;
                Partic.SetPxPyPzE(trackMomX[particle], trackMomY[particle], trackMomZ[particle], trackEng[particle]);
                double trackP  = Partic.P();
                double trackPt = Partic.Pt();
                double trackEta = Partic.Eta();

                if (trackP <= 1) continue;
                if (trackEta >= 1 && trackEta <= 1.3) continue;
                if (trackEta <= -1.25) continue;
                //cout<<"KUpa"<<endl;


                int simuID = simuAssoc[particle];

                TrackCalHits ecalHits = CollectHits(simuID, ecalDetectors, Partic, TIMING_CUT_NS);
                TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP);

                TrackCalHits hcalHits = CollectHits(simuID, hcalDetectors, Partic, TIMING_CUT_NS);
                TrackFeatures hcalF = ComputeTrackFeatures(hcalHits, trackP);

                if (ecalF.Number <= 0 && hcalF.Number <= 0) continue;

                std::vector<float> raw = build_raw_features(
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

                float Pmu = run_muon_id(session, memory_info, raw);

                // Filling Response and Efficiency/Rejection counters
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
                else // Pion File
                {
                    h_Response_Pion->Fill(Pmu);

                    h_Pion_Total_vs_Pt->Fill(trackPt);
                    h_Pion_Total_vs_P->Fill(trackP);
                    h_Pion_Total_vs_Eta->Fill(trackEta);

                    if (Pmu <= MUON_ID_CUT) // Pion correctly rejected
                    {
                        h_Pion_Rejected_vs_Pt->Fill(trackPt);
                        h_Pion_Rejected_vs_P->Fill(trackP);
                        h_Pion_Rejected_vs_Eta->Fill(trackEta);
                    }
                }
            }
        }

        delete mychain;
    }

    // -----------------------------------------------------------------
    // Compute final efficiency and rejection histograms (ratios)
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

    TH1F *h_Pion_Rejection_vs_Eta  = (TH1F*)h_Pion_Rejected_vs_Eta->Clone("h_Pion_Rejected_vs_Eta");
    h_Pion_Rejection_vs_Eta->SetTitle("Pion Rejection vs #eta;#eta;Rejection Fraction");
    h_Pion_Rejection_vs_Eta->Divide(h_Pion_Rejected_vs_Eta, h_Pion_Total_vs_Eta, 1.0, 1.0, "B");

    // Save to file
    outfile->Write();
    outfile->Close();

    cout << "Efficiency and rejection histograms saved to Plots/MuonID_Performance.root" << endl;
}