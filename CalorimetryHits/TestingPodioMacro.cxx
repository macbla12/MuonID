// TestingMacro_Podio.cxx
//
// Podio / EDM4eic version of TestingMacro.cxx.
//
// The original TestingMacro.cxx reads the input files through TTreeReader on
// the flat ROOT branches produced by the podio ROOT dumper (e.g.
// "_EcalBarrelImagingClusterAssociations_sim.index",
// "EcalBarrelImagingClusters.hits_begin", "EcalBarrelRecHits.energy", ...).
//
// This file reads the *same* input files but through the podio::Frame /
// EDM4eic collection API, exactly the way MuonID.cxx / MuonID.hpp access
// clusters, hits and associations at runtime in EICrecon. The physics logic
// (cuts, feature definitions, histogram binning, ONNX model, output file)
// is left completely unchanged so this produces the same
// Plots/MuonID_Performance.root as TestingMacro.cxx.
//
// NOTES / ASSUMPTIONS (please verify before using for production training):
//
//   1) Association collection naming: for a cluster collection "XClusters"
//      the corresponding truth-matching collection is assumed to be named
//      "XClusterAssociations" (edm4eic::MCRecoClusterParticleAssociation),
//      i.e. exactly the podio collection that the flat branch
//      "_XClusterAssociations_sim.index" used in TestingMacro.cxx comes from.
//   2) "ReconstructedChargedParticleAssociations" is assumed to be stored in
//      the same order as "ReconstructedChargedParticles" (same assumption
//      the original macro made by indexing the flat "_..._sim.index" array
//      with the track loop index "particle").
//   3) Cluster hits are read directly via cluster.getHits() as
//      edm4eic::CalorimeterHit (getEnergy(), getTime(), getPosition()),
//      instead of manually walking hits_begin/hits_end into a separate
//      RecHits array - podio already gives us that relation directly.
//   4) Same timing cut (20 ns) as the original macro is applied per hit.
//
// If any of the above does not match your EICrecon output, the feature
// vector fed into the ONNX model will be built from the wrong hits (or
// Frame::get<T> will throw if a collection name is wrong).

#include <TH1.h>
#include <TFile.h>
#include <TROOT.h>
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

using namespace std;

// =====================================================================
// Small helper: expand a shell-style wildcard (e.g. "reco_*.root") into a
// list of real file paths. This replaces TChain::Add(pattern), which did
// the globbing internally.
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
// Generic per-detector association bundle.
//
// Replaces the old DetectorHits struct (which held TTreeReaderArrays for
// the flat sim-association / hits_begin / hits_end / RecHits branches).
// In the podio world all we need to remember per detector is which
// association collection to pull from the Frame - the cluster -> hits
// relation and the hit energy/time/position are read directly off the
// EDM4eic objects.
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

    // Same naming convention as the flat branch used in TestingMacro.cxx:
    // "_XClusterAssociations_sim.index" comes from the podio collection
    // "XClusterAssociations" (edm4eic::MCRecoClusterParticleAssociation).
    d.assocCollName = clusterName.substr(0, clusterName.size() - 1) + "Associations";

    cout << "  Detector '" << clusterName << "' -> Association collection '"
         << d.assocCollName << "'" << endl;

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

// Same 29-feature layout as build_raw_features() in TestingMacro.cxx -
// left completely unchanged so the ONNX model sees the exact same input.
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

// =====================================================================
// CollectHits - podio version.
//
// Instead of walking the flat "_XClusterAssociations_sim.index" array plus
// "hits_begin"/"hits_end" plus a separate RecHits array by integer index
// (as TestingMacro.cxx does), we:
//   1) fetch the MCRecoClusterParticleAssociation collection for each
//      detector directly from the Frame,
//   2) keep only the associations whose truth particle matches the track's
//      associated sim particle (same truth-matching logic as
//      "if (simuID != (*det.simuAssocCluster)[iCluster]) continue;"),
//   3) read the matched cluster's hits directly via cluster.getHits().
// =====================================================================
TrackCalHits CollectHits(const edm4hep::MCParticle &simPart,
                          const vector<DetectorAssoc> &detectors,
                          const podio::Frame &frame,
                          const TLorentzVector &Partic,
                          double timingCutNs)
{
    TrackCalHits result;

    for (const auto &det : detectors)
    {
        const auto &assocColl =
            frame.get<edm4eic::MCRecoClusterParticleAssociationCollection>(det.assocCollName);

        for (const auto &assoc : assocColl)
        {
            // Truth-based matching: only keep clusters associated to the
            // same sim particle as the track we are currently processing.
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

// Unchanged with respect to TestingMacro.cxx - pure math on TrackCalHits,
// independent of how the hits were collected.
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
    Ort::Session &session,
    Ort::MemoryInfo &mem,
    const std::vector<float> &raw)
{
    int64_t shape[2] = {1, (int64_t)raw.size()};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float *>(raw.data()), raw.size(), shape, 2);

    const char *input_names[] = {"raw_features"};
    const char *output_names[] = {"probabilities"};

    auto output = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    float *probs = output[0].GetTensorMutableData<float>();
    return probs[1]; // P(muon)
}

void TestingPodioMacro()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    const double TIMING_CUT_NS = 20.0;
    const float MUON_ID_CUT = 0.511f;

    // ONNX Runtime initialization
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "ONNX/xgb_muonID.onnx", session_options);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    static constexpr int NumOfFiles = 2;
    vector<TString> filePatterns(NumOfFiles);

    //filePatterns.at(0) = "/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
    filePatterns.at(0) = "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_1*.root";
    filePatterns.at(1) = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
    //filePatterns.at(1) = "/run/media/epic/Data/Tau/reco/Energy_10x275/old/double_pi/recoDoublePi.root";

    vector<string> ecalClusterNames = {"EcalBarrelImagingClusters", "EcalBarrelScFiClusters", "EcalEndcapPClusters", "EcalEndcapNClusters"};
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    // Output file
    TFile *outfile = new TFile("Plots/MuonID_Performance.root", "RECREATE");

    // -----------------------------------------------------------------
    // Response and efficiency histogram declarations (unchanged)
    // -----------------------------------------------------------------
    TH1F *h_Response_Muon = new TH1F("h_Response_Muon", "P(muon) Response for True Muons;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Pion = new TH1F("h_Response_Pion", "P(muon) Response for True Pions;P(muon);Counts", 100, 0, 1);

    TH1F *h_Muon_Total_vs_Pt = new TH1F("h_Muon_Total_vs_Pt", "Muon Total vs pT;p_{T} [GeV/c];Counts", 30, 0, 20);
    TH1F *h_Muon_Passed_vs_Pt = new TH1F("h_Muon_Passed_vs_Pt", "Muon Passed vs pT;p_{T} [GeV/c];Counts", 30, 0, 20);

    TH1F *h_Muon_Total_vs_P  = new TH1F("h_Muon_Total_vs_P", "Muon Total vs p;p [GeV/c];Counts", 40, 0, 20);
    TH1F *h_Muon_Passed_vs_P = new TH1F("h_Muon_Passed_vs_P", "Muon Passed vs p;p [GeV/c];Counts", 40, 0, 20);

    TH1F *h_Muon_Total_vs_Eta  = new TH1F("h_Muon_Total_vs_Eta", "Muon Total vs #eta;#eta;Counts", 30, -1.2, 3.4);
    TH1F *h_Muon_Passed_vs_Eta = new TH1F("h_Muon_Passed_vs_Eta", "Muon Passed vs #eta;#eta;Counts", 30, -1.2, 3.4);

    TH1F *h_Pion_Total_vs_Pt   = new TH1F("h_Pion_Total_vs_Pt", "Pion Total vs pT;p_{T} [GeV/c];Counts", 30, 0, 20);
    TH1F *h_Pion_Rejected_vs_Pt = new TH1F("h_Pion_Rejected_vs_Pt", "Pion Rejected vs pT;p_{T} [GeV/c];Counts", 30, 0, 20);

    TH1F *h_Pion_Total_vs_P    = new TH1F("h_Pion_Total_vs_P", "Pion Total vs p;p [GeV/c];Counts", 40, 0, 20);
    TH1F *h_Pion_Rejected_vs_P = new TH1F("h_Pion_Rejected_vs_P", "Pion Rejected vs p;p [GeV/c];Counts", 40, 0, 20);

    TH1F *h_Pion_Total_vs_Eta  = new TH1F("h_Pion_Total_vs_Eta", "Pion Total vs #eta;#eta;Counts", 30, -1.2, 3.4);
    TH1F *h_Pion_Rejected_vs_Eta = new TH1F("h_Pion_Rejected_vs_Eta", "Pion Rejected vs #eta;#eta;Counts", 30, -1.2, 3.4);

    for (int File = 0; File < NumOfFiles; File++)
    {
        bool isMuonFile = (File == 0);
        string name = isMuonFile ? "Muons" : "Pions";

        // Expand the wildcard pattern into a concrete list of files
        // (replaces TChain::Add(pattern)).
        vector<string> fileList = ExpandGlob(string(filePatterns.at(File)));

        podio::ROOTReader reader;
        reader.openFiles(fileList);

        unsigned nEvents = reader.getEntries("events");
        cout << "Total events in chain (" << name << "): " << nEvents << endl;

        // Build detector -> association-collection lookup tables once per file.
        vector<DetectorAssoc> ecalDetectors;
        for (auto &n : ecalClusterNames) ecalDetectors.push_back(MakeDetectorAssoc(n));

        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

        int eventID = 0;

        for (unsigned entry = 0; entry < 10000; ++entry)
        //for (unsigned entry = 0; entry < nEvents; ++entry)
        {
            eventID++;
            if (eventID == 200000) break;
            if (eventID % 50000 == 0) cout << "File " << name << " event: " << eventID << endl;

            podio::Frame frame(reader.readEntry("events", entry));

            const auto &tracks =
                frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedChargedParticles");
            const auto &trackAssocs =
                frame.get<edm4eic::MCRecoParticleAssociationCollection>("ReconstructedChargedParticleAssociations");

            for (size_t particle = 0; particle < tracks.size(); ++particle)
            {
                const auto rcp = tracks[particle];

                auto mom = rcp.getMomentum();
                TLorentzVector Partic;
                Partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
                double trackP  = Partic.P();
                double trackPt = Partic.Pt();
                double trackEta = Partic.Eta();

                if (trackP <= 1) continue;
                if (trackEta >= 1 && trackEta <= 1.3) continue;
                if (trackEta <= -1.25) continue;

                // Same assumption as the original macro's "int simuID =
                // simuAssoc[particle];": the association collection is
                // parallel-ordered to the track collection.
                if (particle >= trackAssocs.size()) continue;
                const auto simPart = trackAssocs[particle].getSim();

                TrackCalHits ecalHits = CollectHits(simPart, ecalDetectors, frame, Partic, TIMING_CUT_NS);
                TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP);

                TrackCalHits hcalHits = CollectHits(simPart, hcalDetectors, frame, Partic, TIMING_CUT_NS);
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

                // Filling response and efficiency/rejection counters
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

                    if (Pmu <= MUON_ID_CUT) // Pion correctly rejected
                    {
                        h_Pion_Rejected_vs_Pt->Fill(trackPt);
                        h_Pion_Rejected_vs_P->Fill(trackP);
                        h_Pion_Rejected_vs_Eta->Fill(trackEta);
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

    TH1F *h_Pion_Rejection_vs_Eta  = (TH1F*)h_Pion_Rejected_vs_Eta->Clone("h_Pion_Rejected_vs_Eta");
    h_Pion_Rejection_vs_Eta->SetTitle("Pion Rejection vs #eta;#eta;Rejection Fraction");
    h_Pion_Rejection_vs_Eta->Divide(h_Pion_Rejected_vs_Eta, h_Pion_Total_vs_Eta, 1.0, 1.0, "B");

    // Save to file
    outfile->Write();
    outfile->Close();

    cout << "Efficiency and rejection histograms have been saved to Plots/MuonID_Performance.root" << endl;
}