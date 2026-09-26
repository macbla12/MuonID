// TestingMacro_Fixed_Podio.cxx
//
// Podio / EDM4eic version of TestingMacro_Fixed.cxx.
//
// The original TestingMacro_Fixed.cxx reads the input files through
// TTreeReader on the flat ROOT branches produced by the podio ROOT dumper
// (e.g. "_EcalBarrelImagingClusterAssociations_sim.index",
// "EcalBarrelImagingClusters.hits_begin", "EcalBarrelRecHits.energy",
// "TOFBarrelRecHits.position.x", ...).
//
// This file reads the *same* input files but through the podio::Frame /
// EDM4eic collection API, exactly the way MuonID.cxx / MuonID.hpp access
// clusters, hits, associations and ToF hits at runtime in EICrecon. The
// physics logic (cuts, feature definitions, RAW_COLS ordering, histogram
// binning/titles, ONNX model, output file) is left completely unchanged,
// so this produces the same Plots/MuonID_Performance_fromPythonFeatures.root
// as TestingMacro_Fixed.cxx.
//
// ToFSim.cxx is included as-is (RK4 propagation / field map) - no changes
// needed there, it only takes a TLorentzVector, a charge and a TVector3 and
// has no dependency on how the ToF hit was read.
//
// NOTES / ASSUMPTIONS (please verify before using for production training):
//
//   1) Association collection naming: for a cluster collection "XClusters"
//      the corresponding truth-matching collection is assumed to be named
//      "XClusterAssociations" (edm4eic::MCRecoClusterParticleAssociation),
//      i.e. exactly the podio collection that the flat branch
//      "_XClusterAssociations_sim.index" used in TestingMacro_Fixed.cxx
//      comes from.
//   2) "ReconstructedChargedParticleAssociations" is assumed to be stored in
//      the same order as "ReconstructedChargedParticles" (same assumption
//      the original macro made by indexing the flat "_..._sim.index" array
//      with the track loop index "particle").
//   3) Cluster hits are read directly via cluster.getHits() as
//      edm4eic::CalorimeterHit (getEnergy(), getTime(), getPosition()),
//      instead of manually walking hits_begin/hits_end into a separate
//      RecHits array.
//   4) ToF hit collection names/types: "TOFBarrelRecHits" / "TOFEndcapRecHits"
//      of type edm4eic::TrackerHitCollection (getPosition(), getTime()) -
//      same assumption as MuonID.hpp. Adjust if different in your EICrecon.
//   5) Same timing cut (20 ns) for calo hits, same dR_cut/dist_cut (0.8 / 6.0
//      mm) for ToF hits as the original macro.

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

using namespace std;

// -----------------------------------------------------------------------------
// Include ToFSim implementation (must be present in the same directory).
// This brings in ToFSim(), ToFResults, c_light, etc. UNCHANGED.
// -----------------------------------------------------------------------------
#include "ToFSim.cxx"

// -----------------------------------------------------------------------------
// Small helper: expand a shell-style wildcard (e.g. "reco_*.root") into a
// list of real file paths. This replaces TChain::Add(pattern), which did
// the globbing internally.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Utility: combine multiple ToF hits into a single beta estimate. Unchanged.
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
// Calorimeter helper types and functions - podio version.
// -----------------------------------------------------------------------------

// Replaces the old DetectorHits struct: in podio we only need to remember
// which association collection to pull from the Frame per detector.
struct DetectorAssoc
{
    string clusterName;
    string assocCollName;
};

struct HistSet
{
    TH1F *HitR, *HitPhi, *HitZ, *HitEnergy, *HitTime;
    TH1F *DeltaEta, *DeltaPhi;
    TH1F *h_EoverP, *h_NHits;
};

DetectorAssoc MakeDetectorAssoc(const string &clusterName)
{
    DetectorAssoc d;
    d.clusterName = clusterName;

    // Same naming convention as the flat branch used in
    // TestingMacro_Fixed.cxx: "_XClusterAssociations_sim.index" comes from
    // the podio collection "XClusterAssociations"
    // (edm4eic::MCRecoClusterParticleAssociation).
    d.assocCollName = clusterName.substr(0, clusterName.size() - 1) + "Associations";

    cout << "  Detector '" << clusterName << "' -> Association collection '"
         << d.assocCollName << "'" << endl;
    return d;
}

struct TrackCalHits
{
    double sumEnergy = 0.0;
    double maxHitE = -1.0;
    vector<double> E;
};

// CollectHits - podio version.
//
// Instead of walking the flat "_XClusterAssociations_sim.index" array plus
// "hits_begin"/"hits_end" plus a separate RecHits array by integer index
// (as TestingMacro_Fixed.cxx does), we:
//   1) fetch the MCRecoClusterParticleAssociation collection for each
//      detector directly from the Frame,
//   2) keep only the associations whose truth particle matches the track's
//      associated sim particle (same truth-matching logic as
//      "if (simuID != (*det.simuAssocCluster)[iCluster]) continue;"),
//   3) read the matched cluster's hits directly via cluster.getHits().
// Optional diagnostic histogram filling is unchanged from the original.
TrackCalHits CollectHits(const edm4hep::MCParticle &simPart,
                          const vector<DetectorAssoc> &detectors,
                          const podio::Frame &frame,
                          const TLorentzVector &Partic,
                          double timingCutNs,
                          TH1F *hR=nullptr, TH1F *hPhi=nullptr, TH1F *hZ=nullptr, TH1F *hE=nullptr, TH1F *hT=nullptr,
                          TH1F *hDEta=nullptr, TH1F *hDPhi=nullptr)
{
    TrackCalHits result;

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

                if (hT) hT->Fill(HitT);
                if (HitT > timingCutNs) continue;

                TVector3 hitVec(pos.x, pos.y, pos.z);
                double R = sqrt(pos.x * pos.x + pos.y * pos.y);

                if (hR)   hR->Fill(R);
                if (hPhi) hPhi->Fill(hitVec.Phi());
                if (hZ)   hZ->Fill(pos.z);
                if (hE)   hE->Fill(HitE);

                double dEta = hitVec.Eta() - Partic.Eta();
                double dPhi = TVector2::Phi_mpi_pi(hitVec.Phi() - Partic.Phi());
                if (hDEta) hDEta->Fill(dEta);
                if (hDPhi) hDPhi->Fill(dPhi);

                result.sumEnergy += HitE;
                result.E.push_back(HitE);
                if (HitE > result.maxHitE) result.maxHitE = HitE;
            }
        }
    }
    return result;
}

struct TrackCaloFeatures
{
    float Energy = 0.f;
    float Number = 0.f;
    float EoverP = 0.f;
    float MaxHitFrac = 0.f;
};

// Unchanged - pure math on TrackCalHits.
TrackCaloFeatures ComputeCaloFeatures(const TrackCalHits &hits, double trackP, HistSet &h)
{
    TrackCaloFeatures f;
    if (hits.sumEnergy <= 0 || hits.E.empty()) return f;

    f.Energy     = static_cast<float>(hits.sumEnergy);
    f.Number     = static_cast<float>(hits.E.size());
    f.EoverP     = static_cast<float>(hits.sumEnergy / trackP);
    f.MaxHitFrac = static_cast<float>(hits.maxHitE / hits.sumEnergy);

    if (h.h_EoverP) h.h_EoverP->Fill(f.EoverP);
    if (h.h_NHits)  h.h_NHits->Fill(f.Number);
    return f;
}

// Unchanged.
HistSet MakeHistSet(const string &prefix)
{
    HistSet h;
    h.HitR      = new TH1F((prefix + "_HitR").c_str(), (prefix + " Hit R;R [mm];Counts").c_str(), 200, 0, 3200);
    h.HitPhi    = new TH1F((prefix + "_HitPhi").c_str(), (prefix + " Hit Phi;#phi;Counts").c_str(), 200, -3.15, 3.15);
    h.HitZ      = new TH1F((prefix + "_HitZ").c_str(), (prefix + " Hit Z;Z [mm];Counts").c_str(), 200, -3000, 3000);
    h.HitEnergy = new TH1F((prefix + "_HitEnergy").c_str(), (prefix + " Hit Energy;E [GeV];Counts").c_str(), 200, 0, 1);
    h.HitTime   = new TH1F((prefix + "_HitTime").c_str(), (prefix + " Hit Time;t [ns];Counts").c_str(), 200, 0, 200);
    h.DeltaEta  = new TH1F((prefix + "_DeltaEta").c_str(), (prefix + " #Delta#eta;#Delta#eta;Counts").c_str(), 200, -1, 1);
    h.DeltaPhi  = new TH1F((prefix + "_DeltaPhi").c_str(), (prefix + " #Delta#phi;#Delta#phi;Counts").c_str(), 200, -1, 1);
    h.h_EoverP  = new TH1F((prefix + "_h_EoverP").c_str(), (prefix + " E/p;E/p;Counts").c_str(), 150, 0, 3);
    h.h_NHits   = new TH1F((prefix + "_h_NHits").c_str(), (prefix + " N hits;N;Counts").c_str(), 60, 0, 60);
    return h;
}

// -----------------------------------------------------------------------------
// ToF helper types and functions - podio version.
//
// Same struct / feature definitions as TestingMacro_Fixed.cxx. Only the hit
// source changes: instead of separate TTreeReaderArrays for position.x/y/z
// and time, we take an edm4eic::TrackerHitCollection (same type MuonID.hpp
// assumes for "TOFBarrelRecHits"/"TOFEndcapRecHits") and use
// hit.getPosition() / hit.getTime() directly.
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

template <typename HitCollection>
TrackToFFeatures ComputeToFFeatures(
    const TLorentzVector &Partic, int charge,
    const HitCollection &barrelHits, const HitCollection &endcapHits,
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
        auto pos = hit.getPosition();
        TVector3 ToFPos(pos.x, pos.y, pos.z);
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
        auto pos = hit.getPosition();
        TVector3 ToFPos(pos.x, pos.y, pos.z);
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

// -----------------------------------------------------------------------------
// ONNX runtime helper: run the merged ONNX model (preprocessor + XGBoost).
// The ONNX graph expects a single input named "raw_features" with shape (N,21)
// and returns "probabilities" (softmax) or similar; we read P(muon) = probs[1].
// Unchanged.
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
// Unchanged.
// -----------------------------------------------------------------------------
static const int N_RAW = 21;
static const float MISSING_SENTINEL = -999.f;

std::vector<float> build_raw_features_from_components(
    float ECalEnergy, float ECalNumber, float ECalEoverP, float ECalMaxHitFrac,
    float HCalEnergy, float HCalNumber, float HCalEoverP, float HCalMaxHitFrac,
    float ToFBeta, float ToFMassSq, float ToFNHitsBarrel, float ToFNHitsEndcap,
    float ToFNHitsTotal, float ToFMinDistBarrel, float ToFMinDistEndcap,
    float ToFAvgLenBarrel, float ToFAvgLenEndcap, float ToFHasToF,
    float TrackMomentum, float TrackEta, float TrackPhi)
{
    std::vector<float> raw;
    raw.reserve(N_RAW);

    raw.push_back(ECalEnergy);
    raw.push_back(ECalNumber);
    raw.push_back(ECalEoverP);
    raw.push_back(ECalMaxHitFrac);

    raw.push_back(HCalEnergy);
    raw.push_back(HCalNumber);
    raw.push_back(HCalEoverP);
    raw.push_back(HCalMaxHitFrac);

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

    raw.push_back(TrackMomentum);
    raw.push_back(TrackEta);
    raw.push_back(TrackPhi);

    return raw;
}

// -----------------------------------------------------------------------------
// Main testing macro: iterate over muon/pion files, compute features, run ONNX,
// fill histograms for response, efficiency and rejection vs p, pT, eta.
// -----------------------------------------------------------------------------
void TestingPodioMacro()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    const double TIMING_CUT_NS = 20.0;
    const float MUON_ID_CUT = 0.2f;

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load merged ONNX model (preprocessor + XGBoost)
    Ort::Session session(env, "ONNX/xgb_muonID.onnx", session_options);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    static constexpr int NumOfFiles = 2;
    vector<TString> filePatterns(NumOfFiles);
    //filePatterns.at(0) = "/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
    filePatterns.at(0)="/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_1*.root";
    filePatterns.at(1) = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
    //filePatterns.at(1)="/run/media/epic/Data/Tau/reco/Energy_10x275/old/double_pi/recoDoublePi.root";

    vector<string> ecalClusterNames = {"EcalBarrelImagingClusters","EcalBarrelScFiClusters", "EcalEndcapPClusters", "EcalEndcapNClusters"};
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    // Output file for histograms
    TFile *outfile = new TFile("Plots/MuonID_Performance_fromPythonFeatures.root", "RECREATE");

    // Response histograms
    TH1F *h_Response_Muon = new TH1F("h_Response_Muon", "P(muon) Response for True Muons;P(muon);Counts", 100, 0, 1);
    TH1F *h_Response_Pion = new TH1F("h_Response_Pion", "P(muon) Response for True Pions;P(muon);Counts", 100, 0, 1);

    // Efficiency / rejection counters vs pT, p, eta
    TH1F *h_Muon_Total_vs_Pt = new TH1F("h_Muon_Total_vs_Pt", "Muon Total vs pT;p_{T} [GeV/c];Counts",30,0,2);
    TH1F *h_Muon_Passed_vs_Pt = new TH1F("h_Muon_Passed_vs_Pt", "Muon Passed vs pT;p_{T} [GeV/c];Counts",30,0,2);

    TH1F *h_Muon_Total_vs_P  = new TH1F("h_Muon_Total_vs_P", "Muon Total vs p;p [GeV/c];Counts",40,0,2);
    TH1F *h_Muon_Passed_vs_P = new TH1F("h_Muon_Passed_vs_P", "Muon Passed vs p;p [GeV/c];Counts",40,0,2);

    TH1F *h_Muon_Total_vs_Eta  = new TH1F("h_Muon_Total_vs_Eta", "Muon Total vs #eta;#eta;Counts",30,-1.2,3.4);
    TH1F *h_Muon_Passed_vs_Eta = new TH1F("h_Muon_Passed_vs_Eta", "Muon Passed vs #eta;#eta;Counts",30,-1.2,3.4);

    TH1F *h_Pion_Total_vs_Pt   = new TH1F("h_Pion_Total_vs_Pt", "Pion Total vs pT;p_{T} [GeV/c];Counts",30,0,2);
    TH1F *h_Pion_Rejected_vs_Pt = new TH1F("h_Pion_Rejected_vs_Pt", "Pion Rejected vs pT;p_{T} [GeV/c];Counts",30,0,2);

    TH1F *h_Pion_Total_vs_P    = new TH1F("h_Pion_Total_vs_P", "Pion Total vs p;p [GeV/c];Counts",40,0,2);
    TH1F *h_Pion_Rejected_vs_P = new TH1F("h_Pion_Rejected_vs_P", "Pion Rejected vs p;p [GeV/c];Counts",40,0,2);

    TH1F *h_Pion_Total_vs_Eta  = new TH1F("h_Pion_Total_vs_Eta", "Pion Total vs #eta;#eta;Counts",30,-1.2,3.4);
    TH1F *h_Pion_Rejected_vs_Eta = new TH1F("h_Pion_Rejected_vs_Eta", "Pion Rejected vs #eta;#eta;Counts",30,-1.2,3.4);

    // Loop over files (muons then pions)
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

        // Build detector groups for calo (association-collection names derived automatically).
        vector<DetectorAssoc> ecalDetectors;
        for (auto &n : ecalClusterNames) ecalDetectors.push_back(MakeDetectorAssoc(n));

        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

        int eventID = 0;

        //for (unsigned entry = 0; entry < nEvents; ++entry)
        for (unsigned entry = 0; entry < 10000; ++entry)
        {
            eventID++;
            if (eventID > 200000) break;
            if (eventID % 50000 == 0) cout << "File " << name << " event: " << eventID << endl;

            podio::Frame frame(reader.readEntry("events", entry));

            const auto &tracks =
                frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedChargedParticles");
            const auto &trackAssocs =
                frame.get<edm4eic::MCRecoParticleAssociationCollection>("ReconstructedChargedParticleAssociations");

            // ToF hit collections (fetched once per event, same role as the
            // per-event TTreeReaderArrays in the original macro).
            const auto &tofBarrelHits =
                frame.get<edm4eic::TrackerHitCollection>("TOFBarrelRecHits");
            const auto &tofEndcapHits =
                frame.get<edm4eic::TrackerHitCollection>("TOFEndcapRecHits");

            for (size_t particle = 0; particle < tracks.size(); particle++)
            {
                const auto rcp = tracks[particle];

                auto mom = rcp.getMomentum();
                TLorentzVector Partic;
                Partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
                double trackP  = Partic.P();
                double trackPt = Partic.Pt();
                double trackEta = Partic.Eta();
                double trackPhi = Partic.Phi();

                // Keep selection consistent with training macro
                if (trackP > 2.0) continue;
                if (trackEta >= 1.0 && trackEta <= 1.3) continue;
                if (trackEta <= -1.25) continue;

                int charge = static_cast<int>(rcp.getCharge());

                // Same assumption as the original macro's "int simuID =
                // simuAssoc[particle];": the association collection is
                // parallel-ordered to the track collection.
                if (particle >= trackAssocs.size()) continue;
                const auto simPart = trackAssocs[particle].getSim();

                // Collect calorimeter hits and compute features
                HistSet ecalHists = MakeHistSet("ECal_tmp");
                HistSet hcalHists = MakeHistSet("HCal_tmp");
                TrackCalHits ecalHits = CollectHits(simPart, ecalDetectors, frame, Partic, TIMING_CUT_NS,
                    ecalHists.HitR, ecalHists.HitPhi, ecalHists.HitZ, ecalHists.HitEnergy, ecalHists.HitTime,
                    ecalHists.DeltaEta, ecalHists.DeltaPhi);
                TrackCaloFeatures ecalF = ComputeCaloFeatures(ecalHits, trackP, ecalHists);

                TrackCalHits hcalHits = CollectHits(simPart, hcalDetectors, frame, Partic, TIMING_CUT_NS,
                    hcalHists.HitR, hcalHists.HitPhi, hcalHists.HitZ, hcalHists.HitEnergy, hcalHists.HitTime,
                    hcalHists.DeltaEta, hcalHists.DeltaPhi);
                TrackCaloFeatures hcalF = ComputeCaloFeatures(hcalHits, trackP, hcalHists);

                // Compute ToF features
                TrackToFFeatures tofF = ComputeToFFeatures(
                    Partic, charge,
                    tofBarrelHits, tofEndcapHits,
                    /*dR_cut_barrel*/0.8, /*dR_cut_endcap*/0.8,
                    /*dist_cut_barrel*/6.0, /*dist_cut_endcap*/6.0
                );

                // Drop tracks invisible everywhere (same logic as training)
                if (ecalF.Number <= 0 && hcalF.Number <= 0 && tofF.HasToF < 0.5) continue;

                // Map computed features -> raw fields expected by ONNX preprocessor
                float ECalEnergy     = ecalF.Energy;
                float ECalNumber     = ecalF.Number;
                float ECalEoverP     = (ecalF.Number > 0) ? ecalF.EoverP : MISSING_SENTINEL;
                float ECalMaxHitFrac = (ecalF.Number > 0) ? ecalF.MaxHitFrac : MISSING_SENTINEL;

                float HCalEnergy     = hcalF.Energy;
                float HCalNumber     = hcalF.Number;
                float HCalEoverP     = (hcalF.Number > 0) ? hcalF.EoverP : MISSING_SENTINEL;
                float HCalMaxHitFrac = (hcalF.Number > 0) ? hcalF.MaxHitFrac : MISSING_SENTINEL;

                float ToFBeta           = tofF.Beta;
                float ToFMassSq         = tofF.MassSq;
                float ToFNHitsBarrel    = tofF.NHitsBarrel;
                float ToFNHitsEndcap    = tofF.NHitsEndcap;
                float ToFNHitsTotal     = tofF.NHitsTotal;
                float ToFMinDistBarrel  = tofF.MinDistBarrel;
                float ToFMinDistEndcap  = tofF.MinDistEndcap;
                float ToFAvgLenBarrel   = tofF.AvgLenBarrel;
                float ToFAvgLenEndcap   = tofF.AvgLenEndcap;
                float ToFHasToF         = tofF.HasToF;

                float TrackMomentum = static_cast<float>(trackP);
                float TrackEta_f    = static_cast<float>(trackEta);
                float TrackPhi_f    = static_cast<float>(trackPhi);

                // Build raw vector in the exact RAW_COLS order
                std::vector<float> raw = build_raw_features_from_components(
                    ECalEnergy, ECalNumber, ECalEoverP, ECalMaxHitFrac,
                    HCalEnergy, HCalNumber, HCalEoverP, HCalMaxHitFrac,
                    ToFBeta, ToFMassSq, ToFNHitsBarrel, ToFNHitsEndcap, ToFNHitsTotal,
                    ToFMinDistBarrel, ToFMinDistEndcap, ToFAvgLenBarrel, ToFAvgLenEndcap, ToFHasToF,
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

// If compiled as a standalone program, provide a main that calls
// TestingMacro_Fixed_Podio(). Unchanged pattern from the original file.
#ifndef __CLING__
int main(int argc, char** argv)
{
    TestingPodioMacro();
    return 0;
}
#endif