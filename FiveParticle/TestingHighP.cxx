// TestingMacro_Podio.cxx
//
// Podio / EDM4eic version of TestingMacro.cxx.
//
// Cluster selection is identical to TrainingMacro_Podio.cxx (and ElectronID):
//  - ECal: clusters attached to the reconstructed particle, rcp.getClusters()
//  - HCal: clusters from TrackClusterMatch (EICrecon) for the track of the particle
// Hits are read from cluster.getHits(). No MC association is used.
//
// Output: muon efficiency + rejection of the whole background (combined)
//         AND rejection separately for Pion / Electron / Kaon / Proton.

#include <TH1.h>
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

using namespace std;

// =====================================================================
// glob: expand "reco_*.root" into a list of files (replaces TChain::Add(pattern)).
// =====================================================================
vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t g;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) result.push_back(g.gl_pathv[i]);
    globfree(&g);
    return result;
}

// =====================================================================
// Per detector: only the name of the TrackClusterMatch collection is needed.
// "XClusters" -> "XTrackClusterMatches"
// =====================================================================
struct DetectorAssoc
{
    string clusterName;
    string matchCollName;
    bool   available = true;   // set after checking the first event
};

DetectorAssoc MakeDetectorAssoc(const string &clusterName)
{
    DetectorAssoc d;
    d.clusterName   = clusterName;
    d.matchCollName = clusterName.substr(0, clusterName.size() - string("Clusters").size()) + "TrackClusterMatches";
    return d;
}

// Checks whether the detector collections exist in the file (Frame::get throws for a missing one).
void CheckAvailability(vector<DetectorAssoc> &dets, const podio::Frame &frame)
{
    auto names = frame.getAvailableCollections();
    for (auto &d : dets) {
        bool hasMatch = find(names.begin(), names.end(), d.matchCollName) != names.end();
        d.available = hasMatch;
        cout << "  " << d.clusterName << " -> '" << d.matchCollName << "' : "
             << (hasMatch ? "OK" : "MISSING (detector skipped)") << endl;
    }
}

// =====================================================================
// Poprawka na vz (per czastka, z rcp.getReferencePoint(), jak w ToFMass):
//  - czas hitu:  t_corr = t - T0_PER_MM*vz = t + vz/c   (t0 = -vz/c), uzywany w cieciu czasowym
//  - kierunek:   eta/phi hitu liczone wzgledem reference point czastki, a nie poczatku ukladu
// UWAGA: cechy musza byc liczone tak samo jak w treningu (TrainingMacro_Podio.cxx)!
// =====================================================================
static constexpr double c_light = 299.792458;   // mm/ns
const bool   USE_VZ_TIME_CORR      = true;
const bool   USE_VZ_DIRECTION_CORR = false;     // zmienia cechy (dEta/dPhi/spread) - wlacz tylko jesli trening tez tak liczy
const double T0_PER_MM = -1.0 / c_light;        // t0 = T0_PER_MM * vz   [ns/mm]

struct TrackCalHits
{
    double sumEnergy = 0.0;
    double phiMin = 1e9, phiMax = -1e9;
    double etaMin = 1e9, etaMax = -1e9;
    double Rmin = 1e9, Rmax = -1e9;
    double maxHitE = -1.0;
    vector<double> R, dEta, dPhi, E;
};

// Same 29-feature layout as build_raw_features() in TestingMacro.cxx.
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
// CollectHits - same selection as in TrainingMacro_Podio.cxx.
//  ECal (fromRcpClusters = true):  rcp.getClusters()
//  HCal (fromRcpClusters = false): TrackClusterMatch for the track of this particle
// Hits are read from cluster.getHits().
// =====================================================================
TrackCalHits CollectHits(const edm4eic::ReconstructedParticle &rcp,
                         const vector<DetectorAssoc> &detectors,
                         const podio::Frame &frame,
                         bool fromRcpClusters,
                         const TLorentzVector &Partic,
                         const TVector3 &refPoint)
{
    TrackCalHits result;
    const double vz = refPoint.Z();   // vz tej czastki

    vector<edm4eic::Cluster> clusters;
    set<podio::ObjectID> imagingHitIDs;


    if (fromRcpClusters)
    {
        for (const auto &c : rcp.getClusters()) clusters.push_back(c);
        auto avail = frame.getAvailableCollections();
        if (find(avail.begin(), avail.end(), "EcalBarrelImagingRecHits") != avail.end())
        {
            for (const auto &ih : frame.get<edm4eic::CalorimeterHitCollection>("EcalBarrelImagingRecHits"))
                imagingHitIDs.insert(ih.getObjectID());
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

// Pure math on TrackCalHits.
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

void TestingHighP()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    // Same as in TrainingMacro_Podio.cxx - features must be built the same way as in training.
    const float MUON_ID_CUT = 0.5f;

    // ONNX Runtime initialization
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "ONNX/xgb_muonID_HighP.onnx", session_options);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // File 0 = signal (muons); Files 1..4 = background: Pion, Electron, Kaon, Proton
    static constexpr int NumOfFiles = 5;
    vector<string> filePatterns(NumOfFiles);
    filePatterns.at(0) = "/run/media/epic/Data/Background/Muon/Podio-1.7.99/reco_*.root";
    filePatterns.at(1) = "/run/media/epic/Data/Background/Pion/Podio-1.7.99/reco_*.root";
    filePatterns.at(2) = "/run/media/epic/Data/Background/Electron/Podio-1.7.99/reco_*.root";
    filePatterns.at(3) = "/run/media/epic/Data/Background/Kaon/Podio-1.7.99/reco_*.root";
    filePatterns.at(4) = "/run/media/epic/Data/Background/Proton/Podio-1.7.99/reco_*.root";

    // ECal is taken from rcp.getClusters(), so only the HCal names are needed.
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    // Output file
    TFile *outfile = new TFile("Plots/Performance_HighP.root", "RECREATE");

    // -----------------------------------------------------------------
    // Muon (signal) histograms
    // -----------------------------------------------------------------
    TH1F *h_Response_Muon = new TH1F("h_Response_Muon", "P(muon) Response for True Muons;P(muon);Counts", 100, 0, 1);

    TH1F *h_Muon_Total_vs_Pt = new TH1F("h_Muon_Total_vs_Pt", "Muon Total vs pT;p_{T} [GeV/c];Counts", 30, 0, 20);
    TH1F *h_Muon_Passed_vs_Pt = new TH1F("h_Muon_Passed_vs_Pt", "Muon Passed vs pT;p_{T} [GeV/c];Counts", 30, 0, 20);

    TH1F *h_Muon_Total_vs_P  = new TH1F("h_Muon_Total_vs_P", "Muon Total vs p;p [GeV/c];Counts", 40, 0, 20);
    TH1F *h_Muon_Passed_vs_P = new TH1F("h_Muon_Passed_vs_P", "Muon Passed vs p;p [GeV/c];Counts", 40, 0, 20);

    TH1F *h_Muon_Total_vs_Eta  = new TH1F("h_Muon_Total_vs_Eta", "Muon Total vs #eta;#eta;Counts", 30, -3.4, 3.4);
    TH1F *h_Muon_Passed_vs_Eta = new TH1F("h_Muon_Passed_vs_Eta", "Muon Passed vs #eta;#eta;Counts", 30, -3.4, 3.4);

    // -----------------------------------------------------------------
    // Combined background (names h_Background_*;
    // existing plotting scripts - these now mean ALL background species)
    // -----------------------------------------------------------------
    TH1F *h_Response_Background = new TH1F("h_Response_Background", "P(muon) Response for All Background;P(muon);Counts", 100, 0, 1);

    TH1F *h_Background_Total_vs_Pt    = new TH1F("h_Background_Total_vs_Pt", "Background Total vs pT;p_{T} [GeV/c];Counts", 30, 0, 20);
    TH1F *h_Background_Rejected_vs_Pt = new TH1F("h_Background_Rejected_vs_Pt", "Background Rejected vs pT;p_{T} [GeV/c];Counts", 30, 0, 20);

    TH1F *h_Background_Total_vs_P    = new TH1F("h_Background_Total_vs_P", "Background Total vs p;p [GeV/c];Counts", 40, 0, 20);
    TH1F *h_Background_Rejected_vs_P = new TH1F("h_Background_Rejected_vs_P", "Background Rejected vs p;p [GeV/c];Counts", 40, 0, 20);

    TH1F *h_Background_Total_vs_Eta    = new TH1F("h_Background_Total_vs_Eta", "Background Total vs #eta;#eta;Counts", 30, -3.4, 3.4);
    TH1F *h_Background_Rejected_vs_Eta = new TH1F("h_Background_Rejected_vs_Eta", "Background Rejected vs #eta;#eta;Counts", 30, -3.4, 3.4);

    // -----------------------------------------------------------------
    // Separately for each background species: 0=Pion, 1=Electron, 2=Kaon, 3=Proton
    // (index = File - 1)
    // -----------------------------------------------------------------
    const int NBkg = 4;
    const char *bkgName[NBkg] = {"Pion", "Electron", "Kaon", "Proton"};

    TH1F *h_Resp_Bkg[NBkg];
    TH1F *h_Bkg_Total_vs_Pt[NBkg],  *h_Bkg_Rej_vs_Pt[NBkg];
    TH1F *h_Bkg_Total_vs_P[NBkg],   *h_Bkg_Rej_vs_P[NBkg];
    TH1F *h_Bkg_Total_vs_Eta[NBkg], *h_Bkg_Rej_vs_Eta[NBkg];

    for (int i = 0; i < NBkg; ++i)
    {
        const char *n = bkgName[i];
        h_Resp_Bkg[i] = new TH1F(Form("h_Response_%s_Sep", n), Form("P(muon) Response for True %ss;P(muon);Counts", n), 100, 0, 1);

        h_Bkg_Total_vs_Pt[i] = new TH1F(Form("h_%s_Total_vs_Pt_Sep", n),    Form("%s Total vs pT;p_{T} [GeV/c];Counts", n), 30, 0, 20);
        h_Bkg_Rej_vs_Pt[i]   = new TH1F(Form("h_%s_Rejected_vs_Pt_Sep", n), Form("%s Rejected vs pT;p_{T} [GeV/c];Counts", n), 30, 0, 20);

        h_Bkg_Total_vs_P[i]  = new TH1F(Form("h_%s_Total_vs_P_Sep", n),     Form("%s Total vs p;p [GeV/c];Counts", n), 40, 0, 20);
        h_Bkg_Rej_vs_P[i]    = new TH1F(Form("h_%s_Rejected_vs_P_Sep", n),  Form("%s Rejected vs p;p [GeV/c];Counts", n), 40, 0, 20);

        h_Bkg_Total_vs_Eta[i] = new TH1F(Form("h_%s_Total_vs_Eta_Sep", n),    Form("%s Total vs #eta;#eta;Counts", n), 30, -3.4, 3.4);
        h_Bkg_Rej_vs_Eta[i]   = new TH1F(Form("h_%s_Rejected_vs_Eta_Sep", n), Form("%s Rejected vs #eta;#eta;Counts", n), 30, -3.4, 3.4);
    }

    for (int File = 0; File < NumOfFiles; File++)
    {
        string name = (File == 0) ? "Muons" : (File == 1) ? "Pions" : (File == 2) ? "Electrons" : (File == 3) ? "Kaons" : (File == 4) ? "Protons" : "Unknown";

        vector<string> fileList = ExpandGlob(filePatterns.at(File));
        if (fileList.empty()) {
            cerr << "[ERROR] No files match: " << filePatterns.at(File) << endl;
            continue;
        }

        podio::ROOTReader reader;
        reader.openFiles(fileList);

        unsigned nEvents = reader.getEntries("events");
        cout << "Total events in chain (" << name << "): " << nEvents
             << " (" << fileList.size() << " files)" << endl;

        cout << "Building HCal detector group:" << endl;
        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

        // ECal is read from rcp.getClusters() - empty list, kept only to match CollectHits signature.
        vector<DetectorAssoc> ecalDetectors;
        vector<std::string> collections = {
            "ReconstructedParticles", 
            "MCParticles", 
            "ReconstructedParticleAssociations", 
            "HcalBarrelTrackClusterMatches", 
            "HcalEndcapNTrackClusterMatches", 
            "LFHCALTrackClusterMatches", 
            "EcalBarrelClusters",
            "EcalBarrelImagingRecHits",
            "EcalBarrelScFiRecHits",
            "EcalEndcapNClusters",
            "EcalEndcapNRecHits",
            "EcalEndcapPClusters",
            "EcalEndcapPRecHits",
            "HcalBarrelClusters",
            "HcalBarrelRecHits",
            "HcalEndcapNClusters",
            "HcalEndcapNMergedHits",
            "LFHCALClusters",
            "LFHCALRecHits",
        };
        int eventID = 0;

        for (unsigned entry = 0; entry < nEvents; ++entry)
        {
            eventID++;
            if (eventID % 50000 == 0) cout << "File " << name << " event: " << eventID << endl;

            podio::Frame frame(reader.readEntry("events", entry, collections));

            if (entry == 0) {
                cout << "Checking collections in first event:" << endl;
                CheckAvailability(hcalDetectors, frame);
            }

            const auto &tracks = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");

            for (size_t particle = 0; particle < tracks.size(); ++particle)
            {
                const auto rcp = tracks[particle];

                auto mom = rcp.getMomentum();
                TLorentzVector Partic;
                Partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
                double trackP   = Partic.P();
                double trackPt  = Partic.Pt();
                double trackEta = Partic.Eta();

                if (trackP <= 0.7) continue;

                // reference point czastki (wierzcholek) - korekta vz per czastka
                auto rp = rcp.getReferencePoint();
                TVector3 refPoint(rp.x, rp.y, rp.z);

                // -------- ECal: rcp.getClusters() --------
                TrackCalHits ecalHits = CollectHits(rcp, ecalDetectors, frame, true, Partic, refPoint);
                TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP);

                // -------- HCal: TrackClusterMatch --------
                TrackCalHits hcalHits = CollectHits(rcp, hcalDetectors, frame, false, Partic, refPoint);
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

                if (File == 0) // Muon file
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
                else // Background file (combined + per species)
                {
                    const int ib = File - 1;   // 0=Pion, 1=Electron, 2=Kaon, 3=Proton

                    // --- combined (all background) ---
                    h_Response_Background->Fill(Pmu);
                    h_Background_Total_vs_Pt->Fill(trackPt);
                    h_Background_Total_vs_P->Fill(trackP);
                    h_Background_Total_vs_Eta->Fill(trackEta);

                    // --- per species ---
                    h_Resp_Bkg[ib]->Fill(Pmu);
                    h_Bkg_Total_vs_Pt[ib]->Fill(trackPt);
                    h_Bkg_Total_vs_P[ib]->Fill(trackP);
                    h_Bkg_Total_vs_Eta[ib]->Fill(trackEta);

                    if (Pmu <= MUON_ID_CUT) // correctly rejected
                    {
                        h_Background_Rejected_vs_Pt->Fill(trackPt);
                        h_Background_Rejected_vs_P->Fill(trackP);
                        h_Background_Rejected_vs_Eta->Fill(trackEta);

                        h_Bkg_Rej_vs_Pt[ib]->Fill(trackPt);
                        h_Bkg_Rej_vs_P[ib]->Fill(trackP);
                        h_Bkg_Rej_vs_Eta[ib]->Fill(trackEta);
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

    // Combined background
    TH1F *h_Background_Rejection_vs_Pt   = (TH1F*)h_Background_Rejected_vs_Pt->Clone("h_Background_Rejection_vs_Pt");
    h_Background_Rejection_vs_Pt->SetTitle("Background Rejection vs p_{T};p_{T} [GeV/c];Rejection Fraction");
    h_Background_Rejection_vs_Pt->Divide(h_Background_Rejected_vs_Pt, h_Background_Total_vs_Pt, 1.0, 1.0, "B");

    TH1F *h_Background_Rejection_vs_P    = (TH1F*)h_Background_Rejected_vs_P->Clone("h_Background_Rejection_vs_P");
    h_Background_Rejection_vs_P->SetTitle("Background Rejection vs p;p [GeV/c];Rejection Fraction");
    h_Background_Rejection_vs_P->Divide(h_Background_Rejected_vs_P, h_Background_Total_vs_P, 1.0, 1.0, "B");

    TH1F *h_Background_Rejection_vs_Eta  = (TH1F*)h_Background_Rejected_vs_Eta->Clone("h_Background_Rejection_vs_Eta");
    h_Background_Rejection_vs_Eta->SetTitle("Background Rejection vs #eta;#eta;Rejection Fraction");
    h_Background_Rejection_vs_Eta->Divide(h_Background_Rejected_vs_Eta, h_Background_Total_vs_Eta, 1.0, 1.0, "B");

    // Per species: h_<Name>_Rejection_vs_{Pt,P,Eta}_Sep
    for (int i = 0; i < NBkg; ++i)
    {
        const char *n = bkgName[i];

        TH1F *r_pt = (TH1F*)h_Bkg_Rej_vs_Pt[i]->Clone(Form("h_%s_Rejection_vs_Pt_Sep", n));
        r_pt->SetTitle(Form("%s Rejection vs p_{T};p_{T} [GeV/c];Rejection Fraction", n));
        r_pt->Divide(h_Bkg_Rej_vs_Pt[i], h_Bkg_Total_vs_Pt[i], 1.0, 1.0, "B");

        TH1F *r_p = (TH1F*)h_Bkg_Rej_vs_P[i]->Clone(Form("h_%s_Rejection_vs_P_Sep", n));
        r_p->SetTitle(Form("%s Rejection vs p;p [GeV/c];Rejection Fraction", n));
        r_p->Divide(h_Bkg_Rej_vs_P[i], h_Bkg_Total_vs_P[i], 1.0, 1.0, "B");

        TH1F *r_eta = (TH1F*)h_Bkg_Rej_vs_Eta[i]->Clone(Form("h_%s_Rejection_vs_Eta_Sep", n));
        r_eta->SetTitle(Form("%s Rejection vs #eta;#eta;Rejection Fraction", n));
        r_eta->Divide(h_Bkg_Rej_vs_Eta[i], h_Bkg_Total_vs_Eta[i], 1.0, 1.0, "B");
    }

    outfile->Write();
    outfile->Close();

    cout << "Efficiency and rejection histograms have been saved to Plots/MuonID_Performance_HighP.root" << endl;
}