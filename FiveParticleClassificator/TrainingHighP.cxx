// TrainingMacro_Podio.cxx
//
// Podio / EDM4eic version of TrainingMacro.cxx.
// The underlying physics, features, eta values, histogram binning, ML tree, and output files are the same
// (ONNX/MLDataHits.root and Plots/Hits/HitsAll_<Muons|Pions>.root); only the data-reading method changes:
// podio::Frame instead of TTreeReader on flat branches.
//
// Cluster selection (same as in ElectronID):
//  - ECal: clusters attached to the reconstructed particle, rcp.getClusters()
//  - HCal: clusters from TrackClusterMatch (EICrecon) for the track of the particle
// Hits are read from cluster.getHits(). No MC association is used to select clusters.
//
// Run: root -l -q TrainingMacro_Podio.cxx++

#include <TH1.h>
#include <TH2.h>
#include <TTree.h>
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
#include <set>
#include <map>
#include <cstdint>

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

// Bundle of all histograms for one group (ECal or HCal).
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

// =====================================================================
// CollectHits - podio version.
//  1) read the TrackClusterMatch collection from the Frame for each detector,
//  2) keep only those matches whose track is the track of the given reconstructed particle,
//  3) read hits from cluster.getHits().
// =====================================================================
TrackCalHits CollectHits(const edm4eic::ReconstructedParticle &rcp,
                         const vector<DetectorAssoc> &detectors,
                         const podio::Frame &frame,
                         bool fromRcpClusters,   // true = ECal (rcp.getClusters()), false = HCal (TrackClusterMatch)
                         const TLorentzVector &Partic,
                         TH1F *hR, TH2F *hEtaR, TH2F *hPhiR, TH2F *hEtaPhi,
                         TH1F *hPhi, TH1F *hZ, TH1F *hE, TH1F *hT,
                         TH1F *hDEta, TH1F *hDPhi)
{
    TrackCalHits result;

    vector<edm4eic::Cluster> clusters;
    set<podio::ObjectID> imagingHitIDs;
    if (fromRcpClusters)
    {
        // ECal (as in ElectronID): clusters attached to the reconstructed particle
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
        // HCal (as in ElectronID): clusters from TrackClusterMatch for the track of this particle
        if (rcp.getTracks().empty()) return result;
        const auto trk = rcp.getTracks().at(0);

        for (const auto &det : detectors)
        {
            if (!det.available) continue;
            for (const auto &m : frame.get<edm4eic::TrackClusterMatchCollection>(det.matchCollName))
                if (m.getTrack() == trk) clusters.push_back(m.getCluster());
        }
    }

    {
        for (const auto &cluster : clusters)
        {
            for (const auto &hit : cluster.getHits())
            {
                if (imagingHitIDs.count(hit.getObjectID())) continue;
                float HitE = hit.getEnergy();
                float HitT = hit.getTime();
                auto  pos  = hit.getPosition();

                if (hT) hT->Fill(HitT);

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

// Same as in TrainingMacro.cxx (pure math on TrackCalHits).
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

    double spreadPhi = acos(cos(hits.phiMax - hits.phiMin));
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

    h.h_Energy       = new TH1F((prefix + "_h_Energy").c_str(), (prefix + " Total Energy;E_{calo} [GeV];Counts").c_str(), 50, 0, 20);
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
    h.h_Phi_DispWeighted    = new TH1F((prefix + "_h_Phi_DispWeighted").c_str(), (prefix + " #phi dispersion (energy-weighted);D_{#phi}^{w} [rad];Counts").c_str(), 60, 0, 0.6);

    return h;
}

void TrainingHighP()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    const long   MAX_EVENTS    = -1;   // -1 = all events in the file/files
    const double MOMENTUM_CUT = 0.7;

    static constexpr int NumOfFiles = 5;
    vector<string> filePatterns(NumOfFiles);
    filePatterns.at(0) = "/run/media/epic/Data/Background/Muon/Podio-1.7.99/reco_*.root";
    filePatterns.at(1) = "/run/media/epic/Data/Background/Pion/Podio-1.7.99/reco_*.root";
    filePatterns.at(2) = "/run/media/epic/Data/Background/Electron/Podio-1.7.99/reco_*.root";
    filePatterns.at(3) = "/run/media/epic/Data/Background/Kaon/Podio-1.7.99/reco_*.root";
    filePatterns.at(4) = "/run/media/epic/Data/Background/Proton/Podio-1.7.99/reco_*.root";

    vector<string> ecalClusterNames = {"EcalBarrelClusters", "EcalEndcapPClusters", "EcalEndcapNClusters"};
    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    // -----------------------------------------------------------------
    // ML TTree: created once, outside the file loop (muons + pions with IsMuon label).
    // -----------------------------------------------------------------
    float ECalEnergy, ECalNumber, ECalEoverP, ECalAvgHitEnergy, ECalSpreadPhi, ECalSpreadEta, ECalSpreadR,
          ECalMaxHitFrac, ECalEnergyStdDev, ECalEnergyConcentration, ECalR_Disp, ECalR_DispWeighted,
          ECalEta_DispWeighted, ECalPhi_DispWeighted;

    float HCalEnergy, HCalNumber, HCalEoverP, HCalAvgHitEnergy, HCalSpreadPhi, HCalSpreadEta, HCalSpreadR,
          HCalMaxHitFrac, HCalEnergyStdDev, HCalEnergyConcentration, HCalR_Disp, HCalR_DispWeighted,
          HCalEta_DispWeighted, HCalPhi_DispWeighted;

    float TrackMomentum, TrackEta;
    float IsMuon, FileIndex;

    TFile *mlFile = new TFile("ONNX/MLDataHighPHits.root", "RECREATE");
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

    MLDataTree->Branch("TrackMomentum", &TrackMomentum, "TrackMomentum/F");
    MLDataTree->Branch("TrackEta", &TrackEta, "TrackEta/F");
    MLDataTree->Branch("IsMuon", &IsMuon, "IsMuon/F");
    MLDataTree->Branch("FileIndex", &FileIndex, "FileIndex/F");

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


        unsigned nEntries = reader.getEntries("events");
        cout << "Total events in chain (" << name << "): " << nEntries << " (" << fileList.size() << " files)" << endl;

        cout << "Building ECal detector group:" << endl;
        vector<DetectorAssoc> ecalDetectors;
        for (auto &n : ecalClusterNames) ecalDetectors.push_back(MakeDetectorAssoc(n));

        cout << "Building HCal detector group:" << endl;
        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

        // Output file - histograms are created after it, so they are written to it.
        TFile *outfile = new TFile(Form("Plots/FeaturesHighP_%s.root", name.c_str()), "RECREATE");

        HistSet ecalHists = MakeHistSet("ECal");
        HistSet hcalHists = MakeHistSet("HCal");

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
            "CentralTrackSegments",
            "CentralCKFTracks",
            "CentralTrackSeeds",
        };

        long nToProcess = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : nEntries;

        set<string> needed;   // #include <set>
        for (long e = 0; e < min<long>(200, nEntries); ++e)
        {
            podio::Frame f(reader.readEntry("events", e, collections));   // wszystko
            auto names = f.getAvailableCollections();

            // mapa ID -> nazwa
            map<uint32_t, string> idToName;
            for (auto &n : names) {
                auto *c = f.get(n);
                if (c) idToName[c->getID()] = n;
            }

            auto scan = [&](const edm4eic::Cluster &cl) {
                for (const auto &h : cl.getHits()) {
                    auto it = idToName.find(h.getObjectID().collectionID);
                    if (it != idToName.end()) needed.insert(it->second);
                }
                auto it = idToName.find(cl.getObjectID().collectionID);
                if (it != idToName.end()) needed.insert(it->second);
            };

            for (const auto &rcp : f.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles"))
                for (const auto &cl : rcp.getClusters()) scan(cl);

            for (auto &n : {"HcalBarrelTrackClusterMatches", "HcalEndcapNTrackClusterMatches", "LFHCALTrackClusterMatches"})
                for (const auto &m : f.get<edm4eic::TrackClusterMatchCollection>(n)) scan(m.getCluster());
        }
        cout << "Kolekcje potrzebne do hitow:" << endl;
        for (auto &n : needed) cout << "   \"" << n << "\"," << endl;

        for (long entry = 0; entry < nToProcess; ++entry)
        {
            if ((entry + 1) % 50000 == 0) cout << "File " << name << " and event number... " << entry + 1 << endl;
            podio::Frame frame(reader.readEntry("events", entry, collections));

            if (entry == 0) {
                cout << "Checking collections in first event:" << endl;
                // ECal uses rcp.getClusters(), no match collection needed
                CheckAvailability(hcalDetectors, frame);
            }
            auto &tracks = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");


            for (size_t particle = 0; particle < tracks.size(); ++particle)
            {
                const auto rcp = tracks[particle];

                auto mom = rcp.getMomentum();
                TLorentzVector Partic;
                Partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
                double trackP   = Partic.P();
                double trackEta = Partic.Eta();

                if (trackP <= MOMENTUM_CUT ) continue;
                //cout<<"blabla"<<endl;
                // -------- ECal --------
                TrackCalHits ecalHits = CollectHits(rcp, ecalDetectors, frame, true, Partic,
                    ecalHists.HitR, ecalHists.hHitEtaR, ecalHists.hHitPhiR, ecalHists.hHitEtaPhi,
                    ecalHists.HitPhi, ecalHists.HitZ, ecalHists.HitEnergy, ecalHists.HitTime,
                    ecalHists.DeltaEta, ecalHists.DeltaPhi);
                TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP, ecalHists);

                // -------- HCal --------
                TrackCalHits hcalHits = CollectHits(rcp, hcalDetectors, frame, false, Partic,
                    hcalHists.HitR, hcalHists.hHitEtaR, hcalHists.hHitPhiR, hcalHists.hHitEtaPhi,
                    hcalHists.HitPhi, hcalHists.HitZ, hcalHists.HitEnergy, hcalHists.HitTime,
                    hcalHists.DeltaEta, hcalHists.DeltaPhi);
                TrackFeatures hcalF = ComputeTrackFeatures(hcalHits, trackP, hcalHists);

                if (ecalF.Number <= 0 && hcalF.Number <= 0) continue;

                ECalEnergy              = ecalF.Energy;
                ECalNumber              = ecalF.Number;
                ECalEoverP              = ecalF.EoverP;
                ECalAvgHitEnergy        = ecalF.AvgHitEnergy;
                ECalSpreadPhi           = ecalF.SpreadPhi;
                ECalSpreadEta           = ecalF.SpreadEta;
                ECalSpreadR             = ecalF.SpreadR;
                ECalMaxHitFrac          = ecalF.MaxHitFrac;
                ECalEnergyStdDev        = ecalF.EnergyStdDev;
                ECalEnergyConcentration = ecalF.EnergyConcentration;
                ECalR_Disp              = ecalF.R_Disp;
                ECalR_DispWeighted      = ecalF.R_DispWeighted;
                ECalEta_DispWeighted    = ecalF.Eta_DispWeighted;
                ECalPhi_DispWeighted    = ecalF.Phi_DispWeighted;

                HCalEnergy              = hcalF.Energy;
                HCalNumber              = hcalF.Number;
                HCalEoverP              = hcalF.EoverP;
                HCalAvgHitEnergy        = hcalF.AvgHitEnergy;
                HCalSpreadPhi           = hcalF.SpreadPhi;
                HCalSpreadEta           = hcalF.SpreadEta;
                HCalSpreadR             = hcalF.SpreadR;
                HCalMaxHitFrac          = hcalF.MaxHitFrac;
                HCalEnergyStdDev        = hcalF.EnergyStdDev;
                HCalEnergyConcentration = hcalF.EnergyConcentration;
                HCalR_Disp              = hcalF.R_Disp;
                HCalR_DispWeighted      = hcalF.R_DispWeighted;
                HCalEta_DispWeighted    = hcalF.Eta_DispWeighted;
                HCalPhi_DispWeighted    = hcalF.Phi_DispWeighted;

                TrackMomentum = static_cast<float>(trackP);
                TrackEta      = static_cast<float>(trackEta);
                IsMuon        = (File == 0) ? 1.f : 0.f;   // File 0 = Muons, File 1 = Pions
                FileIndex     = static_cast<float>(File);

                MLDataTree->Fill();
            }
        }

        cout << "===========================" << endl;
        cout << "End of " << name << " file" << endl;
        cout << "Number of events processed: " << nToProcess << endl;
        cout << "===========================" << endl;

        outfile->Write();
        outfile->Close();
    }

    mlFile->cd();
    MLDataTree->Write();
    mlFile->Close();
}