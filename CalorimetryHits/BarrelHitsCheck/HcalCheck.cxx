// TrainingMacro_Podio_Compare.cxx
//
// Skrypt porównujący cechy HCal oraz pozycje klastrów/hitów względem toru 
// dla standardowych muonów oraz muonów z pliku GRAPE w celu diagnostyki "ogonów".
//
// Uruchomienie: root -l -q TrainingMacro_Podio_Compare.cxx++

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

vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t g;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) result.push_back(g.gl_pathv[i]);
    globfree(&g);
    return result;
}

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
        bool hasMatch = find(names.begin(), names.end(), d.matchCollName) != names.end();
        d.available = hasMatch;
        cout << "  " << d.clusterName << " -> '" << d.matchCollName << "' : "
             << (hasMatch ? "OK" : "MISSING (detector skipped)") << endl;
    }
}

// Zestaw histów dla HCal rozszerzony o DeltaEta i DeltaPhi klastrów/hitów względem tracku
struct HCalHistSet
{
    TH1F *HitR, *HitPhi, *HitZ, *HitEnergy, *HitTime;
    TH1F *DeltaEta, *DeltaPhi;
    TH2F *hHitEtaR, *hHitPhiR, *hHitEtaPhi;

    TH1F *h_Energy, *h_EoverP, *h_AvgHitEnergy, *h_NHits;
    TH1F *h_SpreadPhi, *h_SpreadEta, *h_SpreadR;
    TH1F *h_MaxHitFrac, *h_EnergyStdDev;

    TH1F *h_R_Disp, *h_R_DispWeighted, *h_EnergyConcentration;
    TH1F *h_Eta_DispWeighted, *h_Phi_DispWeighted;

    // Nowe histogramy diagnostyczne dopasowania Klastr-Track
    TH1F *h_ClusterDeltaEta;
    TH1F *h_ClusterDeltaPhi;
    TH2F *h_ClusterDeltaEtaPhi;
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

TrackCalHits CollectHCalHitsAndClusters(const edm4eic::ReconstructedParticle &rcp,
                                       const vector<DetectorAssoc> &detectors,
                                       const podio::Frame &frame,
                                       const TLorentzVector &Partic,
                                       HCalHistSet &h)
{
    TrackCalHits result;
    if (rcp.getTracks().empty()) return result;
    const auto trk = rcp.getTracks().at(0);

    vector<edm4eic::Cluster> clusters;
    for (const auto &det : detectors)
    {
        if (!det.available) continue;
        for (const auto &m : frame.get<edm4eic::TrackClusterMatchCollection>(det.matchCollName))
        {
            if (m.getTrack() == trk)
            {
                auto cl = m.getCluster();
                clusters.push_back(cl);

                // Diagnostyka pozycji klastra względem toru
                auto clPos = cl.getPosition();
                TVector3 clVec(clPos.x, clPos.y, clPos.z);
                double clEta = clVec.Eta();
                double clPhi = clVec.Phi();

                double dEta = clEta - Partic.Eta();
                double dPhi = TVector2::Phi_mpi_pi(clPhi - Partic.Phi());

                if (h.h_ClusterDeltaEta)    h.h_ClusterDeltaEta->Fill(dEta);
                if (h.h_ClusterDeltaPhi)    h.h_ClusterDeltaPhi->Fill(dPhi);
                if (h.h_ClusterDeltaEtaPhi) h.h_ClusterDeltaEtaPhi->Fill(dEta, dPhi);
            }
        }
    }

    for (const auto &cluster : clusters)
    {
        for (const auto &hit : cluster.getHits())
        {
            float HitE = hit.getEnergy();
            float HitT = hit.getTime();
            auto  pos  = hit.getPosition();

            if (h.HitTime) h.HitTime->Fill(HitT);

            TVector3 hitVec(pos.x, pos.y, pos.z);
            double hitEta = hitVec.Eta();
            double hitPhi = hitVec.Phi();
            double R = sqrt(pos.x * pos.x + pos.y * pos.y);

            if (h.HitR)      h.HitR->Fill(R);
            if (h.hHitEtaR)   h.hHitEtaR->Fill(hitEta, R);
            if (h.hHitPhiR)   h.hHitPhiR->Fill(hitPhi, R);
            if (h.hHitEtaPhi) h.hHitEtaPhi->Fill(hitEta, hitPhi);
            if (h.HitPhi)    h.HitPhi->Fill(hitPhi);
            if (h.HitZ)      h.HitZ->Fill(pos.z);
            if (h.HitEnergy) h.HitEnergy->Fill(HitE);

            double dEta = hitEta - Partic.Eta();
            double dPhi = TVector2::Phi_mpi_pi(hitPhi - Partic.Phi());
            if (h.DeltaEta) h.DeltaEta->Fill(dEta);
            if (h.DeltaPhi) h.DeltaPhi->Fill(dPhi);

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

TrackFeatures ComputeTrackFeatures(const TrackCalHits &hits, double trackP, HCalHistSet &h)
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

HCalHistSet MakeHCalHistSet(const string &prefix)
{
    HCalHistSet h;
    h.HitR       = new TH1F((prefix + "_HitR").c_str(), (prefix + " Hit R position;R [mm];Counts").c_str(), 2000, 0, 3200);
    h.HitPhi     = new TH1F((prefix + "_HitPhi").c_str(), (prefix + " Hit Phi position;#phi;Counts").c_str(), 200, -3.15, 3.15);
    h.HitZ       = new TH1F((prefix + "_HitZ").c_str(), (prefix + " Hit Z position;Z [mm];Counts").c_str(), 200, -3000, 3000);
    h.HitEnergy  = new TH1F((prefix + "_HitEnergy").c_str(), (prefix + " Hit Energy;Energy [GeV];Counts").c_str(), 200, 0, 1);
    h.HitTime    = new TH1F((prefix + "_HitTime").c_str(), (prefix + " Hit Time;Time [ns];Counts").c_str(), 200, 0, 200);

    h.DeltaEta   = new TH1F((prefix + "_DeltaEta").c_str(), (prefix + " Hit Delta Eta (hit - track);#Delta#eta;Counts").c_str(), 200, -1, 1);
    h.DeltaPhi   = new TH1F((prefix + "_DeltaPhi").c_str(), (prefix + " Hit Delta Phi (hit - track);#Delta#phi;Counts").c_str(), 200, -1, 1);

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

    // Dodatkowe histy klastrów
    h.h_ClusterDeltaEta    = new TH1F((prefix + "_ClusterDeltaEta").c_str(), (prefix + " Cluster Delta Eta (cluster - track);#Delta#eta_{cluster};Counts").c_str(), 2000, -7, 7);
    h.h_ClusterDeltaPhi    = new TH1F((prefix + "_ClusterDeltaPhi").c_str(), (prefix + " Cluster Delta Phi (cluster - track);#Delta#phi_{cluster};Counts").c_str(), 2000, -7, 7);
    h.h_ClusterDeltaEtaPhi = new TH2F((prefix + "_ClusterDeltaEtaPhi").c_str(), (prefix + " Cluster #Delta#eta vs #Delta#phi;#Delta#eta_{cluster};#Delta#phi_{cluster}").c_str(), 100, -7, 7, 100, -7, 7);

    return h;
}

void HcalCheck()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    const long   MAX_EVENTS    = 10000; 
    const double MOMENTUM_CUT = 0.7;

    // Definiujemy tylko 2 pliki/źródła do porównania: Standardowe Muony vs Muony z GRAPE
    vector<string> filePatterns = {
        "/run/media/epic/Data/Background/Muon/Podio-1.7.99/reco_*.root",       // 0: Standard Muons
        "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_07*.root"  // 1: GRAPE Muons
    };
    vector<string> sampleNames = {"StandardMuons", "GrapeMuons"};

    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    for (size_t File = 0; File < filePatterns.size(); File++)
    {
        string name = sampleNames[File];
        vector<string> fileList = ExpandGlob(filePatterns[File]);
        if (fileList.empty()) {
            cerr << "[ERROR] No files match pattern for " << name << ": " << filePatterns[File] << endl;
            continue;
        }

        podio::ROOTReader reader;
        reader.openFiles(fileList);
        unsigned nEntries = reader.getEntries("events");
        cout << "Processing " << name << " | Total events: " << nEntries << " (" << fileList.size() << " files)" << endl;

        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));

        TFile *outfile = new TFile(Form("HCal_Compare_%s.root", name.c_str()), "RECREATE");
        HCalHistSet hcalHists = MakeHCalHistSet("HCal");

        vector<std::string> collections = {
            "ReconstructedParticles", 
            "MCParticles", 
            "ReconstructedParticleAssociations", 
            "CentralCKFTracks",
            "CentralTrackSeeds",
            

            // HCal Barrel
            "HcalBarrelClusters",
            "HcalBarrelRecHits",
            "HcalBarrelMergedHits",
            "HcalBarrelTrackClusterMatches",

            // HCal Endcap N
            "HcalEndcapNClusters",
            "HcalEndcapNRecHits",
            "HcalEndcapNMergedHits",
            "HcalEndcapNTrackClusterMatches",

            // LFHCAL
            "LFHCALClusters",
            "LFHCALRecHits",
            "LFHCALTrackClusterMatches",
        };

        long nToProcess = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : nEntries;

        for (long entry = 0; entry < nToProcess; ++entry)
        {
            if ((entry + 1) % 5000 == 0) cout << "  -> Event " << entry + 1 << "/" << nToProcess << endl;
            podio::Frame frame(reader.readEntry("events", entry, collections));

            if (entry == 0) {
                cout << "Checking HCal match collections availability for " << name << ":" << endl;
                CheckAvailability(hcalDetectors, frame);
            }

            auto &tracks = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");

            for (size_t particle = 0; particle < tracks.size(); ++particle)
            {
                const auto rcp = tracks[particle];
                auto mom = rcp.getMomentum();
                TLorentzVector Partic;
                Partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
                double trackP = Partic.P();

                if (trackP <= MOMENTUM_CUT || trackP > 4.0) continue;

                // Zbieramy HCal hity, klastry oraz dopasowania do tracku
                TrackCalHits hcalHits = CollectHCalHitsAndClusters(rcp, hcalDetectors, frame, Partic, hcalHists);
                TrackFeatures hcalF = ComputeTrackFeatures(hcalHits, trackP, hcalHists);
            }
        }

        cout << "Finished processing " << name << ". Saving histograms..." << endl;
        outfile->Write();
        outfile->Close();
    }
    cout << "All done! Porównanie zapisane w katalogu jako HCal_Compare_*.root" << endl;
}