// CombinedFeatures_Podio.cxx
//
// Histogramy cech ECal, HCal oraz cech zlaczonych ("Engineered Calorimeter Features", sekcja 1.6.9).
// Bez TTree - tylko histogramy.
//
// Wyjscie (2 pliki, w kazdym 5 katalogow: Muons, Pions, Electrons, Kaons, Protons):
//   Plots/CombinedFeatures_LowP.root   (0 < P <= 0.7 GeV)
//   Plots/CombinedFeatures_HighP.root  (P > 0.7 GeV)
//
// Wybor klastrow (jak w poprzednim makrze):
//  - ECal: rcp.getClusters()  (bez hitow z EcalBarrelImagingRecHits)
//  - HCal: TrackClusterMatch dla tracka tej czastki
//
// Run: root -l -q CombinedFeatures.cxx++

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TDirectory.h>
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
#include <array>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/TrackClusterMatchCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>

using namespace std;

// =====================================================================
// Ustawienia
// =====================================================================
static const double MOMENTUM_CUT = 0.7;   // GeV: granica low / high
static const double E0           = 1.0;   // GeV: skala w E_log = ln(1 + E/E0)

// =====================================================================
// Pomocnicze
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

TH1F *H1(const string &name, const string &title, int nb, double lo, double hi)
{
    return new TH1F(name.c_str(), title.c_str(), nb, lo, hi);
}
TH2F *H2(const string &name, const string &title, int nbx, double xlo, double xhi, int nby, double ylo, double yhi)
{
    return new TH2F(name.c_str(), title.c_str(), nbx, xlo, xhi, nby, ylo, yhi);
}

// =====================================================================
// Zbieranie hitow
// =====================================================================
struct TrackCalHits
{
    double sumEnergy = 0.0;
    double phiMin = 1e9, phiMax = -1e9;
    double etaMin = 1e9, etaMax = -1e9;
    double Rmin = 1e9, Rmax = -1e9;
    double maxHitE = -1.0;
    vector<double> R, dEta, dPhi, E;
};

TrackCalHits CollectHits(const edm4eic::ReconstructedParticle &rcp,
                         const vector<DetectorAssoc> &detectors,
                         const podio::Frame &frame,
                         bool fromRcpClusters,   // true = ECal, false = HCal
                         const TLorentzVector &Partic)
{
    TrackCalHits result;

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

// =====================================================================
// Cechy pojedynczego detektora
// =====================================================================
struct TrackFeatures
{
    float Energy = 0.f, Number = 0.f, EoverP = 0.f, AvgHitEnergy = 0.f;
    float SpreadPhi = 0.f, SpreadEta = 0.f, SpreadR = 0.f;
    float MaxHitFrac = 0.f, EnergyStdDev = 0.f, EnergyConcentration = 0.f;
    float R_Disp = 0.f, R_DispWeighted = 0.f, Eta_DispWeighted = 0.f, Phi_DispWeighted = 0.f;
};

struct HistSet   // histogramy cech pojedynczego detektora
{
    TH1F *h_Energy, *h_EoverP, *h_AvgHitEnergy, *h_NHits;
    TH1F *h_SpreadPhi, *h_SpreadEta, *h_SpreadR;
    TH1F *h_MaxHitFrac, *h_EnergyStdDev;
    TH1F *h_R_Disp, *h_R_DispWeighted, *h_EnergyConcentration;
    TH1F *h_Eta_DispWeighted, *h_Phi_DispWeighted;
};

HistSet MakeHistSet(const string &prefix, double eMax)
{
    HistSet h;
    h.h_Energy       = H1(prefix + "_h_Energy", prefix + " Total Energy;E_{calo} [GeV];Counts", 100, 0, eMax);
    h.h_EoverP       = H1(prefix + "_h_EoverP", prefix + " E/p_{track};E/p;Counts", 60, 0, 3);
    h.h_AvgHitEnergy = H1(prefix + "_h_AvgHitEnergy", prefix + " Average Hit Energy per Track;E_{avg} [GeV];Counts", 50, 0, 1);
    h.h_NHits        = H1(prefix + "_h_NHits", prefix + " Number of Hits per Track;N_{hits};Counts", 50, -0.5, 99.5);
    h.h_SpreadPhi    = H1(prefix + "_h_SpreadPhi", prefix + " Spread in #phi (max-min);#phi_{Spread} [rad];Counts", 50, 0, 1.0);
    h.h_SpreadEta    = H1(prefix + "_h_SpreadEta", prefix + " Spread in #eta (max-min);#eta_{Spread};Counts", 50, 0, 1.0);
    h.h_SpreadR      = H1(prefix + "_h_SpreadR", prefix + " Radial spread of hits (max-min);R_{Spread} [mm];Counts", 50, 0, 2000);
    h.h_MaxHitFrac   = H1(prefix + "_h_MaxHitFrac", prefix + " Max hit energy / total;f_{max};Counts", 50, 0, 1.05);
    h.h_EnergyStdDev = H1(prefix + "_h_EnergyStdDev", prefix + " Std Dev of hit energies;#sigma_{E} [GeV];Counts", 60, 0, 0.3);
    h.h_R_Disp       = H1(prefix + "_h_R_Disp", prefix + " R dispersion (unweighted);D_{R} [mm];Counts", 50, 0, 400);
    h.h_R_DispWeighted      = H1(prefix + "_h_R_DispWeighted", prefix + " R dispersion (energy-weighted);D_{R}^{w} [mm];Counts", 50, 0, 400);
    h.h_EnergyConcentration = H1(prefix + "_h_EnergyConcentration", prefix + " Energy concentration;C_{E};Counts", 50, 0, 1.05);
    h.h_Eta_DispWeighted    = H1(prefix + "_h_Eta_DispWeighted", prefix + " #eta dispersion (energy-weighted);D_{#eta}^{w};Counts", 60, 0, 0.6);
    h.h_Phi_DispWeighted    = H1(prefix + "_h_Phi_DispWeighted", prefix + " #phi dispersion (energy-weighted);D_{#phi}^{w} [rad];Counts", 60, 0, 0.6);
    return h;
}

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

    double spreadPhi = acos(cos(hits.phiMax - hits.phiMin));
    double spreadEta = hits.etaMax - hits.etaMin;
    double spreadR   = hits.Rmax - hits.Rmin;

    double sumRw = 0.0;
    for (int k = 0; k < n; ++k) sumRw += hits.R[k] * hits.E[k];
    double meanR_w = sumRw / sumEnergy;

    double sumR2diff = 0.0, sumR2diffW = 0.0, sumDEta2diffW = 0.0, sumDPhi2diffW = 0.0;
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

    h.h_NHits->Fill(n);
    h.h_AvgHitEnergy->Fill(meanE);
    h.h_EnergyStdDev->Fill(energyStdDev);
    h.h_EnergyConcentration->Fill(energyConcentration);
    h.h_Energy->Fill(sumEnergy);
    h.h_EoverP->Fill(sumEnergy / trackP);
    h.h_SpreadPhi->Fill(spreadPhi);
    h.h_SpreadEta->Fill(spreadEta);
    h.h_SpreadR->Fill(spreadR);
    h.h_MaxHitFrac->Fill(hits.maxHitE / sumEnergy);
    h.h_R_Disp->Fill(R_disp_unweighted);
    h.h_R_DispWeighted->Fill(R_disp_weighted);
    h.h_Eta_DispWeighted->Fill(Eta_disp_weighted);
    h.h_Phi_DispWeighted->Fill(Phi_disp_weighted);

    f.Energy              = sumEnergy;
    f.Number              = n;
    f.EoverP              = sumEnergy / trackP;
    f.AvgHitEnergy        = meanE;
    f.SpreadPhi           = spreadPhi;
    f.SpreadEta           = spreadEta;
    f.SpreadR             = spreadR;
    f.MaxHitFrac          = hits.maxHitE / sumEnergy;
    f.EnergyStdDev        = energyStdDev;
    f.EnergyConcentration = energyConcentration;
    f.R_Disp              = R_disp_unweighted;
    f.R_DispWeighted      = R_disp_weighted;
    f.Eta_DispWeighted    = Eta_disp_weighted;
    f.Phi_DispWeighted    = Phi_disp_weighted;
    return f;
}

// =====================================================================
// Cechy zlaczone ECal + HCal (sekcja 1.6.9)
// =====================================================================
struct CombSet
{
    TH1F *h_fECal, *h_fHCal, *h_Etot;
    TH1F *h_REp;                 // (E/p)_ECal / (E/p)_HCal
    TH1F *h_ElogECal, *h_ElogHCal;
    TH1F *h_DeltaFmax;           // f_max^ECal - f_max^HCal
    TH1F *h_REhit;               // <E_hit>_ECal / <E_hit>_HCal
    TH1F *h_SE;                  // C_E^ECal - C_E^HCal
    TH1F *h_RDRw;                // D_R,w^ECal / D_R,w^HCal
    TH1F *h_HasECal, *h_HasHCal; // wskazniki binarne
    TH1F *h_Category;            // 0 = brak, 1 = tylko ECal, 2 = tylko HCal, 3 = oba
    TH2F *h_fECal_vs_P, *h_ElogECal_vs_ElogHCal, *h_fECal_vs_EoverP;
};

CombSet MakeCombSet(double eMax, double pMax)
{
    CombSet c;
    c.h_fECal  = H1("Comb_fECal", "Energy fraction in ECal;f_{ECal};Counts", 50, 0, 1.05);
    c.h_fHCal  = H1("Comb_fHCal", "Energy fraction in HCal;f_{HCal};Counts", 50, 0, 1.05);
    c.h_Etot   = H1("Comb_Etot", "Total calorimeter energy;E_{tot} [GeV];Counts", 100, 0, eMax);
    c.h_REp    = H1("Comb_REp", "Ratio (E/p)_{ECal} / (E/p)_{HCal};R_{E/p};Counts", 100, 0, 10);
    c.h_ElogECal = H1("Comb_ElogECal", "ECal log energy;ln(1+E_{ECal}/E_{0});Counts", 60, 0, 3);
    c.h_ElogHCal = H1("Comb_ElogHCal", "HCal log energy;ln(1+E_{HCal}/E_{0});Counts", 60, 0, 3);
    c.h_DeltaFmax = H1("Comb_DeltaFmax", "f_{max}^{ECal} - f_{max}^{HCal};#Deltaf;Counts", 60, -1.05, 1.05);
    c.h_REhit  = H1("Comb_REhit", "Ratio of mean hit energy ECal/HCal;R_{E_{hit}};Counts", 100, 0, 10);
    c.h_SE     = H1("Comb_SE", "C_{E}^{ECal} - C_{E}^{HCal};S_{E};Counts", 60, -1.05, 1.05);
    c.h_RDRw   = H1("Comb_RDRw", "Ratio D_{R,w}^{ECal} / D_{R,w}^{HCal};R_{D_{R,w}};Counts", 100, 0, 10);
    c.h_HasECal = H1("Comb_HasECal", "ECal has hits;HasECal;Counts", 2, -0.5, 1.5);
    c.h_HasHCal = H1("Comb_HasHCal", "HCal has hits;HasHCal;Counts", 2, -0.5, 1.5);
    c.h_Category = H1("Comb_Category", "0:none 1:ECal only 2:HCal only 3:both;Category;Counts", 4, -0.5, 3.5);
    c.h_fECal_vs_P = H2("Comb_fECal_vs_P", "f_{ECal} vs p;p [GeV];f_{ECal}", 60, 0, pMax, 50, 0, 1.05);
    c.h_ElogECal_vs_ElogHCal = H2("Comb_ElogECal_vs_ElogHCal", "ln(1+E/E_{0}): ECal vs HCal;HCal;ECal", 60, 0, 3, 60, 0, 3);
    c.h_fECal_vs_EoverP = H2("Comb_fECal_vs_EoverP", "f_{ECal} vs E_{tot}/p;E_{tot}/p;f_{ECal}", 60, 0, 3, 50, 0, 1.05);
    return c;
}

void FillCombined(const TrackFeatures &e, const TrackFeatures &h, double trackP, CombSet &c)
{
    bool hasE = e.Number > 0, hasH = h.Number > 0;

    c.h_HasECal->Fill(hasE ? 1 : 0);
    c.h_HasHCal->Fill(hasH ? 1 : 0);
    c.h_Category->Fill((hasE ? 1 : 0) + (hasH ? 2 : 0));

    double Etot = e.Energy + h.Energy;
    if (Etot <= 0) return;

    double fE = e.Energy / Etot;
    double fH = h.Energy / Etot;

    c.h_Etot->Fill(Etot);
    c.h_fECal->Fill(fE);
    c.h_fHCal->Fill(fH);
    c.h_fECal_vs_P->Fill(trackP, fE);
    c.h_fECal_vs_EoverP->Fill(Etot / trackP, fE);

    // Log energie: dobrze zdefiniowane zawsze (0 gdy brak odpowiedzi)
    double ElogE = log(1.0 + e.Energy / E0);
    double ElogH = log(1.0 + h.Energy / E0);
    c.h_ElogECal->Fill(ElogE);
    c.h_ElogHCal->Fill(ElogH);
    c.h_ElogECal_vs_ElogHCal->Fill(ElogH, ElogE);

    // Cechy porownawcze: tylko gdy sa hity w OBU kalorymetrach,
    // inaczej ksztalt/stosunek jest nieokreslony (tu wlasnie przydaja sie wskazniki binarne).
    if (hasE && hasH)
    {
        if (h.EoverP > 0)        c.h_REp->Fill(e.EoverP / h.EoverP);
        if (h.AvgHitEnergy > 0)  c.h_REhit->Fill(e.AvgHitEnergy / h.AvgHitEnergy);
        if (h.R_DispWeighted > 0) c.h_RDRw->Fill(e.R_DispWeighted / h.R_DispWeighted);
        c.h_DeltaFmax->Fill(e.MaxHitFrac - h.MaxHitFrac);
        c.h_SE->Fill(e.EnergyConcentration - h.EnergyConcentration);
    }
}

// =====================================================================
// Komplet histogramow jednej czastki w jednym zakresie pedu
// =====================================================================
struct ParticleHists
{
    HistSet ecal, hcal;
    CombSet comb;
    TH1F *h_P, *h_Eta;
    TDirectory *dir = nullptr;
};

ParticleHists MakeParticleHists(TFile *file, const string &particleName, bool highP)
{
    ParticleHists p;
    p.dir = file->mkdir(particleName.c_str());
    p.dir->cd();

    double eMax = highP ? 20.0 : 5.0;
    double pMax = highP ? 30.0 : MOMENTUM_CUT;

    p.ecal = MakeHistSet("ECal", eMax);
    p.hcal = MakeHistSet("HCal", eMax);
    p.comb = MakeCombSet(eMax, pMax);
    p.h_P   = H1("TrackMomentum", "Track momentum;p [GeV];Counts", 100, 0, pMax);
    p.h_Eta = H1("TrackEta", "Track #eta;#eta;Counts", 120, -3.5, 3.5);
    return p;
}

// =====================================================================
// MAIN
// =====================================================================
void CombinedFeatures()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    const long MAX_EVENTS = -1;   // -1 = wszystkie

    static constexpr int NumOfFiles = 5;
    vector<string> filePatterns(NumOfFiles);
    filePatterns.at(0) = "/run/media/epic/Data/Background/Muon/Podio-1.7.99/reco_*.root";
    filePatterns.at(1) = "/run/media/epic/Data/Background/Pion/Podio-1.7.99/reco_*.root";
    filePatterns.at(2) = "/run/media/epic/Data/Background/Electron/Podio-1.7.99/reco_*.root";
    filePatterns.at(3) = "/run/media/epic/Data/Background/Kaon/Podio-1.7.99/reco_*.root";
    filePatterns.at(4) = "/run/media/epic/Data/Background/Proton/Podio-1.7.99/reco_*.root";
    const vector<string> names = {"Muons", "Pions", "Electrons", "Kaons", "Protons"};

    vector<string> hcalClusterNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};

    // Dwa pliki wyjsciowe: [0] = low P, [1] = high P
    TFile *outLow  = new TFile("Plots/CombinedFeatures_LowP.root",  "RECREATE");
    TFile *outHigh = new TFile("Plots/CombinedFeatures_HighP.root", "RECREATE");

    // Histogramy: [czastka][0=low,1=high]
    vector<array<ParticleHists, 2>> hists;
    for (int i = 0; i < NumOfFiles; ++i)
    {
        array<ParticleHists, 2> a;
        a[0] = MakeParticleHists(outLow,  names[i], false);
        a[1] = MakeParticleHists(outHigh, names[i], true);
        hists.push_back(a);
    }

    vector<std::string> collections = {
        "ReconstructedParticles",
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

    for (int File = 0; File < NumOfFiles; File++)
    {
        const string &name = names[File];

        vector<string> fileList = ExpandGlob(filePatterns.at(File));
        if (fileList.empty()) {
            cerr << "[ERROR] No files match: " << filePatterns.at(File) << endl;
            continue;
        }

        podio::ROOTReader reader;
        reader.openFiles(fileList);

        unsigned nEntries = reader.getEntries("events");
        cout << "Total events (" << name << "): " << nEntries << " (" << fileList.size() << " files)" << endl;

        vector<DetectorAssoc> hcalDetectors;
        for (auto &n : hcalClusterNames) hcalDetectors.push_back(MakeDetectorAssoc(n));
        vector<DetectorAssoc> ecalDetectors;   // ECal bierze klastry z rcp.getClusters()

        long nToProcess = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : nEntries;

        for (long entry = 0; entry < nToProcess; ++entry)
        {
            if ((entry + 1) % 50000 == 0) cout << "File " << name << " and event number... " << entry + 1 << endl;
            podio::Frame frame(reader.readEntry("events", entry, collections));

            if (entry == 0) {
                cout << "Checking collections in first event:" << endl;
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

                if (trackP <= 0) continue;

                ParticleHists &ph = hists[File][trackP > MOMENTUM_CUT ? 1 : 0];

                TrackCalHits ecalHits = CollectHits(rcp, ecalDetectors, frame, true,  Partic);
                TrackCalHits hcalHits = CollectHits(rcp, hcalDetectors, frame, false, Partic);

                // wymaga: odpowiedz w co najmniej jednym kalorymetrze
                if (ecalHits.E.empty() && hcalHits.E.empty()) continue;

                TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP, ph.ecal);
                TrackFeatures hcalF = ComputeTrackFeatures(hcalHits, trackP, ph.hcal);

                if (ecalF.Number <= 0 && hcalF.Number <= 0) continue;

                ph.h_P->Fill(trackP);
                ph.h_Eta->Fill(trackEta);
                FillCombined(ecalF, hcalF, trackP, ph.comb);
            }
        }

        cout << "===========================" << endl;
        cout << "End of " << name << ", events processed: " << nToProcess << endl;
        cout << "===========================" << endl;
    }

    outLow->Write();
    outLow->Close();
    outHigh->Write();
    outHigh->Close();
}