// MCDistributions_Podio.cxx
//
// Zapisuje rozkłady MC (eta, p, pT, phi, eta vs p) dla każdego typu cząstki.
// Tylko cząstki pierwotne: generatorStatus == 1.
// Wynik: Plots/MCDistributions.root, w nim katalog na każdy typ cząstki.
//
// Run: root -l -q -b MCDistributions_Podio.cxx++

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TROOT.h>
#include <TMath.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <glob.h>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include <edm4hep/MCParticleCollection.h>

using namespace std;

vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t g;
    if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &g) == 0)
        for (size_t i = 0; i < g.gl_pathc; ++i) result.push_back(g.gl_pathv[i]);
    globfree(&g);
    return result;
}

void MCDistributions_Podio()
{
    gROOT->SetBatch(kTRUE);

    const long MAX_EVENTS = -1;   // -1 = wszystkie eventy
    const int  PDG_FILTER_OFF = 0; // zostaw 0, żeby nie filtrować po PDG

    const vector<string> names = {"Muons", "Pions", "Electrons", "Kaons", "Protons"};
    const vector<string> patterns = {
        "/run/media/epic/Data/Background/Muon/Podio-1.7.99/reco_*.root",
        "/run/media/epic/Data/Background/Pion/Podio-1.7.99/reco_*.root",
        "/run/media/epic/Data/Background/Electron/Podio-1.7.99/reco_*.root",
        "/run/media/epic/Data/Background/Kaon/Podio-1.7.99/reco_*.root",
        "/run/media/epic/Data/Background/Proton/Podio-1.7.99/reco_*.root"};

    TFile *out = new TFile("Plots/MCDistributions.root", "RECREATE");

    for (size_t i = 0; i < names.size(); ++i)
    {
        const string &name = names[i];

        vector<string> fileList = ExpandGlob(patterns[i]);
        if (fileList.empty()) {
            cerr << "[ERROR] No files match: " << patterns[i] << endl;
            continue;
        }

        out->cd();
        TDirectory *dir = out->mkdir(name.c_str());
        dir->cd();

        // Histogramy należą jawnie do katalogu danego typu cząstki
        TH1F *hEta = new TH1F("MC_Eta", (name + " MC #eta;#eta;Counts").c_str(), 150, -3, 3);
        TH1F *hP   = new TH1F("MC_P",   (name + " MC p;p [GeV];Counts").c_str(), 200, 0, 20);
        TH1F *hPt  = new TH1F("MC_Pt",  (name + " MC p_{T};p_{T} [GeV];Counts").c_str(), 200, 0, 20);
        TH1F *hPhi = new TH1F("MC_Phi", (name + " MC #phi;#phi [rad];Counts").c_str(), 180, -TMath::Pi(), TMath::Pi());
        TH2F *hEtaP = new TH2F("MC_EtaVsP", (name + " MC #eta vs p;#eta;p [GeV]").c_str(),
                               150, -3, 3, 200, 0, 20);
        TH1F *hPDG = new TH1F("MC_PDG", (name + " PDG of primaries;PDG;Counts").c_str(), 1000, -500, 500);
        TH1F *hNPrim = new TH1F("MC_NPrimaries", (name + " primaries per event;N;Events").c_str(), 20, -0.5, 19.5);

        podio::ROOTReader reader;
        reader.openFiles(fileList);

        const unsigned nEntries = reader.getEntries("events");
        const long nToProcess = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : (long)nEntries;
        cout << name << ": " << nEntries << " events in " << fileList.size() << " files, processing "
             << nToProcess << endl;

        for (long entry = 0; entry < nToProcess; ++entry)
        {
            if ((entry + 1) % 50000 == 0) cout << name << " event " << entry + 1 << endl;

            podio::Frame frame(reader.readEntry("events", entry, {"MCParticles"}));
            const auto &mc = frame.get<edm4hep::MCParticleCollection>("MCParticles");

            int nPrim = 0;
            for (const auto &p : mc)
            {
                if (p.getGeneratorStatus() != 1) continue;
                (void)PDG_FILTER_OFF;

                const auto mom = p.getMomentum();
                const double px = mom.x, py = mom.y, pz = mom.z;
                const double pt = sqrt(px * px + py * py);
                const double ptot = sqrt(pt * pt + pz * pz);
                if (ptot <= 0 || pt <= 0) continue;

                const double eta = asinh(pz / pt);
                const double phi = atan2(py, px);

                hEta->Fill(eta);
                hP->Fill(ptot);
                hPt->Fill(pt);
                hPhi->Fill(phi);
                hEtaP->Fill(eta, ptot);
                hPDG->Fill(p.getPDG());
                ++nPrim;
            }
            hNPrim->Fill(nPrim);
        }

        dir->Write();
        cout << "Done: " << name << endl;
    }

    out->Close();
    delete out;
}