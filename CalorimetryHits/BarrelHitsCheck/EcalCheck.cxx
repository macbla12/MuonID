// EcalCheck.cxx
//
// Maly skrypt kontrolny: E/p i cechy hitow osobno dla kazdego subdetektora ECal.
// Wybor klastrow jak w TCM: rcp.getClusters(), a przynaleznosc klastra do subdetektora
// ustalana przez porownanie z kolekcja <Detektor>Clusters. Zadnego MC.
// Wszystkie wykresy ladowane do JEDNEGO pliku PDF (strona = czastka x subdetektor,
// ostatnia strona = nalozone E/p wszystkich czastek).
//
// Uruchom: root -l -q EcalCheck.cxx++

#include <TH1F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <glob.h>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <podio/ObjectID.h>
#include <set>

using namespace std;

// ------------------------------- konfiguracja --------------------------------
const vector<string> DETS = {"EcalBarrelClusters", "EcalEndcapNClusters", "EcalEndcapPClusters"};
const vector<string> PART_NAMES = {"Muons", "Muons-Grape", "Electrons", "Kaons", "Protons"};
const vector<string> PART_FILES = {
    "/run/media/epic/Data/Background/Muon/Podio-1.7.99/reco_Muons1.root",
    "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_07*.root",
    "/run/media/epic/Data/Background/Electron/Podio-1.7.99/reco_Electrons1.root",
    "/run/media/epic/Data/Background/Kaon/Podio-1.7.99/reco_Kaons1.root",
    "/run/media/epic/Data/Background/Proton/Podio-1.7.99/reco_Protons1.root"};
const double P_MIN = 0.0, P_MAX = 4.0;   // cut na pedu [GeV]
const long   MAX_EVENTS = 100000;          // -1 = wszystkie
const char  *PDF = "EcalCheck.pdf";
const bool   NORMALIZE = true;   
const int    DET_COLORS[] = {kBlue, kRed, kGreen + 2};  
const bool   SKIP_BARREL_IMAGING_HITS = true;   // true = pomijaj hity z EcalBarrelImagingRecHits

// ------------------------------- histogramy ----------------------------------
enum { kEnergy, kEoverP, kNHits, kHitE, kN };
struct HDef { const char *key, *title; int n; double lo, hi; };
const HDef HD[kN] = {
    {"Energy",  "Sum of hit energy;E [GeV]",         100, 0, 1.5},
    {"EoverP",  "E/p;E/p",                           100, 0, 3},
    {"NHits",   "Hits per track;N_{hits}",            50, 0, 100},
    {"HitE",    "Hit energy;E [GeV]",                100, 0, 0.2},
};

// ------------------------------- pomocnicze ----------------------------------
vector<string> Expand(const string &pat)
{
    vector<string> r; glob_t g;
    glob(pat.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) r.push_back(g.gl_pathv[i]); // poprawiono na push_back
    globfree(&g);
    return r;
}

void PrintPage(TCanvas &c, int &page, int total)
{
    ++page;
    const char *suffix = (page == 1) ? "(" : (page == total ? ")" : "");
    c.Print(Form("%s%s", PDF, suffix), "pdf");
}

// ---------------------------------- main -------------------------------------
void EcalCheck()
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(1110);
    gErrorIgnoreLevel = 1001;   

    const int nP = PART_NAMES.size(), nD = DETS.size();

    vector<vector<vector<TH1F*>>> H(nP, vector<vector<TH1F*>>(nD, vector<TH1F*>(kN)));
    for (int p = 0; p < nP; ++p)
        for (int d = 0; d < nD; ++d)
            for (int k = 0; k < kN; ++k) {
                string nm = PART_NAMES[p] + "_" + DETS[d] + "_" + HD[k].key;
                string ti = PART_NAMES[p] + " / " + DETS[d] + ": " + HD[k].title + ";Counts";
                H[p][d][k] = new TH1F(nm.c_str(), ti.c_str(), HD[k].n, HD[k].lo, HD[k].hi);
            }

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
    collections.insert(collections.end(), DETS.begin(), DETS.end());

    // ----- petla po zdarzeniach -----
    for (int p = 0; p < nP; ++p)
    {
        auto files = Expand(PART_FILES[p]);
        if (files.empty()) { cerr << "[ERROR] brak plikow: " << PART_FILES[p] << endl; continue; }

        podio::ROOTReader reader;
        reader.openFiles(files);
        unsigned nEntries = reader.getEntries("events");
        long nProc = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : nEntries;
        cout << PART_NAMES[p] << ": " << nProc << " zdarzen" << endl;

        for (long ev = 0; ev < nProc; ++ev)
        {
            if ((ev + 1) % 50000 == 0) cout << "  " << PART_NAMES[p] << " event " << ev + 1 << endl;
            podio::Frame frame(reader.readEntry("events", ev, collections));

            auto avail = frame.getAvailableCollections();
            vector<const edm4eic::ClusterCollection*> detColl(nD, nullptr);
            for (int d = 0; d < nD; ++d)
                if (find(avail.begin(), avail.end(), DETS[d]) != avail.end())
                    detColl[d] = &frame.get<edm4eic::ClusterCollection>(DETS[d]);
            // ID hitow z EcalBarrelImagingRecHits (do odfiltrowania)
            set<podio::ObjectID> imagingHitIDs;
            if (SKIP_BARREL_IMAGING_HITS &&
                find(avail.begin(), avail.end(), "EcalBarrelImagingRecHits") != avail.end())
            {
                for (const auto &ih : frame.get<edm4eic::CalorimeterHitCollection>("EcalBarrelImagingRecHits"))
                    imagingHitIDs.insert(ih.getObjectID());
            }

            for (const auto &rcp : frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles"))
            {
                auto m = rcp.getMomentum();
                TLorentzVector v; v.SetPxPyPzE(m.x, m.y, m.z, rcp.getEnergy());
                double P = v.P();
                if (P <= P_MIN || P > P_MAX) continue;

                for (int d = 0; d < nD; ++d)
                {
                    if (!detColl[d]) continue;

                    double sumE = 0, maxE = 0; int nHits = 0;
                    auto &h = H[p][d];

                    for (const auto &cl : rcp.getClusters())
                    {
                        bool inDet = false;
                        for (const auto &dc : *detColl[d]) if (dc == cl) { inDet = true; break; }
                        if (!inDet) continue;

                        for (const auto &hit : cl.getHits())
                        {
                            if (SKIP_BARREL_IMAGING_HITS && imagingHitIDs.count(hit.getObjectID())) continue;   // pomin hity imaging barrel

                            double E = hit.getEnergy();
                            auto pos = hit.getPosition();
                            TVector3 hv(pos.x, pos.y, pos.z);

                            h[kHitE]->Fill(E);
                            sumE += E; maxE = max(maxE, E); ++nHits;
                        }
                    }
                    if (nHits == 0 || sumE <= 0) continue;
                    h[kEnergy]->Fill(sumE);
                    h[kEoverP]->Fill(sumE / P);
                    h[kNHits]->Fill(nHits);
                }
            }
        }
    }

    // ----- rysowanie do jednego PDF -----
    const int total = nP + 1;   
    int page = 0;
    TCanvas c("c", "c", 1800, 1200);

    for (int p = 0; p < nP; ++p)
    {
        c.Clear(); c.Divide(2, 2);
        for (int k = 0; k < kN; ++k)
        {
            c.cd(k + 1);

            double ymax = 0;
            for (int d = 0; d < nD; ++d) {
                auto *h = H[p][d][k];
                if (NORMALIZE && h->Integral() > 0) h->Scale(1.0 / h->Integral());
                ymax = max(ymax, h->GetMaximum());
            }

            TLegend *leg = new TLegend(0.60, 0.70, 0.89, 0.89);
            leg->SetBorderSize(0);
            bool first = true;
            for (int d = 0; d < nD; ++d) {
                auto *h = H[p][d][k];
                h->SetLineColor(DET_COLORS[d % 3]);
                h->SetLineWidth(2);
                h->SetStats(0);
                h->SetTitle(Form("%s: %s;%s", PART_NAMES[p].c_str(),
                                 HD[k].title, NORMALIZE ? "a.u." : "Counts"));
                h->SetMaximum(1.2 * ymax);
                h->Draw(first ? "hist" : "hist same");
                leg->AddEntry(h, Form("%s (N=%d)", DETS[d].c_str(), (int)h->GetEntries()), "l");
                first = false;
            }
            leg->Draw();
        }
        PrintPage(c, page, total);
    }

    c.Clear(); c.Divide(2, 2);
    const int colors[] = {kBlue, kRed, kGreen + 2, kMagenta, kOrange + 1};
    for (int d = 0; d < nD; ++d) {
        c.cd(d + 1);
        TLegend *leg = new TLegend(0.6, 0.65, 0.89, 0.89);
        leg->SetBorderSize(0);
        bool first = true;
        double ymax = 0;
        
        for (int p = 0; p < nP; ++p) {
            auto *h = H[p][d][kEoverP];
            if (h->Integral() > 0) ymax = max(ymax, h->GetMaximum());
        }
        
        for (int p = 0; p < nP; ++p) {
            auto *h = H[p][d][kEoverP];
            h->SetLineColor(colors[p % 5]); 
            h->SetLineWidth(2);
            h->SetTitle(Form("%s: E/p (normalized);E/p;a.u.", DETS[d].c_str()));
            h->SetMaximum(1.2 * ymax);
            h->SetStats(0);
            h->Draw(first ? "hist" : "hist same");
            leg->AddEntry(h, PART_NAMES[p].c_str(), "l");
            first = false;
        }
        leg->Draw();
    }
    PrintPage(c, page, total);

    cout << "Zapisano: " << PDF << " (" << total << " stron)" << endl;
}