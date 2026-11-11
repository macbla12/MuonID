// HitFeatures_Podio.cxx
// Histogramy cech hitow (ECAL/HCAL, negative eta) - wersja podio::Frame / EDM4eic.
// Uruchom: root -l -q HitFeatures_Podio.cxx
// Porownaj z: HitFeatures_Branches.cxx (te same nazwy histogramow i binning).

#include <TH1.h>
#include <TFile.h>
#include <TROOT.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <glob.h>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>
#include <edm4hep/MCParticleCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/MCRecoParticleAssociationCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/MCRecoClusterParticleAssociationCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>

using namespace std;

// ---------------- ustawienia wspolne z wersja branch ----------------
const double TIMING_CUT_NS   = 20.0;
const double MIN_TRACK_P     = 1.0;
const bool   ONLY_NEG_ETA    = true;   // tylko tracki z eta < 0
const int    MAX_EVENTS      = 100000;  // na plik (muony / piony)
const string OUT_NAME        = "HitFeatures_Podio.root";

// ---------------- histogramy dla jednego detektora (identyczne jak w wersji branch) ----------------
struct Hists
{
    TH1F *E, *T, *R, *Z, *Eta, *Phi, *NHits, *SumE;
};

Hists MakeHists(const string &det, const string &tag)
{
    string p = "h_" + det + "_" + tag + "_";
    Hists h;
    h.E     = new TH1F((p + "HitE").c_str(),   (det + " " + tag + " hit energy;E [GeV];Hits").c_str(), 200, 0, 0.5);
    h.T     = new TH1F((p + "HitT").c_str(),   (det + " " + tag + " hit time (before cut);t [ns];Hits").c_str(), 200, 0, 100);
    h.R     = new TH1F((p + "HitR").c_str(),   (det + " " + tag + " hit R;R [mm];Hits").c_str(), 150, 0, 3500);
    h.Z     = new TH1F((p + "HitZ").c_str(),   (det + " " + tag + " hit z;z [mm];Hits").c_str(), 200, -5000, 5000);
    h.Eta   = new TH1F((p + "HitEta").c_str(), (det + " " + tag + " hit #eta;#eta;Hits").c_str(), 100, -4.5, 0.5);
    h.Phi   = new TH1F((p + "HitPhi").c_str(), (det + " " + tag + " hit #phi;#phi;Hits").c_str(), 64, -TMath::Pi(), TMath::Pi());
    h.NHits = new TH1F((p + "NHitsPerTrack").c_str(), (det + " " + tag + " hits per track;N hits;Tracks").c_str(), 100, 0, 100);
    h.SumE  = new TH1F((p + "SumEPerTrack").c_str(),  (det + " " + tag + " sum E per track;#SigmaE [GeV];Tracks").c_str(), 200, 0, 10);
    return h;
}

struct Det
{
    string cluster;
    string assocName;   // "XClusters" -> "XClusterAssociations"
    Hists h;
};

vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t g;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) result.push_back(g.gl_pathv[i]);
    globfree(&g);
    return result;
}

// ---------------- wypelnianie histogramow dla tracka ----------------
void FillDet(Det &d, const edm4hep::MCParticle &simPart, const podio::Frame &frame)
{
    int n = 0;
    double sumE = 0;

    const auto &assocs = frame.get<edm4eic::MCRecoClusterParticleAssociationCollection>(d.assocName);

    for (const auto &assoc : assocs)
    {
        if (assoc.getSim() != simPart) continue;

        for (const auto &hit : assoc.getRec().getHits())
        {
            float E = hit.getEnergy(), t = hit.getTime();
            auto pos = hit.getPosition();

            d.h.T->Fill(t);
            if (t > TIMING_CUT_NS) continue;

            TVector3 v(pos.x, pos.y, pos.z);
            d.h.E->Fill(E);
            d.h.R->Fill(sqrt(pos.x * pos.x + pos.y * pos.y));
            d.h.Z->Fill(pos.z);
            d.h.Eta->Fill(v.Eta());
            d.h.Phi->Fill(v.Phi());
            n++;
            sumE += E;
        }
    }
    if (n > 0) { d.h.NHits->Fill(n); d.h.SumE->Fill(sumE); }
}

void HitFeatures_Podio()
{
    gROOT->SetBatch(kTRUE);

    const vector<string> clusters = {"EcalEndcapNClusters", "HcalEndcapNClusters"};
    const vector<string> patterns = {
        "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_562.root",
          "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_123.root"};
    const vector<string> tags = {"Muons", "Pions"};

    TFile out(OUT_NAME.c_str(), "RECREATE");

    for (size_t f = 0; f < patterns.size(); ++f)
    {
        podio::ROOTReader reader;
        reader.openFiles(ExpandGlob(patterns[f]));
        unsigned nEntries = reader.getEntries("events");
        unsigned nMax = min<unsigned>(nEntries, MAX_EVENTS);

        vector<Det> dets;
        for (const auto &cl : clusters)
        {
            Det d;
            d.cluster   = cl;
            d.assocName = cl.substr(0, cl.size() - 1) + "Associations";
            d.h         = MakeHists(cl, tags[f]);
            dets.push_back(d);
        }

        for (unsigned entry = 0; entry < nMax; ++entry)
        {
            podio::Frame frame(reader.readEntry("events", entry));

            const auto &tracks = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedChargedParticles");
            const auto &tAssoc = frame.get<edm4eic::MCRecoParticleAssociationCollection>("ReconstructedChargedParticleAssociations");

            for (size_t p = 0; p < tracks.size(); ++p)
            {
                auto mom = tracks[p].getMomentum();
                TLorentzVector v;
                v.SetPxPyPzE(mom.x, mom.y, mom.z, tracks[p].getEnergy());
                if (v.P() <= MIN_TRACK_P) continue;
                if (ONLY_NEG_ETA && v.Eta() >= 0) continue;

                if (p >= tAssoc.size()) continue;
                const auto simPart = tAssoc[p].getSim();

                for (auto &d : dets) FillDet(d, simPart, frame);
            }
        }
        cout << tags[f] << ": " << nMax << " events processed" << endl;
    }

    out.Write();
    out.Close();
    cout << "Saved " << OUT_NAME << endl;
}