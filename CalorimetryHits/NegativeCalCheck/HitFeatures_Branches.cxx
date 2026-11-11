// HitFeatures_Branches.cxx
// Histogramy cech hitow (ECAL/HCAL, negative eta) - wersja TTreeReader / flat branche.
// Uruchom: root -l -q HitFeatures_Branches.cxx
// Porownaj z: HitFeatures_Podio.cxx (te same nazwy histogramow i binning).

#include <TH1.h>
#include <TFile.h>
#include <TROOT.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <cmath>

using namespace std;

// ---------------- ustawienia wspolne z wersja podio ----------------
const double TIMING_CUT_NS   = 20.0;
const double MIN_TRACK_P     = 1.0;
const bool   ONLY_NEG_ETA    = true;   // tylko tracki z eta < 0
const int    MAX_EVENTS      = 10000;  // na plik (muony / piony)
const string OUT_NAME        = "HitFeatures_Branches.root";

// Kolekcja hitow, na ktora wskazuje "<cluster>_hits.index", NIE zawsze nazywa sie <X>RecHits.
// HcalEndcapN klastruje na HcalEndcapNMergedHits (CalorimeterHitsMerger po RecHits).
// Domyslnie (brak wpisu) uzywane jest <X>RecHits.
// Potwierdzone Diag_Podio: collID hitow HcalEndcapNClusters == HcalEndcapNMergedHits.
const map<string, string> HIT_COLL = {
    {"HcalEndcapNClusters", "HcalEndcapNMergedHits"},
};

// ---------------- histogramy dla jednego detektora ----------------
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

// ---------------- reader dla jednego detektora ----------------
struct Det
{
    string cluster;
    unique_ptr<TTreeReaderArray<int>>          assoc;
    unique_ptr<TTreeReaderArray<unsigned int>> hitsBegin, hitsEnd;
    unique_ptr<TTreeReaderArray<int>>          hitsIndex;
    unique_ptr<TTreeReaderArray<float>>        E, T, X, Y, Z;
    Hists h;
};

void InitDet(Det &d, TTreeReader &r, const string &cl, const string &tag)
{
    d.cluster = cl;
    string rec = cl;
    size_t pos = rec.rfind("Clusters");
    if (pos != string::npos) rec.replace(pos, 8, "RecHits");
    auto it = HIT_COLL.find(cl);
    if (it != HIT_COLL.end()) rec = it->second;
    cout << "  " << cl << " -> hits from '" << rec << "'" << endl;

    d.assoc     = make_unique<TTreeReaderArray<int>>(r, ("_" + cl.substr(0, cl.size() - 1) + "Associations_sim.index").c_str());
    d.hitsBegin = make_unique<TTreeReaderArray<unsigned int>>(r, (cl + ".hits_begin").c_str());
    d.hitsEnd   = make_unique<TTreeReaderArray<unsigned int>>(r, (cl + ".hits_end").c_str());
    d.hitsIndex = make_unique<TTreeReaderArray<int>>(r, ("_" + cl + "_hits.index").c_str());
    d.E = make_unique<TTreeReaderArray<float>>(r, (rec + ".energy").c_str());
    d.T = make_unique<TTreeReaderArray<float>>(r, (rec + ".time").c_str());
    d.X = make_unique<TTreeReaderArray<float>>(r, (rec + ".position.x").c_str());
    d.Y = make_unique<TTreeReaderArray<float>>(r, (rec + ".position.y").c_str());
    d.Z = make_unique<TTreeReaderArray<float>>(r, (rec + ".position.z").c_str());
}

// ---------------- wypelnianie histogramow dla tracka ----------------
void FillDet(Det &d, int simuID)
{
    int n = 0;
    double sumE = 0;

    for (size_t ic = 0; ic < d.assoc->GetSize(); ++ic)
    {
        if ((*d.assoc)[ic] != simuID) continue;

        for (unsigned int i = (*d.hitsBegin)[ic]; i < (*d.hitsEnd)[ic]; ++i)
        {
            int idx = (*d.hitsIndex)[i];
            float E = (*d.E)[idx], t = (*d.T)[idx];
            float x = (*d.X)[idx], y = (*d.Y)[idx], z = (*d.Z)[idx];

            d.h.T->Fill(t);
            if (t > TIMING_CUT_NS) continue;

            TVector3 v(x, y, z);
            d.h.E->Fill(E);
            d.h.R->Fill(sqrt(x * x + y * y));
            d.h.Z->Fill(z);
            d.h.Eta->Fill(v.Eta());
            d.h.Phi->Fill(v.Phi());
            n++;
            sumE += E;
        }
    }
    if (n > 0) { d.h.NHits->Fill(n); d.h.SumE->Fill(sumE); }
}

void HitFeatures_Branches()
{
    gROOT->SetBatch(kTRUE);

    const vector<string> ecal = {"EcalEndcapNClusters"};
    const vector<string> hcal = {"HcalEndcapNClusters"};
    const vector<string> patterns = {
        "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_562.root",
          "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_123.root"};
    const vector<string> tags = {"Muons", "Pions"};

    TFile out(OUT_NAME.c_str(), "RECREATE");

    for (size_t f = 0; f < patterns.size(); ++f)
    {
        TChain chain("events");
        chain.Add(patterns[f].c_str());
        TTreeReader reader(&chain);

        TTreeReaderArray<float> px(reader, "ReconstructedChargedParticles.momentum.x");
        TTreeReaderArray<float> py(reader, "ReconstructedChargedParticles.momentum.y");
        TTreeReaderArray<float> pz(reader, "ReconstructedChargedParticles.momentum.z");
        TTreeReaderArray<float> en(reader, "ReconstructedChargedParticles.energy");
        TTreeReaderArray<int>   simAssoc(reader, "_ReconstructedChargedParticleAssociations_sim.index");

        vector<Det> dets(ecal.size() + hcal.size());
        for (size_t i = 0; i < dets.size(); ++i)
        {
            bool isEcal = i < ecal.size();
            const string &cl = isEcal ? ecal[i] : hcal[i - ecal.size()];
            InitDet(dets[i], reader, cl, tags[f]);
            dets[i].h = MakeHists(cl, tags[f]);
        }

        int ev = 0;
        while (reader.Next())
        {
            if (++ev > MAX_EVENTS) break;

            for (size_t p = 0; p < px.GetSize(); ++p)
            {
                TLorentzVector v;
                v.SetPxPyPzE(px[p], py[p], pz[p], en[p]);
                if (v.P() <= MIN_TRACK_P) continue;
                if (ONLY_NEG_ETA && v.Eta() >= 0) continue;

                int simuID = simAssoc[p];
                for (auto &d : dets) FillDet(d, simuID);
            }
        }
        cout << tags[f] << ": " << ev - 1 << " events processed" << endl;
    }

    out.Write();
    out.Close();
    cout << "Saved " << OUT_NAME << endl;
}