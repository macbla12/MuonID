// AssocComparison3.cxx
//
// Porównanie trzech metod przypisania klastrów (ECAL, HCAL) do śladu, dla pliku mionowego:
//   (A) McAssoc : klaster z kolekcji <Klastry>Associations, którego sim == MCParticle przypisana do śladu   [PRAWDA MC]
//   (B) dR      : klaster z kolekcji <Klastry> z dR(klaster, ślad) < cut(p), z korektą dPhi na zakrzywienie toru
//   (C) TCM     : TrackClusterMatch z EICrecon (dopasowanie ślad-klaster z projekcji toru na powierzchnię kalorymetru,
//                 bez MC): kolekcje <Klastry bez "Clusters">TrackClusterMatches, wybieramy te z m.getTrack() == track cząstki
//
// Korekta dPhi (teoria):  phi_exp = phi_trk + q * asin( 0.3*B*R / (2*pT) ),  R = promień klastra [m]
// Okno dR zależne od pędu (rozpraszanie wielokrotne ~ 1/p):  cut(p) = cut * max(1, P_REF / p)
//
// Wyniki: Plots/AssocComparison3.root + 4 pliki PNG.
// Uruchomienie: root -l -q AssocComparison3.cxx++

#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TLine.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TROOT.h>
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

#include <edm4hep/MCParticleCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/MCRecoParticleAssociationCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/MCRecoClusterParticleAssociationCollection.h>
#include <edm4eic/TrackClusterMatchCollection.h>

using namespace std;

// ----------------------------- Konfiguracja -----------------------------
const string MUON_FILES      = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
const string OUT_DIR         = "Plots";
const long   MAX_EVENTS      = 30000;   // -1 = wszystkie
const double MIN_P           = 0.0;      // [GeV]
const double B_FIELD_T       = 1.7;      // [T]
const bool   APPLY_DPHI_CORR = true;     // false = "czyste" dR bez korekty
const bool   P_DEPENDENT_CUT = true;     // poszerzanie okna dR dla niskiego p
const double P_REF           = 3.0;      // [GeV] poniżej tego pędu okno rośnie jak P_REF/p

// ----------------------------- Struktury -----------------------------
struct Detector {
    string cluster, assoc, match;
    bool   ok = true;        // istnieje kolekcja klastrów i asocjacji
    bool   matchOk = true;   // istnieje kolekcja TrackClusterMatch
};

struct Group {                       // ECAL albo HCAL
    string name;
    double dRcut;
    vector<Detector> dets;
    TH1F *dE, *dE_tcm, *category, *dR_all, *dR_mc, *dR_mc_raw, *dR_tcm;
    TProfile *eff_dR, *eff_tcm;
};

// "XClusters" -> asocjacje "XClusterAssociations", dopasowania "XTrackClusterMatches"
Detector MakeDet(const string &c) {
    Detector d;
    d.cluster = c;
    d.assoc   = c.substr(0, c.size() - 1) + "Associations";
    d.match   = c.substr(0, c.size() - 8) + "TrackClusterMatches";   // "Clusters" = 8 znaków
    return d;
}

vector<string> ExpandGlob(const string &pattern) {
    vector<string> out; glob_t g;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) out.push_back(g.gl_pathv[i]);
    globfree(&g);
    return out;
}

void CheckAvailability(Group &g, const podio::Frame &frame) {
    auto names = frame.getAvailableCollections();
    auto has = [&](const string &n) { return find(names.begin(), names.end(), n) != names.end(); };
    for (auto &d : g.dets) {
        d.ok      = has(d.cluster) && has(d.assoc);
        d.matchOk = has(d.match);
        cout << "  [" << g.name << "] " << d.cluster << " : " << (d.ok ? "OK" : "MISSING (skipped)")
             << " | " << d.match << " : " << (d.matchOk ? "OK" : "MISSING (no TCM for this detector)") << endl;
    }
}

Group MakeGroup(const string &name, double cut, const vector<string> &clusters) {
    Group g; g.name = name; g.dRcut = cut;
    for (auto &c : clusters) g.dets.push_back(MakeDet(c));

    g.dE       = new TH1F((name + "_dE").c_str(),
                 (name + ": E(method) - E(McAssoc) per track;#DeltaE [GeV];Tracks").c_str(), 100, -10, 10);
    g.dE_tcm   = new TH1F((name + "_dE_tcm").c_str(), "", 100, -10, 10);

    g.category = new TH1F((name + "_cat").c_str(),
                 (name + ": clusters by method;;Clusters").c_str(), 7, 0, 7);
    const char *labels[7] = {"Mc", "dR", "Mc+dR", "TCM", "Mc+TCM", "dR+TCM", "all 3"};   // maska: Mc=1, dR=2, TCM=4
    for (int b = 0; b < 7; ++b) g.category->GetXaxis()->SetBinLabel(b + 1, labels[b]);

    g.dR_all    = new TH1F((name + "_dR_all").c_str(),    (name + ": dR(cluster,track);dR;Clusters").c_str(), 200, 0, 1.5);
    g.dR_mc     = new TH1F((name + "_dR_mc").c_str(),     "", 200, 0, 1.5);
    g.dR_mc_raw = new TH1F((name + "_dR_mc_raw").c_str(), "", 200, 0, 1.5);
    g.dR_tcm    = new TH1F((name + "_dR_tcm").c_str(),    "", 200, 0, 1.5);

    g.eff_dR  = new TProfile((name + "_eff_dR").c_str(),
                (name + ": fraction of McAssoc clusters recovered vs p_{T};p_{T} [GeV];Efficiency").c_str(), 40, 0, 10, 0, 1.0);
    g.eff_tcm = new TProfile((name + "_eff_tcm").c_str(), "", 40, 0, 10, 0, 1.0);
    return g;
}

// Zakrzywienie toru: przesunięcie phi między wierzchołkiem a promieniem R [mm]
double BendDphi(double q, double pT, double R_mm) {
    double x = 0.3 * B_FIELD_T * (R_mm * 1e-3) / (2.0 * pT);
    return q * asin(min(x, 0.99));
}

bool Contains(const vector<edm4eic::Cluster> &v, const edm4eic::Cluster &c) {
    return find(v.begin(), v.end(), c) != v.end();
}

// ----------------------------- Główna funkcja -----------------------------
void AssocComparison()
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    gSystem->mkdir(OUT_DIR.c_str(), true);

    auto files = ExpandGlob(MUON_FILES);
    if (files.empty()) { cerr << "[ERROR] brak plików: " << MUON_FILES << endl; return; }

    podio::ROOTReader reader;
    reader.openFiles(files);
    const unsigned nEntries = reader.getEntries("events");
    const long nProc = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : nEntries;
    cout << "Events: " << nEntries << " (" << files.size() << " files), processing " << nProc << endl;

    TFile out((OUT_DIR + "/AssocComparison.root").c_str(), "RECREATE");

    Group ecal = MakeGroup("ECal", 0.6, {"EcalBarrelClusters", //"EcalBarrelImagingClusters", "EcalBarrelScFiClusters",
                                          "EcalEndcapPClusters", "EcalEndcapNClusters"});
    Group hcal = MakeGroup("HCal", 0.6, {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"});
    vector<Group*> groups = {&ecal, &hcal};

    // 2D: pT vs dPhi dla klastrów przypisanych przez McAssoc (ECAL+HCAL razem)
    auto *h2_raw  = new TH2F("h2_pT_dPhi_raw",  "Muons, McAssoc clusters: p_{T} vs #Delta#phi (no correction);#Delta#phi = #phi_{cl}-#phi_{trk} [rad];p_{T} [GeV]",
                             200, -1.0, 1.0, 100, 0, 20);
    auto *h2_corr = new TH2F("h2_pT_dPhi_corr", "Muons, McAssoc clusters: p_{T} vs #Delta#phi (corrected);#Delta#phi_{corr} [rad];p_{T} [GeV]",
                             200, -1.0, 1.0, 100, 0, 20);
    auto *h2_qpt  = new TH2F("h2_qOverPt_dPhi", "Muons, McAssoc clusters: q/p_{T} vs #Delta#phi (no correction);#Delta#phi [rad];q/p_{T} [1/GeV]",
                             200, -1.0, 1.0, 100, -1.5, 1.5);

    long nMisaligned = 0, nTracks = 0;

    for (long entry = 0; entry < nProc; ++entry)
    {
        if ((entry + 1) % 50000 == 0) cout << "event " << entry + 1 << endl;
        podio::Frame frame(reader.readEntry("events", entry));

        if (entry == 0) {
            for (auto *g : groups) CheckAvailability(*g, frame);
            cout << "Available TrackClusterMatch collections in file:" << endl;
            for (auto &n : frame.getAvailableCollections())
                if (n.find("TrackClusterMatch") != string::npos) cout << "  " << n << endl;
        }

        const auto &tracks     = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedChargedParticles");
        const auto &trackAssoc = frame.get<edm4eic::MCRecoParticleAssociationCollection>("ReconstructedChargedParticleAssociations");

        for (size_t i = 0; i < tracks.size(); ++i)
        {
            const auto rcp = tracks[i];
            const auto mom = rcp.getMomentum();
            const TVector3 p3(mom.x, mom.y, mom.z);
            const double p = p3.Mag(), pT = p3.Perp();
            if (p <= MIN_P || pT <= 0) continue;

            if (i >= trackAssoc.size()) continue;
            if (trackAssoc[i].getRec() != rcp) nMisaligned++;
            const auto sim = trackAssoc[i].getSim();

            ++nTracks;
            const double trkEta = p3.Eta(), trkPhi = p3.Phi(), q = rcp.getCharge();
            const double cutScale = P_DEPENDENT_CUT ? max(1.0, P_REF / p) : 1.0;   
            const bool   hasTrk   = !rcp.getTracks().empty();

            for (auto *g : groups)
            {
                const double dRcut = g->dRcut * cutScale;
                double E_mc = 0, E_dR = 0, E_tcm = 0;
                int nMc = 0, nMcDR = 0, nMcTCM = 0;

                for (const auto &det : g->dets)
                {
                    if (!det.ok) continue;

                    // (A) klastry z McAssoc dla tej cząstki MC
                    vector<edm4eic::Cluster> mcClusters;
                    for (const auto &a : frame.get<edm4eic::MCRecoClusterParticleAssociationCollection>(det.assoc))
                        if (a.getSim() == sim) mcClusters.push_back(a.getRec());

                    // (C) klastry z TrackClusterMatch (EICrecon) dla śladu tej cząstki
                    vector<edm4eic::Cluster> tcmClusters;
                    if (hasTrk && det.matchOk) {
                        const auto trk = rcp.getTracks().at(0);
                        for (const auto &m : frame.get<edm4eic::TrackClusterMatchCollection>(det.match))
                            if (m.getTrack() == trk) tcmClusters.push_back(m.getCluster());
                    }

                    // (B) wszystkie klastry -> dR
                    for (const auto &cl : frame.get<edm4eic::ClusterCollection>(det.cluster))
                    {
                        const auto pos = cl.getPosition();
                        const TVector3 v(pos.x, pos.y, pos.z);
                        const double R = v.Perp();

                        const double dEta    = v.Eta() - trkEta;
                        const double dPhiRaw = TVector2::Phi_mpi_pi(v.Phi() - trkPhi);
                        const double dPhi    = APPLY_DPHI_CORR
                                             ? TVector2::Phi_mpi_pi(v.Phi() - (trkPhi + BendDphi(q, pT, R)))
                                             : dPhiRaw;
                        const double dRraw = hypot(dEta, dPhiRaw);
                        const double dR    = hypot(dEta, dPhi);

                        const bool inMc  = Contains(mcClusters, cl);
                        const bool inTCM = Contains(tcmClusters, cl);
                        const bool inDR  = dR < dRcut;
                        const double E   = cl.getEnergy();

                        g->dR_all->Fill(dR);
                        if (inMc) {
                            g->dR_mc->Fill(dR);
                            g->dR_mc_raw->Fill(dRraw);
                            h2_raw ->Fill(dPhiRaw, pT);
                            h2_corr->Fill(dPhi, pT);
                            h2_qpt ->Fill(dPhiRaw, q / pT);
                            E_mc += E; ++nMc;
                            if (inDR)  ++nMcDR;
                            if (inTCM) ++nMcTCM;
                        }
                        if (inDR)  E_dR  += E;
                        if (inTCM) { E_tcm += E; g->dR_tcm->Fill(dR); }

                        const int mask = (inMc ? 1 : 0) | (inDR ? 2 : 0) | (inTCM ? 4 : 0);
                        if (mask) g->category->Fill(mask - 1);
                    }
                }

                if (E_mc > 0 || E_dR > 0 || E_tcm > 0) {
                    g->dE->Fill(E_dR - E_mc);
                    g->dE_tcm->Fill(E_tcm - E_mc);
                }
                if (nMc > 0) {   // skuteczność odzyskania klastrów z McAssoc w funkcji pT
                    g->eff_dR ->Fill(pT, double(nMcDR)  / nMc);
                    g->eff_tcm->Fill(pT, double(nMcTCM) / nMc);
                }
            }
        }
    }

    cout << "Muon tracks: " << nTracks << ", track<->assoc misaligned: " << nMisaligned << " (should be 0)" << endl;

    // ----------------------------- Rysunki -----------------------------
    // 1) McAssoc vs dR vs TCM
    TCanvas c1("c1", "McAssoc vs dR vs TCM", 1300, 900);
    c1.Divide(2, 2);
    int pad = 1;
    for (auto *g : groups) {
        c1.cd(pad++); gPad->SetLogy();
        g->dE->SetLineColor(kBlue + 1); g->dE_tcm->SetLineColor(kRed + 1);
        g->dE->SetLineWidth(2);         g->dE_tcm->SetLineWidth(2);
        g->dE->Draw("HIST"); g->dE_tcm->Draw("HIST SAME");
        auto *leg = new TLegend(0.60, 0.74, 0.89, 0.88);
        leg->AddEntry(g->dE, "dR - McAssoc", "l");
        leg->AddEntry(g->dE_tcm, "TCM - McAssoc", "l");
        leg->Draw();
    }
    for (auto *g : groups) {
        c1.cd(pad++);
        g->category->SetFillColor(kAzure - 4);
        g->category->GetXaxis()->SetLabelSize(0.05);
        g->category->Draw("HIST TEXT0");
    }
    c1.SaveAs((OUT_DIR + "/Compare_McAssoc_dR_TCM.png").c_str());

    // 2) rozkłady dR (linia = bazowe cięcie dla p >= P_REF)
    TCanvas c2("c2", "dR", 1200, 500);
    c2.Divide(2, 1);
    pad = 1;
    for (auto *g : groups) {
        c2.cd(pad++); gPad->SetLogy();
        g->dR_all->SetLineColor(kBlack);   g->dR_mc->SetLineColor(kRed + 1);
        g->dR_mc_raw->SetLineColor(kBlue + 1);   g->dR_tcm->SetLineColor(kGreen + 2);
        g->dR_all->SetTitle((g->name + Form(": dR distribution (base cut = %.2f)", g->dRcut) + ";dR;Clusters").c_str());
        g->dR_all->Draw("HIST"); g->dR_mc->Draw("HIST SAME"); g->dR_mc_raw->Draw("HIST SAME"); g->dR_tcm->Draw("HIST SAME");
        TLine *l = new TLine(g->dRcut, 0, g->dRcut, g->dR_all->GetMaximum());
        l->SetLineStyle(2); l->Draw();
        auto *leg = new TLegend(0.55, 0.64, 0.89, 0.88);
        leg->AddEntry(g->dR_all, "all clusters (corrected)", "l");
        leg->AddEntry(g->dR_mc, "McAssoc (corrected)", "l");
        leg->AddEntry(g->dR_mc_raw, "McAssoc (no dPhi corr.)", "l");
        leg->AddEntry(g->dR_tcm, "TCM (corrected)", "l");
        leg->Draw();
    }
    c2.SaveAs((OUT_DIR + "/dR_distributions.png").c_str());

    // 3) pT vs dPhi (efekt pola magnetycznego)
    TCanvas c3("c3", "pT vs dPhi", 1800, 550);
    c3.Divide(3, 1);
    c3.cd(1); gPad->SetLogz(); h2_raw ->Draw("COLZ");
    c3.cd(2); gPad->SetLogz(); h2_corr->Draw("COLZ");
    c3.cd(3); gPad->SetLogz(); h2_qpt ->Draw("COLZ");
    c3.SaveAs((OUT_DIR + "/pT_vs_dPhi.png").c_str());

    // 4) skuteczność odzyskania klastrów z McAssoc vs pT: gdzie metody tracą na niskim pT
    TCanvas c4("c4", "efficiency vs pT", 1200, 500);
    c4.Divide(2, 1);
    pad = 1;
    for (auto *g : groups) {
        c4.cd(pad++);
        g->eff_dR->SetLineColor(kBlue + 1);  g->eff_tcm->SetLineColor(kRed + 1);
        g->eff_dR->SetLineWidth(2);          g->eff_tcm->SetLineWidth(2);
        g->eff_dR->SetMinimum(0); g->eff_dR->SetMaximum(1.1);
        g->eff_dR->Draw("E"); g->eff_tcm->Draw("E SAME");
        auto *leg = new TLegend(0.60, 0.20, 0.89, 0.34);
        leg->AddEntry(g->eff_dR, "dR", "l");
        leg->AddEntry(g->eff_tcm, "TCM (EICrecon)", "l");
        leg->Draw();
    }
    c4.SaveAs((OUT_DIR + "/efficiency_vs_pT.png").c_str());

    out.cd();
    out.Write();
    out.Close();
}