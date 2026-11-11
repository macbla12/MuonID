// TimeCutCheck.cxx
// Sprawdzenie sensu cięcia na t: SpreadPhi / SpreadEta / E/p vs t_max (najpóźniejszy hit w tracku).
// Run: root -l -q TimeCutCheck.cxx++

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TLine.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TMath.h>
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
#include <edm4eic/TrackClusterMatchCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>

using namespace std;

static const double CUT_NS = 20.0;   // obecne cięcie (tylko do narysowania linii)
static const double T_MAX  = 60.0;   // zakres osi t

vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t g;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) result.push_back(g.gl_pathv[i]);
    globfree(&g);
    return result;
}

struct Det { string cluster, match; bool available = true; };

Det MakeDet(const string &c)
{
    Det d; d.cluster = c;
    d.match = c.substr(0, c.size() - string("Clusters").size()) + "TrackClusterMatches";
    return d;
}

// Spread w phi odporny na zawinięcie przy +-pi:
// 2*pi minus największa luka między posortowanymi phi (łącznie z luką przez pi/-pi).
double CircularSpread(vector<double> phis)
{
    if (phis.size() < 2) return 0.0;
    sort(phis.begin(), phis.end());
    double maxGap = phis.front() + 2.0 * TMath::Pi() - phis.back();   // luka przez zawinięcie
    for (size_t i = 1; i < phis.size(); ++i)
        maxGap = max(maxGap, phis[i] - phis[i - 1]);
    return 2.0 * TMath::Pi() - maxGap;
}

struct TimeHists
{
    TH1D *hTmax, *hHitT;
    TH2D *hPhiVsT, *hEtaVsT, *hEoPVsT;
};

TimeHists MakeTimeHists(const string &p)
{
    TimeHists h;
    h.hTmax    = new TH1D((p + "_tMax").c_str(),  (p + " t_{max} of track hits;t_{max} [ns];Tracks").c_str(), 200, 0, T_MAX);
    h.hHitT    = new TH1D((p + "_hitT").c_str(),  (p + " hit time;t [ns];Hits").c_str(), 200, 0, T_MAX);
    h.hPhiVsT  = new TH2D((p + "_PhiVsT").c_str(), (p + " #phi_{Spread} vs t_{max};t_{max} [ns];#phi_{Spread} [rad]").c_str(), 100, 0, T_MAX, 100, 0, 2.0);
    h.hEtaVsT  = new TH2D((p + "_EtaVsT").c_str(), (p + " #eta_{Spread} vs t_{max};t_{max} [ns];#eta_{Spread}").c_str(),      100, 0, T_MAX, 100, 0, 2.0);
    h.hEoPVsT  = new TH2D((p + "_EoPVsT").c_str(), (p + " E/p vs t_{max};t_{max} [ns];E/p").c_str(),                           100, 0, T_MAX, 100, 0, 2.0);
    return h;
}

void FillTrack(const vector<edm4eic::Cluster> &clusters, double trackP, TimeHists &h)
{
    double sumE = 0, tMax = -1e9;
    double etaMin = 1e9, etaMax = -1e9;
    vector<double> phis;

    for (const auto &cl : clusters)
        for (const auto &hit : cl.getHits())
        {
            double t = hit.getTime();
            auto pos = hit.getPosition();
            TVector3 v(pos.x, pos.y, pos.z);

            h.hHitT->Fill(t);
            tMax = max(tMax, t);
            sumE += hit.getEnergy();
            phis.push_back(v.Phi());
            etaMin = min(etaMin, v.Eta()); etaMax = max(etaMax, v.Eta());
        }

    if (phis.empty() || sumE <= 0) return;

    h.hTmax->Fill(tMax);
    h.hPhiVsT->Fill(tMax, CircularSpread(phis));
    h.hEtaVsT->Fill(tMax, etaMax - etaMin);
    h.hEoPVsT->Fill(tMax, sumE / trackP);
}

// Jeden wykres = jedna strona PDF. 5 wykresów na detektor.
void DrawPages(TimeHists &h, const string &pdf)
{
    TCanvas c("c", "c", 900, 700);

    auto draw2D = [&](TH2D *hh) {
        c.Clear();
        c.SetLogy(0); c.SetLogz(1); c.SetRightMargin(0.13);
        hh->Draw("colz");
        TLine l(CUT_NS, hh->GetYaxis()->GetXmin(), CUT_NS, hh->GetYaxis()->GetXmax());
        l.SetLineColor(kRed); l.SetLineWidth(2);
        l.DrawClone("same");
        c.Print(pdf.c_str());
    };
    auto draw1D = [&](TH1D *hh) {
        c.Clear();
        c.SetLogz(0); c.SetLogy(1); c.SetRightMargin(0.05);
        hh->Draw("hist");
        TLine l(CUT_NS, 0, CUT_NS, hh->GetMaximum());
        l.SetLineColor(kRed); l.SetLineWidth(2);
        l.DrawClone("same");
        c.Print(pdf.c_str());
    };

    draw2D(h.hPhiVsT);
    draw2D(h.hEtaVsT);
    draw2D(h.hEoPVsT);
    draw1D(h.hTmax);
    draw1D(h.hHitT);
}

void TimeCutCheck()
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);

    vector<string> patterns = {
        "/run/media/epic/Data/Background/Muons/Continuous/reco_*.root",
        "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root"};
    vector<string> names = {"Muons", "Pions"};

    vector<string> hcalNames = {"HcalBarrelClusters", "HcalEndcapNClusters", "LFHCALClusters"};
    const long MAX_EVENTS = 10000;

    for (int F = 0; F < 2; ++F)
    {
        auto files = ExpandGlob(patterns[F]);
        if (files.empty()) { cerr << "[ERROR] No files: " << patterns[F] << endl; continue; }

        podio::ROOTReader reader;
        reader.openFiles(files);
        unsigned nEntries = reader.getEntries("events");
        long nProc = (MAX_EVENTS > 0) ? min<long>(MAX_EVENTS, nEntries) : nEntries;

        vector<Det> hcal;
        for (auto &n : hcalNames) hcal.push_back(MakeDet(n));

        TFile out(Form("Plots/TimeCheck_%s.root", names[F].c_str()), "RECREATE");
        TimeHists ecalH = MakeTimeHists(names[F] + "_ECal");
        TimeHists hcalH = MakeTimeHists(names[F] + "_HCal");

        for (long e = 0; e < nProc; ++e)
        {
            if ((e + 1) % 50000 == 0) cout << names[F] << " event " << e + 1 << endl;

            podio::Frame frame(reader.readEntry("events", e));

            if (e == 0) {
                auto avail = frame.getAvailableCollections();
                for (auto &d : hcal)
                    d.available = find(avail.begin(), avail.end(), d.match) != avail.end();
            }

            const auto &parts = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");

            for (const auto &rcp : parts)
            {
                auto mom = rcp.getMomentum();
                TLorentzVector P; P.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
                double trackP = P.P();
                if (trackP <= 1) continue;

                // ECal: klastry z rcp
                vector<edm4eic::Cluster> ecalCl;
                for (const auto &c : rcp.getClusters()) ecalCl.push_back(c);
                FillTrack(ecalCl, trackP, ecalH);

                // HCal: TrackClusterMatch
                if (rcp.getTracks().empty()) continue;
                const auto trk = rcp.getTracks().at(0);
                vector<edm4eic::Cluster> hcalCl;
                for (const auto &d : hcal) {
                    if (!d.available) continue;
                    for (const auto &m : frame.get<edm4eic::TrackClusterMatchCollection>(d.match))
                        if (m.getTrack() == trk) hcalCl.push_back(m.getCluster());
                }
                FillTrack(hcalCl, trackP, hcalH);
            }
        }

        out.Write();

        // Jeden PDF na rodzaj cząstki: 5 stron ECal + 5 stron HCal
        string pdf = Form("Plots/TimeCheck_%s.pdf", names[F].c_str());
        {
            TCanvas dummy("dummy", "dummy", 900, 700);
            dummy.Print((pdf + "[").c_str());
            DrawPages(ecalH, pdf);
            DrawPages(hcalH, pdf);
            dummy.Print((pdf + "]").c_str());
        }

        out.Close();
        cout << "Done " << names[F] << " -> " << pdf << endl;
    }
}