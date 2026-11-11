// PlotMuonID.cxx
//
// Ladny plotter wynikow z TestingMacro_MuonID_LowHigh.cxx
// (wejscie: Plots/MuonID_Performance_Full.root, wyjscie: Plots/Final/*.png i *.pdf).
//
// Rysuje:
//   1) Muon efficiency + Background rejection  vs p (0-20), p (zoom 0-2), pT (0-20), pT (zoom 0-2), eta
//   2) Rejection per gatunek (Pion/Electron/Kaon/Proton) + Muon efficiency  (jw.)
//   3) Mis-ID per gatunek w skali log (jw.)
//   4) Response P(muon) dla modelu LowP i HighP - WSZYSTKIE czastki na jednym wykresie.
//        Zrodlo: Plots/Performance_LowP.root oraz Plots/Performance_HighP.root
//        (automatycznie brane sa wszystkie histogramy TH1 z "Response" w nazwie, np. h_Response_*).
//        - osobny wykres dla LowP i dla HighP
//        - panel LowP | HighP obok siebie
//
// Uruchom: root -l -q PlotMuonID.cxx
//      lub: root -l -q 'PlotMuonID.cxx("perf_full.root","low.root","high.root")'

#include <TFile.h>
#include <TKey.h>
#include <TList.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TLegend.h>
#include <TLine.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TColor.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <map>
#include <set>

using namespace std;

// ----------------------------------------------------------------------------
// Ustawienia
// ----------------------------------------------------------------------------
static const double P_SWITCH = 0.7;   // jak w makrze testowym
static const double MUON_CUT = 0.5;   // jak MUON_ID_CUT
static const string OUT_DIR  = "Plots/Final";

static TFile *gIn = nullptr;

// kolory: muon / pion / electron / kaon / proton / caly background
static const int kColMuon = kBlue + 1;
static const int kColAll  = kBlack;
static const int kColSp[4] = {kRed + 1, kOrange + 4, kGreen + 2, kMagenta + 1};
static const int kMkMuon = 20, kMkAll = 21;
static const int kMkSp[4] = {22, 23, 33, 34};
static const char *kSpName[4] = {"Pion", "Electron", "Kaon", "Proton"};

// ----------------------------------------------------------------------------
// Narzedzia
// ----------------------------------------------------------------------------
static void SetStyle()
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    gStyle->SetTextFont(42);
    gStyle->SetLabelFont(42, "XYZ");
    gStyle->SetTitleFont(42, "XYZ");
    gStyle->SetLabelSize(0.045, "XYZ");
    gStyle->SetTitleSize(0.048, "XYZ");
    gStyle->SetTitleOffset(1.05, "X");
    gStyle->SetTitleOffset(1.15, "Y");
    gStyle->SetPadLeftMargin(0.13);
    gStyle->SetPadRightMargin(0.04);
    gStyle->SetPadBottomMargin(0.13);
    gStyle->SetPadTopMargin(0.02);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
    gStyle->SetGridColor(kGray + 1);
    gStyle->SetGridStyle(3);
    gStyle->SetEndErrorSize(3);
}

static TH1F *Get(const string &name)
{
    TH1F *h = dynamic_cast<TH1F*>(gIn->Get(name.c_str()));
    if (!h) { cerr << "[WARN] brak histogramu: " << name << endl; return nullptr; }
    TH1F *c = (TH1F*)h->Clone((name + "_plot").c_str());
    c->SetDirectory(0);
    return c;
}

static void StylePoints(TH1F *h, int col, int mk)
{
    h->SetLineColor(col);
    h->SetMarkerColor(col);
    h->SetMarkerStyle(mk);
    h->SetMarkerSize(1.1);
    h->SetLineWidth(2);
}

void DrawEPICLabel(double x = 0.18, double y = 0.95, double textSize = 0.040) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(textSize);
    latex.SetTextFont(42);
    latex.DrawLatex(x, y, "#bf{ePIC simulation}");
    latex.DrawLatex(x, y - textSize *0.9, "single particle");
}

static void Save(TCanvas &c, const string &name)
{
    c.SaveAs((OUT_DIR + "/" + name + ".pdf").c_str());
    cout << "  zapisano: " << OUT_DIR << "/" << name << ".{png,pdf}" << endl;
}

// ----------------------------------------------------------------------------
// Wykresy efektywnosci / rejection / mis-ID
// ----------------------------------------------------------------------------
struct Curve { TH1F *h; string label; int col; int mk; };

static void DrawCurves(const vector<Curve> &cv, const string &xTitle, const string &yTitle,
                       const string &outName, bool logY, bool showSwitch, const string &note)
{
    vector<Curve> v;
    for (auto &c : cv) if (c.h) v.push_back(c);
    if (v.empty()) return;

    TCanvas c(("c_" + outName).c_str(), outName.c_str(), 950, 700);
    c.SetGrid();
    if (logY) c.SetLogy();

    double ymin = logY ? 1e-3 : 0.6;
    double ymax = logY ? 5.0   : 1.2;

    bool first = true;
    bool zoom = (outName.find("zoom") != string::npos);
    for (auto &cu : v) {
        StylePoints(cu.h, cu.col, cu.mk);
        if(zoom) cu.h->Rebin(2);
        if(zoom) cu.h->Scale(0.5);
        cu.h->SetMinimum(ymin);
        cu.h->SetMaximum(ymax);
        cu.h->GetXaxis()->SetTitle(xTitle.c_str());
        cu.h->GetYaxis()->SetTitle(yTitle.c_str());
        cu.h->Draw(first ? "E1 P" : "E1 P SAME");
        first = false;
    }

    // linia przelaczenia modeli
    if (showSwitch) {
        double xlo = v[0].h->GetXaxis()->GetXmin(), xhi = v[0].h->GetXaxis()->GetXmax();
        if (P_SWITCH > xlo && P_SWITCH < xhi) {
            TLine *l = new TLine(P_SWITCH, ymin, P_SWITCH, logY ? 1.0 : 1.0);
            l->SetLineStyle(2);
            l->SetLineColor(kGray + 2);
            l->SetLineWidth(2);
            l->Draw();
            
        }
    }

    // linia y = 1
    if (!logY) {
        double xlo = v[0].h->GetXaxis()->GetXmin(), xhi = v[0].h->GetXaxis()->GetXmax();
        TLine *l1 = new TLine(xlo, 1.0, xhi, 1.0);
        l1->SetLineStyle(7);
        l1->SetLineColor(kGray + 1);
        l1->Draw();
    }

    TLegend *leg = new TLegend(0.70, 0.75, 0.93, 0.95);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.037);
    for (auto &cu : v) leg->AddEntry(cu.h, cu.label.c_str(), "lep");
    leg->Draw();

    DrawEPICLabel(0.17, 0.92, 0.040);
    if (!note.empty()) {
        TLatex *t = new TLatex(0.17, 0.85, note.c_str());
        t->SetNDC();
        t->SetTextSize(0.035);
        t->Draw();
    }
    c.RedrawAxis();
    Save(c, outName);
}

static void PlotPerformance()
{
    cout << "== Efficiency / rejection / mis-ID ==" << endl;

    const char *vName[5]  = {"Pt", "P", "Eta", "PtZoom", "PZoom"};
    const char *xTit[5]   = {"p_{T} [GeV/c]", "p [GeV/c]", "#eta", "p_{T} [GeV/c]", "p [GeV/c]"};
    const char *tag[5]    = {"Pt_0-20", "P_0-20", "Eta", "Pt_0-2_zoom", "P_0-2_zoom"};
    const bool  isP[5]    = {false, true, false, false, true};
    const char *note[5]   = {"p_{T} 0-20 GeV/c", "p 0-20 GeV/c", "", "p_{T} 0-2 GeV/c", "p 0-2 GeV/c"};

    for (int v = 0; v < 5; ++v)
    {
        string vn = vName[v];

        TH1F *eff = Get("h_Muon_Efficiency_vs_" + vn);
        TH1F *rej = Get("h_Background_Rejection_vs_" + vn);

        // 1) muon eff + caly background rejection
        DrawCurves({{eff, "Muon efficiency", kColMuon, kMkMuon},
                    {rej, "Background rejection", kColAll, kMkAll}},
                   xTit[v], "Efficiency / Rejection",
                   string("Performance_") + tag[v], false, isP[v], note[v]);

        // 2) rejection per gatunek
        TH1F *effB = Get("h_Muon_Efficiency_vs_" + vn);
        vector<Curve> cs = {{effB, "Muon - eff", kColMuon, kMkMuon}};
        for (int i = 0; i < 4; ++i)
            cs.push_back({Get(string("h_Bkg") + kSpName[i] + "_Rejection_vs_" + vn),
                          string(kSpName[i]) + "- rej eff", kColSp[i], kMkSp[i]});
        DrawCurves(cs, xTit[v], "Efficiency / Rejection Efficiency",
                   string("PerSpecies_Rejection_") + tag[v], false, isP[v], note[v]);

    }
}

// ----------------------------------------------------------------------------
// Response P(muon) - z osobnych plikow Performance_LowP.root / Performance_HighP.root
// ----------------------------------------------------------------------------
struct RespCurve { TH1F *h; string label; int col; bool filled; };

// Etykieta z nazwy histogramu - rozpoznaje czastke po slowie kluczowym w nazwie
// (np. "Pion_Sep", "Pion_Response", "BkgPion_Response" -> "Pion").
// W makrach testowych h_Response_Pion to w rzeczywistosci CALY background (nazwa historyczna).
// Jesli chcesz inne podpisy (np. TLatex: "#pi^{#pm}"), zmien teksty ponizej.
// UWAGA: sortowanie w LoadResponses opiera sie na etykietach "Muon" i "All background".
static string RespLabel(const string &name)
{
    if (name == "h_Response_Pion") return "All background";

    string low = name;
    transform(low.begin(), low.end(), low.begin(), [](unsigned char ch) { return (char)tolower(ch); });

    if (low.find("background") != string::npos || low.find("allbkg") != string::npos) return "All background";
    if (low.find("muon")     != string::npos) return "Muon";
    if (low.find("pion")     != string::npos) return "Pion";
    if (low.find("electron") != string::npos) return "Electron";
    if (low.find("kaon")     != string::npos) return "Kaon";
    if (low.find("proton")   != string::npos) return "Proton";

    // nieznana nazwa - zwroc w miare czysta wersje
    string s = name;
    if (s.rfind("h_Response_", 0) == 0) s = s.substr(11);
    else if (s.rfind("h_", 0) == 0)     s = s.substr(2);
    size_t p = s.find("_Response");
    if (p != string::npos) s.erase(p, 9);
    if (s.rfind("Bkg", 0) == 0) s = s.substr(3);
    return s;
}

// Wczytuje wszystkie histogramy z "Response" w nazwie, normalizuje do jedynki.
static vector<RespCurve> LoadResponses(TFile *f)
{
    static const int palette[] = {kRed + 1, kOrange + 4, kGreen + 2, kMagenta + 1};
    vector<RespCurve> out;
    if (!f) return out;

    set<string> seen;
    TIter next(f->GetListOfKeys());
    while (TKey *k = (TKey*)next()) {
        string name = k->GetName();
        if (name.find("Response") == string::npos) continue;
        if (!seen.insert(name).second) continue;          // kolejne cykle tego samego klucza
        TH1F *h = dynamic_cast<TH1F*>(k->ReadObj());
        if (!h) continue;
        h->SetDirectory(0);
        if (h->Integral() <= 0) { cerr << "[WARN] pusty response: " << name << endl; continue; }

        RespCurve rc;
        rc.h = h;
        rc.label = RespLabel(name);
        rc.filled = false;
        rc.col = kBlack;
        out.push_back(rc);
        cout << "    znaleziono: " << name << "  ->  " << rc.label << endl;
    }

    // kolejnosc: Muon, All background, reszta
    auto rank = [](const RespCurve &r) {
        if (r.label.rfind("Muon", 0) == 0) return 0;
        if (r.label.rfind("All background", 0) == 0) return 1;
        return 2;
    };
    stable_sort(out.begin(), out.end(), [&](const RespCurve &a, const RespCurve &b) { return rank(a) < rank(b); });

    int ip = 0;
    for (auto &r : out) {
        int rk = rank(r);
        if (rk == 0)      { r.col = kColMuon; r.filled = true; }
        else if (rk == 1) { r.col = kColAll; }
        else              { r.col = palette[ip++ % 7]; }
        r.h->Rebin(2);
        r.h->Scale(1.0 / r.h->Integral());
    }
    return out;
}

static void DrawResponsePad(const vector<RespCurve> &rc, const string &desc)
{
    gPad->SetGrid();
    gPad->SetLogy();

    if (rc.empty()) {
        TLatex t; t.SetNDC(); t.SetTextSize(0.05);
        t.DrawLatex(0.25, 0.5, "No Response histograms found");
        return;
    }

    double ymax = 0.0;
    for (auto &r : rc) ymax = max(ymax, r.h->GetMaximum());

    bool first = true;
    for (auto &r : rc) {
        r.h->SetLineColor(r.col);
        r.h->SetLineWidth(2);
        r.h->SetMarkerSize(0);
        if (r.filled) { r.h->SetFillColorAlpha(r.col, 0.20); r.h->SetFillStyle(1001); }
        else          { r.h->SetFillStyle(0); }
        r.h->SetMinimum(1e-4);
        r.h->SetMaximum(ymax * 10);
        r.h->GetXaxis()->SetTitle("P(muon)");
        r.h->GetYaxis()->SetTitle("Fraction of tracks / bin");
        r.h->GetXaxis()->SetRangeUser(0, 1);
        r.h->Draw(first ? "HIST" : "HIST SAME");
        first = false;
    }

    TLine *cut = new TLine(MUON_CUT, 1e-4, MUON_CUT, ymax * 10);
    cut->SetLineStyle(2);
    cut->SetLineColor(kGray + 2);
    cut->SetLineWidth(2);
    cut->Draw();


    double legH = 0.048 * rc.size();
    TLegend* leg = new TLegend(0.70, 0.75, 0.93, 0.95);
        leg->SetBorderSize(1);
        leg->SetFillColor(kWhite);
        leg->SetTextSize(0.035);
        leg->SetTextFont(42);
    for (auto &r : rc) leg->AddEntry(r.h, r.label.c_str(), r.filled ? "f" : "l");
    leg->Draw();

    DrawEPICLabel(0.15, 0.92, 0.040);
    TLatex *t = new TLatex(0.15, 0.85, desc.c_str());
    t->SetNDC();
    t->SetTextSize(0.035);
    t->Draw();
    gPad->RedrawAxis();
}

static void PlotResponse(const char *lowFile, const char *highFile)
{
    cout << "== Response P(muon) ==" << endl;

    TFile *fLow  = TFile::Open(lowFile,  "READ");
    TFile *fHigh = TFile::Open(highFile, "READ");
    if (!fLow  || fLow->IsZombie())  { cerr << "[ERROR] nie moge otworzyc: " << lowFile  << endl; fLow  = nullptr; }
    if (!fHigh || fHigh->IsZombie()) { cerr << "[ERROR] nie moge otworzyc: " << highFile << endl; fHigh = nullptr; }

    cout << "  " << lowFile << ":" << endl;
    vector<RespCurve> low  = LoadResponses(fLow);
    cout << "  " << highFile << ":" << endl;
    vector<RespCurve> high = LoadResponses(fHigh);

    const string descLow  = "Low-P model (Calo + ToF)";
    const string descHigh = "High-P model (Calo)";

    { TCanvas c("c_resp_low", "Response Low-P", 1000, 800);
      DrawResponsePad(low, descLow);
      Save(c, "Response_LowP_AllParticles"); }

    { TCanvas c("c_resp_high", "Response High-P", 1000, 800);
      DrawResponsePad(high, descHigh);
      Save(c, "Response_HighP_AllParticles"); }


    if (fLow)  fLow->Close();
    if (fHigh) fHigh->Close();
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
void PlotResults(const char *perfFile = "Plots/MuonID_Performance_Full.root",
                const char *lowFile  = "Plots/Performance_LowP.root",
                const char *highFile = "Plots/Performance_HighP.root")
{
    SetStyle();
    gSystem->mkdir(OUT_DIR.c_str(), kTRUE);
    TH1::AddDirectory(kFALSE);

    gIn = TFile::Open(perfFile, "READ");
    if (!gIn || gIn->IsZombie()) {
        cerr << "[ERROR] nie moge otworzyc: " << perfFile << endl;
    } else {
        PlotPerformance();
        gIn->Close();
    }

    PlotResponse(lowFile, highFile);

    cout << "Gotowe. Wykresy w " << OUT_DIR << "/" << endl;
}