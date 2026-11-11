// plotter.cxx - nakladka wynikow ToFMass.cxx (Grape / SingleMuons / SinglePions)
// Uruchom: root -l -q plotter.cxx++
//
// Czyta:    Plots/ToFMass_<name>.root
// Zapisuje: Plots/Overlay_all.pdf   (wielostronicowy PDF, kazdy wykres = osobny Canvas = osobna strona)
//
// Kolejnosc stron: 1) TProfile m(p) (glowny)  2) TProfile beta(p)
//                  3-10) rozklady masy (all, 3 przedzialy p, barrel, endcap, +, -)

#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TF1.h>
#include <TLine.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TStyle.h>
#include <TROOT.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

// Helper for drawing the ePIC label
void DrawEPICLabel(double x = 0.18, double y = 0.84, double textSize = 0.040) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(textSize);
    latex.SetTextFont(42);
    latex.DrawLatex(x, y, "#bf{ePIC simulation}");
    latex.DrawLatex(x, y - textSize * 1.25, "single particle");
}

struct Sample {
    string name;    // nazwa uzyta w ToFMass.cxx (sufiks histogramow i pliku)
    string label;   // opis w legendzie
    int    color;
    int    marker;
    TFile *file = nullptr;
};

static const double M_MU = 0.10566, M_PIC = 0.13957; // [GeV]

// ---------------------------------------------------------------------------
template <class T>
T *GetH(Sample &s, const string &base, const string &tag)
{
    auto *h = dynamic_cast<T *>(s.file->Get(Form("%s_%s", base.c_str(), s.name.c_str())));
    if (!h) { cerr << "[WARN] brak " << base << "_" << s.name << endl; return nullptr; }
    auto *c = (T *)h->Clone(Form("%s_%s_clone", base.c_str(), tag.c_str()));
    c->SetDirectory(0);
    return c;
}

void Style(TH1 *h, const Sample &s)
{
    h->SetLineColor(s.color);
    h->SetMarkerColor(s.color);
    h->SetMarkerStyle(s.marker);
    h->SetMarkerSize(1.0);
    h->SetLineWidth(2);
}

TCanvas *NewCanvas(const char *name, int w = 1000, int h = 750)
{
    auto *c = new TCanvas(name, name, w, h);
    c->SetLeftMargin(0.13);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.13);
    c->SetTopMargin(0.09);
    //c->SetGrid();
    return c;
}

// ---------------------------------------------------------------------------
// 1D: rozklad masy (znormalizowany do 1), rysowany w biezacym padzie
void DrawMass(vector<Sample> &S, const string &base, const string &title,
              const string &tag, bool withLegend)
{
    auto *leg = new TLegend(0.7, 0.7, 0.95, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.045);

    double ymax = 0;
    vector<TH1D *> hs;
    for (auto &s : S) {
        auto *h = GetH<TH1D>(s, base, s.name + "_" + tag);
        if (!h) continue;
        h->Rebin(4);
        if (h->Integral() > 0) h->Scale(1.0 / h->Integral());
        Style(h, s);
        ymax = max(ymax, h->GetMaximum());
        hs.push_back(h);
        leg->AddEntry(h, s.label.c_str(), "l");

    }
    if (hs.empty()) return;

    bool first = true;
    for (auto *h : hs) {
        if (first) {
            h->SetTitle(Form("%s;m [GeV];Fraction of tracks", title.c_str()));
            h->SetMaximum(ymax * 1.3);
            h->GetXaxis()->SetTitleSize(0.05);
            h->GetYaxis()->SetTitleSize(0.05);
            h->GetYaxis()->SetTitleOffset(1.2);
            h->Draw("HIST");
            first = false;
        } else h->Draw("HIST SAME");
    }

    for (double m : {M_MU, M_PIC}) {
        auto *l = new TLine(m, 0, m, ymax * 1.3);
        l->SetLineStyle(2); l->SetLineColor(kGray + 2);
        l->Draw();
    }
    if (withLegend) leg->Draw();
    DrawEPICLabel(0.20, 0.82, 0.040);
}

// ---------------------------------------------------------------------------
// GLOWNY: TProfile m(p), bledy = odchylenie standardowe masy w binie p
// ---------------------------------------------------------------------------
void DrawMassProfile(vector<Sample> &S)
{
    auto *leg = new TLegend(0.74, 0.72, 0.95, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);

    vector<TProfile *> ps;
    for (auto &s : S) {
        auto *h2 = GetH<TH2D>(s, "MassVsP", s.name + "_mp");
        if (!h2) continue;
        auto *p = h2->ProfileX(Form("MassProf_%s", s.name.c_str()), 1, -1, "s"); // "s": blad = RMS
        p->SetDirectory(0);
        Style(p, s);
        ps.push_back(p);
        leg->AddEntry(p, s.label.c_str(), "lep");
    }
    if (ps.empty()) return;

    double xmin = ps[0]->GetXaxis()->GetXmin();
    double xmax = 1.0; 

    bool first = true;
    for (auto *p : ps) {
        p->Rebin(2); 
        p->GetXaxis()->SetRangeUser(xmin, xmax); 
        

        if (first) {
            p->SetTitle("Mass vs p (error bars = #sigma in p bin);p [GeV];<m> #pm #sigma_{m} [GeV]");
            p->GetYaxis()->SetRangeUser(0.05, 0.25);
            p->GetXaxis()->SetTitleSize(0.05);
            p->GetYaxis()->SetTitleSize(0.05);
            p->GetYaxis()->SetTitleOffset(1.1);
            p->Draw("E1");
            first = false;
        } else p->Draw("E1 SAME");
    }

    for (double m : {M_MU, M_PIC}) {
        auto *l = new TLine(xmin, m, xmax, m);
        l->SetLineStyle(2); l->SetLineColor(kGray + 2);
        l->Draw();
    }
    auto *line = new TLine(0.7,0.05, 0.7, 0.25);
    line->SetLineColor(kRed);
    line->Draw();

    auto *tx = new TLatex();
    tx->SetTextSize(0.035); tx->SetTextColor(kGray + 2);
    tx->DrawLatex(xmax * 0.10, M_MU - 0.01, "m_{#mu}");
    tx->DrawLatex(xmax * 0.10, M_PIC + 0.005, "m_{#pi}");

    leg->Draw();
    DrawEPICLabel(0.20, 0.82, 0.040);
}

// ---------------------------------------------------------------------------
void DrawBetaProfile(vector<Sample> &S)
{
    auto *leg = new TLegend(0.65, 0.2, 0.95, 0.48);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);

    vector<TProfile *> ps;
    for (auto &s : S) {
        auto *h2 = GetH<TH2D>(s, "BetaVsP", s.name + "_bp");
        if (!h2) continue;
        auto *p = h2->ProfileX(Form("BetaProf_%s", s.name.c_str()), 1, -1, "s");
        p->SetDirectory(0);
        Style(p, s);
        ps.push_back(p);
        leg->AddEntry(p, s.label.c_str(), "lep");
    }
    if (ps.empty()) return;

    bool first = true;
    for (auto *p : ps) {
        p->Rebin(2);

        if (first) {
            p->SetTitle("#beta vs p (error bars = #sigma in p bin);p [GeV];<#beta> #pm #sigma_{#beta}");
            p->GetYaxis()->SetRangeUser(0.85, 1.02);
            p->GetXaxis()->SetRangeUser(0.00, 1.00);

            p->GetXaxis()->SetTitleSize(0.05);
            p->GetYaxis()->SetTitleSize(0.05);
            p->GetYaxis()->SetTitleOffset(1.1);
            p->Draw("E1");
            first = false;
        } else p->Draw("E1 SAME");
    }

    double xmax = ps[0]->GetXaxis()->GetXmax();
    auto *fMu = new TF1("fMu", "x/sqrt(x*x+[0]*[0])", 0.01, xmax);
    fMu->SetParameter(0, M_MU);  fMu->SetLineColor(kGray + 2); fMu->SetLineStyle(2);
    auto *fPi = new TF1("fPi", "x/sqrt(x*x+[0]*[0])", 0.01, xmax);
    fPi->SetParameter(0, M_PIC); fPi->SetLineColor(kGray + 2); fPi->SetLineStyle(3);
    fMu->Draw("SAME");
    fPi->Draw("SAME");
    leg->AddEntry(fMu, "#beta(p), m = m_{#mu}", "l");
    leg->AddEntry(fPi, "#beta(p), m = m_{#pi}", "l");
    leg->Draw();
    DrawEPICLabel(0.20, 0.82, 0.040);
}

// ---------------------------------------------------------------------------
void Plotter()
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetTitleFontSize(0.06);

    vector<Sample> S = {
        {"SinglePions", "Single #pi^{+}", kGreen + 2, 21},
        {"SingleMuons", "Single #mu", kBlue + 1,  22},
    };

    for (auto &s : S) {
        s.file = TFile::Open(Form("Plots/ToFMass_%s.root", s.name.c_str()), "READ");
        if (!s.file || s.file->IsZombie()) {
            cerr << "[ERR] nie mozna otworzyc Plots/ToFMass_" << s.name << ".root" << endl;
            s.file = nullptr;
        }
    }
    vector<Sample> ok;
    for (auto &s : S) if (s.file) ok.push_back(s);
    if (ok.empty()) { cerr << "[ERR] brak plikow" << endl; return; }

    const char *pdf = "Plots/Overlay_all.pdf";

    // --- strona 1: glowny wykres, TProfile m(p) ---
    auto *c1 = NewCanvas("c_massprof");
    c1->Print(Form("%s[", pdf));          // otwarcie wielostronicowego PDF
    DrawMassProfile(ok);
    c1->Print(pdf);

    // --- strona 2: beta(p) ---
    auto *c2 = NewCanvas("c_betaprof");
    DrawBetaProfile(ok);
    c2->Print(pdf);

    // --- strony 3-10: rozklady masy ---
    struct Item { const char *base, *title, *tag; };
    vector<Item> items = {
        {"Mass",        "mass (all ToF)",         "all"},
        {"Mass_p0",   "mass, p < 0.3 GeV",      "p0"},
        {"Mass_p1",   "mass, 0.3 < p < 0.5 GeV",  "p1"},
        {"Mass_p2",   "mass, 0.5 < p < 0.7 GeV",        "p2"},
        {"Mass_p3",   "mass, 0.7 < p < 0.9 GeV",        "p3"},
        {"Mass_p4",   "mass, 0.9 < p < 1.0 GeV",        "p4"},
        {"Mass_p5",   "mass, p > 1.1 GeV",        "p5"},


    };
    for (auto &it : items) {
        auto *c = NewCanvas(Form("c_%s", it.tag));
        DrawMass(ok, it.base, it.title, it.tag, /*withLegend=*/ true);
        c->Print(pdf);
    }

    c1->Print(Form("%s]", pdf));          // zamkniecie PDF
    cout << "Gotowe: " << pdf << endl;
}