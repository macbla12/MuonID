// PlotCombinedFeatures.cxx
// Usage: root -l -b -q PlotCombinedFeatures.cxx+
//
// Czyta:  Plots/CombinedFeatures_LowP.root   i   Plots/CombinedFeatures_HighP.root
//         (katalogi: Muons, Pions, Electrons, Kaons, Protons; histogramy Comb_*)
// Zapisuje (osobno dla low i high P):
//   Plots/LowP/Combined/<hist>.pdf   + Plots/Combined_5Particle_Comparison_LowP.pdf
//   Plots/HighP/Combined/<hist>.pdf  + Plots/Combined_5Particle_Comparison_HighP.pdf

#include <TFile.h>
#include <TDirectory.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TKey.h>
#include <TClass.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TString.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

// Ustaw na true, jesli chcesz tez rysowac ECal_h_* i HCal_h_* (domyslnie tylko Comb_*)
static const bool PLOT_ONLY_COMB = true;

struct ParticleInfo {
    string name;      // etykieta w legendzie
    string dirName;   // katalog w pliku ROOT
    Color_t color;
};

void DrawEPICLabel(const string &momentumLabel, double x = 0.18, double y = 0.95, double textSize = 0.040) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(textSize);
    latex.SetTextFont(42);
    latex.DrawLatex(x, y, "#bf{ePIC simulation}");
    latex.DrawLatex(x, y - textSize * 1.25, "single particle");
    latex.DrawLatex(x, y - textSize * 2.5, momentumLabel.c_str());
}

// Automatyczne odciecie ogonow
std::pair<double, double> GetTailCuts(TH1* h, double lowQuantile = 0.005, double highQuantile = 0.995) {
    if (!h || h->Integral() <= 0) {
        return {h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax()};
    }

    double totalIntegral = h->Integral();
    double runningSum = 0.0;
    int nbins = h->GetNbinsX();

    double xMinCut = h->GetXaxis()->GetXmin();
    double xMaxCut = h->GetXaxis()->GetXmax();
    bool foundMin = false;

    for (int i = 1; i <= nbins; ++i) {
        runningSum += h->GetBinContent(i);
        double fraction = runningSum / totalIntegral;

        if (!foundMin && fraction >= lowQuantile) {
            int safeBinMin = max(i - 2, 1);
            xMinCut = h->GetXaxis()->GetBinLowEdge(safeBinMin);
            foundMin = true;
        }
        if (fraction >= highQuantile) {
            int safeBinMax = min(i + 2, nbins);
            xMaxCut = h->GetXaxis()->GetBinUpEdge(safeBinMax);
            break;
        }
    }
    return {xMinCut, xMaxCut};
}

void PlotMode(bool highP) {
    const string tag       = highP ? "HighP" : "LowP";
    const string momLabel  = highP ? "p > 0.7 GeV" : "p #leq 0.7 GeV";
    const string inFile    = "Plots/CombinedFeatures_" + tag + ".root";
    const string outDir    = "Plots/" + tag + "/Combined";
    const string pdfName   = "Plots/Combined_5Particle_Comparison_" + tag + ".pdf";

    gSystem->mkdir(outDir.c_str(), kTRUE);

    vector<ParticleInfo> particles = {
        {"Electron", "Electrons", kOrange + 4},
        {"Muon",     "Muons",     kBlue + 1},
        {"Pion",     "Pions",     kRed + 1},
        {"Kaon",     "Kaons",     kGreen + 2},
        {"Proton",   "Protons",   kMagenta + 1}
    };

    TFile* file = TFile::Open(inFile.c_str(), "READ");
    if (!file || file->IsZombie()) {
        cerr << "[ERROR] Cannot open file: " << inFile << endl;
        return;
    }

    // Katalog referencyjny: pierwszy istniejacy
    TDirectory* refDir = nullptr;
    for (auto& p : particles) {
        refDir = file->GetDirectory(p.dirName.c_str());
        if (refDir) break;
    }
    if (!refDir) {
        cerr << "[ERROR] No particle directories in " << inFile << endl;
        file->Close();
        return;
    }

    bool isFirstPage = true;

    TIter next(refDir->GetListOfKeys());
    TKey* key = nullptr;

    while ((key = (TKey*)next())) {
        TClass* cl = gROOT->GetClass(key->GetClassName());
        if (!cl || !cl->InheritsFrom("TH1") || cl->InheritsFrom("TH2")) continue;

        string hName = key->GetName();
        TString name = hName;
        if (PLOT_ONLY_COMB && !name.BeginsWith("Comb_")) continue;

        const bool isDiscrete = name.EndsWith("_HasECal") || name.EndsWith("_HasHCal") || name.EndsWith("_Category");

        vector<pair<ParticleInfo, TH1*>> validHistos;
        double xMinSmart = 1e9;
        double xMaxSmart = -1e9;

        for (auto& p : particles) {
            TDirectory* d = file->GetDirectory(p.dirName.c_str());
            if (!d) continue;

            TH1* h_orig = dynamic_cast<TH1*>(d->Get(hName.c_str()));
            if (!h_orig) continue;

            TH1* h_clone = (TH1*)h_orig->Clone(("h_" + p.name + "_" + hName).c_str());
            h_clone->SetDirectory(nullptr);

            // Rebin(2) dla trybu LowP (z wylaczeniem histogramow dyskretnych)
            if (!highP && !isDiscrete) {
                h_clone->Rebin(3);
            }

            if (h_clone->Integral() <= 0) { delete h_clone; continue; }
            h_clone->Scale(1.0 / h_clone->Integral());

            if (isDiscrete) {
                xMinSmart = min(xMinSmart, h_clone->GetXaxis()->GetXmin());
                xMaxSmart = max(xMaxSmart, h_clone->GetXaxis()->GetXmax());
            } else {
                auto [minC, maxC] = GetTailCuts(h_clone, 0.01, 0.96);
                xMinSmart = min(xMinSmart, minC);
                xMaxSmart = max(xMaxSmart, maxC);
            }

            validHistos.push_back({p, h_clone});
        }

        if (validHistos.empty()) continue;

        if (xMinSmart >= xMaxSmart) {
            xMinSmart = validHistos[0].second->GetXaxis()->GetXmin();
            xMaxSmart = validHistos[0].second->GetXaxis()->GetXmax();
        }

        double maxVal = 0.0;
        for (auto& item : validHistos) {
            TH1* h = item.second;
            h->GetXaxis()->SetRangeUser(xMinSmart, xMaxSmart);
            int binMin = h->GetXaxis()->FindBin(xMinSmart);
            int binMax = h->GetXaxis()->FindBin(xMaxSmart);
            for (int b = binMin; b <= binMax; ++b) maxVal = max(maxVal, h->GetBinContent(b));
        }
        if (maxVal <= 0) maxVal = 1.0;

        TCanvas* c = new TCanvas(("c_" + hName).c_str(), hName.c_str(), 1000, 800);
        c->SetGrid();

        // Skala log dla szerokich rozkladow stosunkow
        if (name.EndsWith("_REp") || name.EndsWith("_REhit") || name.EndsWith("_RDRw")) {
            maxVal *= 15.0;
            gPad->SetLogy(1);
        } else {
            gPad->SetLogy(0);
        }

        TH1* h_first = validHistos[0].second;
        h_first->SetMinimum(gPad->GetLogy() ? 0.0001 : 0.0);
        h_first->SetMaximum(maxVal * 1.30);
        h_first->SetTitle("");

        h_first->GetXaxis()->SetTitleSize(0.050);
        h_first->GetYaxis()->SetTitleSize(0.050);
        h_first->GetXaxis()->SetTitleOffset(1.15);
        h_first->GetYaxis()->SetTitleOffset(1.30);
        h_first->GetYaxis()->SetTitle("Normalized Counts");
        h_first->GetXaxis()->SetLabelSize(0.045);
        h_first->GetYaxis()->SetLabelSize(0.045);
        h_first->GetXaxis()->SetTitleFont(42);
        h_first->GetYaxis()->SetTitleFont(42);
        h_first->GetXaxis()->SetLabelFont(42);
        h_first->GetYaxis()->SetLabelFont(42);

        TLegend* leg = new TLegend(0.65, 0.70, 0.88, 0.90);
        leg->SetBorderSize(1);
        leg->SetFillColor(kWhite);
        leg->SetTextSize(0.035);
        leg->SetTextFont(42);

        bool firstDraw = true;
        for (auto& item : validHistos) {
            TH1* h = item.second;
            h->SetLineColor(item.first.color);
            h->SetLineWidth(3);
            if (firstDraw) { h->Draw("HIST"); firstDraw = false; }
            else           { h->Draw("HIST SAME"); }
            leg->AddEntry(h, item.first.name.c_str(), "l");
        }

        leg->Draw();
        DrawEPICLabel(momLabel, 0.15, 0.92, 0.040);

        c->SaveAs((outDir + "/" + hName + ".pdf").c_str());
        if (isFirstPage) {
            c->SaveAs((pdfName + "(").c_str());
            isFirstPage = false;
        } else {
            c->SaveAs(pdfName.c_str());
        }

        delete leg;
        for (auto& item : validHistos) delete item.second;
        delete c;
    }

    if (!isFirstPage) {
        TCanvas* cClose = new TCanvas("cClose", "", 1000, 800);
        cClose->SaveAs((pdfName + ")").c_str());
        delete cClose;
        cout << "[OK] " << tag << ": generated plots in " << pdfName << endl;
    }

    file->Close();
    delete file;
}

void PlotCombinedFeatures() {
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(1);
    gStyle->SetTitleBorderSize(0);
    gStyle->SetTitleFontSize(0.050);

    gStyle->SetPadLeftMargin(0.12);
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadRightMargin(0.02);
    gStyle->SetPadTopMargin(0.02);

    PlotMode(false);   // low P
    PlotMode(true);    // high P
}