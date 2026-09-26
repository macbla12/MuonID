// PlotAllDiagFeatures.cxx
// Usage: root -l -b -q PlotAllDiagFeatures.cxx+

#include <TFile.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TKey.h>
#include <TClass.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

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

// Determine automatic tail cutoffs (0.5% on each side by default)
std::pair<double, double> GetTailCuts(TH1D* h, double lowQuantile = 0.005, double highQuantile = 0.995) {
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

        // Cut the left tail
        if (!foundMin && fraction >= lowQuantile) {
            int safeBinMin = max(i - 2, 1);
            xMinCut = h->GetXaxis()->GetBinLowEdge(safeBinMin);
            foundMin = true;
        }

        // Cut the right tail
        if (fraction >= highQuantile) {
            int safeBinMax = min(i + 2, nbins);
            xMaxCut = h->GetXaxis()->GetBinUpEdge(safeBinMax);
            break;
        }
    }

    return {xMinCut, xMaxCut};
}

void PlotAllFeatures() {
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(1);              
    gStyle->SetTitleBorderSize(0);       
    gStyle->SetTitleFontSize(0.050);     

    gStyle->SetPadLeftMargin(0.16);
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadRightMargin(0.05);
    gStyle->SetPadTopMargin(0.08);

    string muonPath = "Diag_Muons.root";
    string pionPath = "Diag_Pions.root";

    TFile* f_muon = TFile::Open(muonPath.c_str(), "READ");
    if (!f_muon || f_muon->IsZombie()) {
        muonPath = "Plots/CaloToF/Diag_Muons.root";
        pionPath = "Plots/CaloToF/Diag_Pions.root";
        f_muon = TFile::Open(muonPath.c_str(), "READ");
    }
    TFile* f_pion = TFile::Open(pionPath.c_str(), "READ");

    if (!f_muon || f_muon->IsZombie() || !f_pion || f_pion->IsZombie()) {
        cerr << "[ERROR] Cannot open Diag_Muons.root or Diag_Pions.root!" << endl;
        return;
    }

    const string pdfName = "Plots/CaloToF_Diag_Comparison.pdf";
    bool isFirstPage = true;

    TIter next(f_muon->GetListOfKeys());
    TKey* key = nullptr;

    while ((key = (TKey*)next())) {
        TClass* cl = gROOT->GetClass(key->GetClassName());
        
        // Skip objects other than TH1 and exclude TH2 plots (2D)
        if (!cl || !cl->InheritsFrom("TH1") || cl->InheritsFrom("TH2")) continue;

        string hMuonName = key->GetName();
        string hPionName = hMuonName;

        // Replace '_Muons' with '_Pions' in the pion histogram name
        size_t pos = hPionName.find("_Muons");
        if (pos != string::npos) {
            hPionName.replace(pos, 6, "_Pions");
        }

        TH1D* h_muon_orig = (TH1D*)f_muon->Get(hMuonName.c_str());
        TH1D* h_pion_orig = (TH1D*)f_pion->Get(hPionName.c_str());

        // If the renamed pion histogram is missing from the second file, try the original name
        if (!h_pion_orig) {
            h_pion_orig = (TH1D*)f_pion->Get(hMuonName.c_str());
        }

        if (!h_muon_orig || !h_pion_orig) continue;

        TH1D* h_muon = (TH1D*)h_muon_orig->Clone((hMuonName + "_norm_muon").c_str());
        TH1D* h_pion = (TH1D*)h_pion_orig->Clone((hPionName + "_norm_pion").c_str());

        // 1. Normalizacja do 1
        if (h_muon->Integral() > 0) h_muon->Scale(1.0 / h_muon->Integral());
        if (h_pion->Integral() > 0) h_pion->Scale(1.0 / h_pion->Integral());

        /// 2. Automatically trim the left and right tails
        auto [minMu, maxMu] = GetTailCuts(h_muon, 0.01, 0.96);
        auto [minPi, maxPi] = GetTailCuts(h_pion, 0.01, 0.96);

        double xMinSmart = min(minMu, minPi);
        double xMaxSmart = max(maxMu, maxPi);

        // Set the X-axis range trimmed at both ends
        h_muon->GetXaxis()->SetRangeUser(xMinSmart, xMaxSmart);
        h_pion->GetXaxis()->SetRangeUser(xMinSmart, xMaxSmart);

        // 3. Find the maximum Y value within the new X range
        int binMin = h_muon->GetXaxis()->FindBin(xMinSmart);
        int binMax = h_muon->GetXaxis()->FindBin(xMaxSmart);

        double maxVal = 0.0;
        for (int b = binMin; b <= binMax; ++b) {
            maxVal = max({maxVal, h_muon->GetBinContent(b), h_pion->GetBinContent(b)});
        }

        if (maxVal <= 0) maxVal = 1.0;

        h_muon->SetMinimum(0.0);
        h_muon->SetMaximum(maxVal * 1.30);

        // Create canvas
        TCanvas* c = new TCanvas(("c_" + hMuonName).c_str(), hMuonName.c_str(), 1000, 800);
        c->SetGrid();
        gPad->SetLogy(0);

        // Set a clean title by removing the "_Muons" suffix
        string baseTitle = hMuonName;
        if (pos != string::npos) baseTitle.erase(pos, 6);
        h_muon->SetTitle(baseTitle.c_str());

        // Styling
        h_muon->SetLineColor(kBlue + 1);
        h_muon->SetLineWidth(3);
        h_pion->SetLineColor(kRed + 1);
        h_pion->SetLineWidth(3);

        // Axes
        h_muon->GetXaxis()->SetTitleSize(0.050);
        h_muon->GetYaxis()->SetTitleSize(0.050);
        h_muon->GetXaxis()->SetTitleOffset(1.15);
        h_muon->GetYaxis()->SetTitleOffset(1.30);
        h_muon->GetYaxis()->SetTitle("Normalized Counts");

        h_muon->GetXaxis()->SetLabelSize(0.045);
        h_muon->GetYaxis()->SetLabelSize(0.045);
        
        h_muon->GetXaxis()->SetTitleFont(42);
        h_muon->GetYaxis()->SetTitleFont(42);
        h_muon->GetXaxis()->SetLabelFont(42);
        h_muon->GetYaxis()->SetLabelFont(42);

        // Draw
        h_muon->Draw("HIST");
        h_pion->Draw("HIST SAME");

        // Legend and labels
        TLegend* leg = new TLegend(0.45, 0.75, 0.69, 0.91);
        leg->SetBorderSize(1);
        leg->SetFillColor(kWhite);
        leg->SetTextSize(0.040);
        leg->SetTextFont(42);
        leg->AddEntry(h_muon, "Muon", "l");
        leg->AddEntry(h_pion, "Pion", "l");
        leg->Draw();

        DrawEPICLabel(0.20, 0.82, 0.040);

        // Save to PDF
        if (isFirstPage) {
            c->SaveAs((pdfName + "(").c_str());
            isFirstPage = false;
        } else {
            c->SaveAs(pdfName.c_str());
        }

        delete leg;
        delete h_muon;
        delete h_pion;
        delete c;
    }

    if (!isFirstPage) {
        TCanvas* cClose = new TCanvas("cClose", "", 1000, 800);
        cClose->SaveAs((pdfName + ")").c_str());
        delete cClose;
        cout << "[OK] Generated all TH1 histograms in: " << pdfName << endl;
    }

    f_muon->Close();
    f_pion->Close();
    delete f_muon;
    delete f_pion;
}