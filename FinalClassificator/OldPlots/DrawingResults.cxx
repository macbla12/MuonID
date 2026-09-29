#include <TFile.h>
#include <TH1F.h>
#include <TEfficiency.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TString.h>
#include <TGraphAsymmErrors.h>
#include <algorithm>

// Helper: find the minimum value (including the downward error)
// among TEfficiency and TH1F points, skipping empty/inactive bins.
double FindMinValue(TEfficiency *eff, TH1F *h) {
    double minVal = 1.0;
    bool foundAny = false;

    if (eff && eff->GetTotalHistogram()) {
        int nBins = eff->GetTotalHistogram()->GetNbinsX();
        for (int b = 1; b <= nBins; ++b) {
            if (eff->GetTotalHistogram()->GetBinContent(b) <= 0) continue; // Skip empty bins
            double val    = eff->GetEfficiency(b);
            double errLow = eff->GetEfficiencyErrorLow(b);
            double candidate = val - errLow;
            if (!foundAny || candidate < minVal) {
                minVal = candidate;
                foundAny = true;
            }
        }
    }

    if (h) {
        int nBinsH = h->GetNbinsX();
        for (int b = 1; b <= nBinsH; ++b) {
            if (h->GetBinContent(b) == 0 && h->GetBinError(b) == 0) continue; // Skip empty bins
            double val = h->GetBinContent(b);
            double err = h->GetBinError(b);
            double candidate = val - err;
            if (!foundAny || candidate < minVal) {
                minVal = candidate;
                foundAny = true;
            }
        }
    }

    if (!foundAny) minVal = 0.0;
    if (minVal < 0.0) minVal = 0.0;

    return minVal;
}

void DrawingResults() {
    // 1. Open ROOT files
    TFile *fGlasgow = TFile::Open("../CalorimetryClusters/Plots/FinalCalID.root", "READ");
    TFile *fCurrent = TFile::Open("TestingMuonID_Performance_Combined.root", "READ");

    if (!fGlasgow || fGlasgow->IsZombie() || !fCurrent || fCurrent->IsZombie()) {
        printf("Error opening ROOT files!\n");
        return;
    }

    gStyle->SetOptStat(0);

    // Configuration for six plots
    struct PlotConfig {
        const char* nameGlasgow;
        const char* nameCurrent;
        const char* title;
        const char* xAxis;
        const char* fileName;
    };

    PlotConfig plots[6] = {
        {"Eff_XGBoost_vs_P_Muon",   "h_Muon_Efficiency_vs_P",   "Muon Identification Efficiency vs p",   "Momentum [GeV/c]", "Muon_Efficiency_vs_P.png"},
        {"Eff_XGBoost_vs_Pt_Muon",  "h_Muon_Efficiency_vs_Pt",  "Muon Identification Efficiency vs p_{T}","p_{T} [GeV/c]",     "Muon_Efficiency_vs_Pt.png"},
        {"Eff_XGBoost_vs_Eta_Muon", "h_Muon_Efficiency_vs_Eta", "Muon Identification Efficiency vs #eta", "#eta",             "Muon_Efficiency_vs_Eta.png"},
        {"Rej_XGBoost_vs_P_Pion",   "h_Pion_Rejection_vs_P",    "Pion Rejection Efficiency vs p",              "Momentum [GeV/c]", "Pion_Rejection_vs_P.png"},
        {"Rej_XGBoost_vs_Pt_Pion",  "h_Pion_Rejection_vs_Pt",   "Pion Rejection Efficiency vs p_{T}",         "p_{T} [GeV/c]",     "Pion_Rejection_vs_Pt.png"},
        {"Rej_XGBoost_vs_Eta_Pion", "h_Pion_Rejection_vs_Eta",  "Pion Rejection Efficiency vs #eta",            "#eta",             "Pion_Rejection_vs_Eta.png"}
    };

    // Create and save each plot
    for (int i = 0; i < 6; ++i) {
        TCanvas *c = new TCanvas(Form("c_%d", i), plots[i].title, 800, 600);
        c->SetGrid();
        // Larger margins to fit the larger labels
        c->SetLeftMargin(0.14);
        c->SetBottomMargin(0.13);
        c->SetRightMargin(0.05);
        c->SetTopMargin(0.08);

        TEfficiency *effGlasgow = (TEfficiency*)fGlasgow->Get(plots[i].nameGlasgow);
        TH1F *hCurrent = (TH1F*)fCurrent->Get(plots[i].nameCurrent);

        if (!effGlasgow || !hCurrent) {
            printf("Object not found: %s or %s\n", plots[i].nameGlasgow, plots[i].nameCurrent);
            delete c;
            continue;
        }

        // Style the Glasgow curve
        effGlasgow->SetLineColor(kRed);
        effGlasgow->SetMarkerColor(kRed);
        effGlasgow->SetMarkerStyle(20);
        effGlasgow->SetMarkerSize(1.0);
        if(i<3) effGlasgow->SetTitle(Form("%s;%s;Efficiency", plots[i].title, plots[i].xAxis));
        else effGlasgow->SetTitle(Form("%s;%s;Rejection", plots[i].title, plots[i].xAxis));

        // Style the current curve
        hCurrent->SetLineColor(kGreen+2);
        hCurrent->SetMarkerColor(kGreen+2);
        hCurrent->SetMarkerStyle(21);
        hCurrent->SetMarkerSize(1.0);

        // Draw the main object
        effGlasgow->Draw("AP");
        gPad->Update(); // Force drawing so PaintedGraph is available

        if (effGlasgow->GetPaintedGraph() && effGlasgow->GetPaintedGraph()->GetHistogram()) {
            TH1 *hAxis = effGlasgow->GetPaintedGraph()->GetHistogram();

            // Dynamically determine the Y-axis minimum from the data
            double dataMin = FindMinValue(effGlasgow, hCurrent);
            double margin  = 0.01; // Small margin to keep points/error bars from being clipped
            double yMin    = std::max(0.0, dataMin - margin);

            hAxis->GetYaxis()->SetRangeUser(yMin, 1.0);

            // Enlarge axis titles and labels
            hAxis->GetXaxis()->SetTitleSize(0.05);
            hAxis->GetXaxis()->SetLabelSize(0.045);
            hAxis->GetXaxis()->SetTitleOffset(1.1);

            hAxis->GetYaxis()->SetTitleSize(0.05);
            hAxis->GetYaxis()->SetLabelSize(0.045);
            hAxis->GetYaxis()->SetTitleOffset(1.2);

            gPad->Modified();
            gPad->Update();
        }

        hCurrent->Draw("P SAME");

        // Legend
        TLegend *leg = new TLegend(0.55, 0.15, 0.88, 0.30);
        leg->SetBorderSize(0);
        leg->SetTextFont(42);
        leg->SetTextSize(0.04);
        leg->AddEntry(effGlasgow, "Glasgow presentation", "pel");
        leg->AddEntry(hCurrent, "Current XGBoost", "pel");
        leg->Draw();

        // Save to a separate PNG file
        c->SaveAs(Form("Results/%s", plots[i].fileName));

        delete c;
    }
}