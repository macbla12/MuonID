// PlotCompare.cxx
// Usage: root -l -b -q PlotCompare.cxx+

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

void Plotter(){
    // ROOT style settings
    gStyle->SetOptStat(0);          // Disable the statistics box
    gStyle->SetPalette(kBird);      // Good color mapping for 2D plots
    gStyle->SetPadRightMargin(0.12);
    const string& filename = "Plots/edep_compare.root";
    const string& pdfName = "Plots/Track_plots.pdf";

    TFile* file = TFile::Open(filename.c_str(), "READ");
    if (!file || file->IsZombie()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }

    // Feature names stored in the ROOT file
    vector<string> features = {
        "EdepSum",
        "NHits",
        "EdepMean",
        "EdepMax",
        "EdepMedian"
    };

    // 1. Open the multipage PDF file
    TCanvas* cDummy = new TCanvas("cDummy", "", 1200, 800);
    cDummy->SaveAs((pdfName + "(").c_str());
    delete cDummy;

    for (const auto& feat : features) {
        // =====================================================================
        // 1. 1D HISTOGRAMS (Muon on top, pion on bottom of one plot)
        // =====================================================================
        string h1_muon_name = feat + "_Muon";
        string h1_pion_name = feat + "_Pion";

        TH1D* h1_muon = (TH1D*)file->Get(h1_muon_name.c_str());
        TH1D* h1_pion = (TH1D*)file->Get(h1_pion_name.c_str());

        if (h1_muon && h1_pion) {
            TCanvas* c1 = new TCanvas(("c1_" + feat).c_str(), feat.c_str(), 1200, 800);
            c1->SetGrid();

            // Styling
            h1_muon->SetLineColor(kBlue + 1);
            h1_muon->SetLineWidth(2);
            
            h1_pion->SetLineColor(kRed + 1);
            h1_pion->SetLineWidth(2);

            // Find the maximum so the Y axis fits both plots
            double max_y = max(h1_muon->GetMaximum(), h1_pion->GetMaximum()) * 1.15;
            h1_muon->SetMaximum(max_y);

            // Draw
            h1_muon->Draw("HIST");
            h1_pion->Draw("HIST SAME");

            // Legend
            TLegend* leg = new TLegend(0.68, 0.75, 0.88, 0.88);
            leg->SetBorderSize(1);
            leg->SetFillColor(kWhite);
            leg->AddEntry(h1_muon, "Muon", "l");
            leg->AddEntry(h1_pion, "Pion", "l");
            leg->Draw();

            // Save the next PDF page
            c1->SaveAs(pdfName.c_str());

            delete leg;
            delete c1;
        }

        // =====================================================================
        // 2. 2D HISTOGRAMS (Muon on top, pion on bottom)
        // =====================================================================
        string h2_muon_name = feat + "_vsP_Muon";
        string h2_pion_name = feat + "_vsP_Pion";

        TH2D* h2_muon = (TH2D*)file->Get(h2_muon_name.c_str());
        TH2D* h2_pion = (TH2D*)file->Get(h2_pion_name.c_str());

        if (h2_muon && h2_pion) {
            TCanvas* c2 = new TCanvas(("c2_" + feat).c_str(), feat.c_str(), 1200, 800);
            c2->Divide(1, 2); // 1 kolumna, 2 wiersze

            // Upper pad: muon
            c2->cd(1);
            gPad->SetGrid();
            h2_muon->SetTitle((feat + " vs P - Muon").c_str());
            h2_muon->Draw("COLZ");

            // Lower pad: pion
            c2->cd(2);
            gPad->SetGrid();
            h2_pion->SetTitle((feat + " vs P - Pion").c_str());
            h2_pion->Draw("COLZ");

            // Save the next PDF page
            c2->SaveAs(pdfName.c_str());

            delete c2;
        }
    }

    // 2. Close the PDF file
    TCanvas* cClose = new TCanvas("cClose", "", 1200, 800);
    cClose->SaveAs((pdfName + ")").c_str());
    delete cClose;

    file->Close();
    delete file;
}