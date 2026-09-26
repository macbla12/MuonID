// PlotDrichCompare.cxx
// Usage: root -l -b -q PlotDrichCompare.cxx+

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

void Plotter() {
    // ROOT style settings
    gStyle->SetOptStat(0);          // Disable the statistics box
    gStyle->SetPalette(kBird);      // Good color mapping for 2D plots
    gStyle->SetPadRightMargin(0.12);
    const string& filename = "Plots/drich_compare.root";
    const string& pdfName = "Plots/drich_plots.pdf";
    TFile* file = TFile::Open(filename.c_str(), "READ");
    if (!file || file->IsZombie()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }

    // Features generated in DrichFeatues.cxx
    vector<string> features = {
        "NPE",
        "ThetaMean",
        "ThetaMedian",
        "ThetaRMS",
        "Mass2Est",
        "WeightPi",
        "dW_PiK",
        "dW_PiE",
        "NpePiHyp"
    };

    // Radiators present in dRICH
    vector<string> radiators = {"Aerogel", "Gas"};

    bool isFirstPage = true;

    for (const auto& rad : radiators) {
        for (const auto& feat : features) {

            // =================================================================
            // 1. 1D HISTOGRAMS (Muon on top, pion on bottom of one plot)
            // File name: <Feature>_<Radiator>_<Sample>
            // =================================================================
            string h1_muon_name = feat + "_" + rad + "_Muon";
            string h1_pion_name = feat + "_" + rad + "_Pion";

            TH1D* h1_muon = (TH1D*)file->Get(h1_muon_name.c_str());
            TH1D* h1_pion = (TH1D*)file->Get(h1_pion_name.c_str());

            if (h1_muon && h1_pion) {
                TCanvas* c1 = new TCanvas(("c1_" + feat + "_" + rad).c_str(), (feat + " (" + rad + ")").c_str(), 1200, 800);
                c1->SetGrid();

                // Styling
                h1_muon->SetLineColor(kBlue + 1);
                h1_muon->SetLineWidth(2);
                
                h1_pion->SetLineColor(kRed + 1);
                h1_pion->SetLineWidth(2);

                // Automatically adjust the Y axis
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

                // Save to PDF
                if (isFirstPage) {
                    c1->SaveAs((pdfName + "(").c_str());
                    isFirstPage = false;
                } else {
                    c1->SaveAs(pdfName.c_str());
                }

                delete leg;
                delete c1;
            }

            // =================================================================
            // 2. 2D HISTOGRAMS (Muon on top, pion on bottom)
            // File name: <Feature>_vsP_<Radiator>_<Sample>
            // =================================================================
            string h2_muon_name = feat + "_vsP_" + rad + "_Muon";
            string h2_pion_name = feat + "_vsP_" + rad + "_Pion";

            TH2D* h2_muon = (TH2D*)file->Get(h2_muon_name.c_str());
            TH2D* h2_pion = (TH2D*)file->Get(h2_pion_name.c_str());

            if (h2_muon && h2_pion) {
                TCanvas* c2 = new TCanvas(("c2_" + feat + "_" + rad).c_str(), (feat + " vs P (" + rad + ")").c_str(), 1200, 800);
                c2->Divide(1, 2); // 1 kolumna, 2 wiersze

                // Upper pad: muon
                c2->cd(1);
                gPad->SetGrid();
                h2_muon->SetTitle((feat + " vs P - " + rad + " (Muon)").c_str());
                h2_muon->Draw("COLZ");

                // Lower pad: pion
                c2->cd(2);
                gPad->SetGrid();
                h2_pion->SetTitle((feat + " vs P - " + rad + " (Pion)").c_str());
                h2_pion->Draw("COLZ");

                // Save to PDF
                if (isFirstPage) {
                    c2->SaveAs((pdfName + "(").c_str());
                    isFirstPage = false;
                } else {
                    c2->SaveAs(pdfName.c_str());
                }

                delete c2;
            }
        }
    }

    // Close the PDF file if at least one page was saved
    if (!isFirstPage) {
        TCanvas* cClose = new TCanvas("cClose", "", 1200, 800);
        cClose->SaveAs((pdfName + ")").c_str());
        delete cClose;
        cout << "[OK] Saved combined dRICH plots to: " << pdfName << endl;
    } else {
        cerr << "Warning: No matching histograms found in the file." << endl;
    }

    file->Close();
    delete file;
}