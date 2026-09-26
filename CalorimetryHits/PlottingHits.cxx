#include <TFile.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TSystem.h>
#include <TStyle.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

void PlottingHits()
{
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kRainBow);

    // Directory for PNG output
    gSystem->Exec("mkdir -p Plots/Hits");

    // Input files
    TFile *fMu = TFile::Open("Plots/Hits/Hits_Muons.root");
    TFile *fPi = TFile::Open("Plots/Hits/Hits_Pions.root");

    if (!fMu || fMu->IsZombie()) { cout << "ERROR: Hits_Muons.root not found\n"; return; }
    if (!fPi || fPi->IsZombie()) { cout << "ERROR: Hits_Pions.root not found\n"; return; }

    // Histograms to compare
    vector<string> histNames = {
        "HitX",
        "HitY",
        "HitZ",
        "HitEnergy",
        "HitTime",
        "DeltaEta",
        "DeltaPhi"
    };

    for (auto &hname : histNames)
    {
        TH1 *hMu = (TH1*)fMu->Get(hname.c_str());
        TH1 *hPi = (TH1*)fPi->Get(hname.c_str());

        if (!hMu || !hPi) {
            cout << "Missing histogram: " << hname << endl;
            continue;
        }

        // Canvas
        TCanvas *c = new TCanvas("c", "c", 900, 700);
        c->SetGrid();

        // Style
        hMu->SetLineWidth(3);
        hPi->SetLineWidth(3);

        hMu->SetLineColor(kBlue+1);   // muons
        hPi->SetLineColor(kRed+1);    // pions

        // Normalize to compare shapes
        hMu->Scale(1.0 / hMu->Integral());
        hPi->Scale(1.0 / hPi->Integral());

        // Draw
        hMu->Draw("HIST");
        hPi->Draw("HIST SAME");

        // Legend
        TLegend *leg = new TLegend(0.65, 0.75, 0.88, 0.88);
        leg->AddEntry(hMu, "Muons", "l");
        leg->AddEntry(hPi, "Pions", "l");
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->Draw();

        // Save
        string outname = "Plots/Hits/Compare_" + hname + ".png";
        c->SaveAs(outname.c_str());

        delete c;
    }

    fMu->Close();
    fPi->Close();

    cout << "All comparison plots saved in Plots/Hits/" << endl;
}
