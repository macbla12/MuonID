// PlotMuonMC.C
//
// Rysuje z Plots/MCDistributions.root dwa osobne wykresy dla mionów:
//   Plots/MuonMC_Eta.pdf  (MC_Eta, skala liniowa)
//   Plots/MuonMC_P.pdf    (MC_P, skala logarytmiczna na osi Y)
//
// Run: root -l -q -b PlotMuonMC.C

#include <TFile.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TLatex.h>
#include <iostream>

void DrawEPICLabel(double x = 0.78, double y = 0.84, double textSize = 0.040) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(textSize);
    latex.SetTextFont(42);
    latex.DrawLatex(x, y, "#bf{ePIC simulation}");
    latex.DrawLatex(x, y - textSize * 1.25, "single particle");
}

static void DrawAndSave(TH1F *h, const char *canvasName, const char *outFile, bool logY)
{
    // Linia i jasnoniebieskie wypełnienie
    h->SetLineWidth(3);
    h->SetLineColor(kBlue + 1);
    h->SetFillColor(kAzure - 2);
    h->SetFillStyle(1001);
    if(!logY) h->Rebin(2);

    // Osie
    h->GetXaxis()->SetTitleSize(0.050);
    h->GetYaxis()->SetTitleSize(0.050);
    h->GetXaxis()->SetTitleOffset(1.15);
    h->GetYaxis()->SetTitleOffset(1.30);

    h->GetXaxis()->SetLabelSize(0.045);
    h->GetYaxis()->SetLabelSize(0.045);

    h->GetXaxis()->SetTitleFont(42);
    h->GetYaxis()->SetTitleFont(42);
    h->GetXaxis()->SetLabelFont(42);
    h->GetYaxis()->SetLabelFont(42);

    TCanvas *c = new TCanvas(canvasName, canvasName, 900, 700);
    c->SetLeftMargin(0.17);
    c->SetBottomMargin(0.15);
    c->SetRightMargin(0.05);
    c->SetTopMargin(0.05);
    c->SetGrid();
    if (logY) {
        c->SetLogy();
        h->SetMinimum(900);   // log wymaga dodatniego minimum
    } else {
        h->SetMinimum(0);
    }

    h->Draw("HIST");
    h->Draw("HIST SAME");     // linia obrysu nad wypełnieniem
    gPad->RedrawAxis();
    DrawEPICLabel(0.70, 0.86, 0.045);

    c->SaveAs(outFile);
    std::cout << "Saved " << outFile << std::endl;
    delete c;
}

void PlotMuonMC(const char *inFile = "Plots/MCDistributions.root")
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);

    TFile *f = TFile::Open(inFile, "READ");
    if (!f || f->IsZombie()) {
        std::cerr << "[ERROR] Cannot open " << inFile << std::endl;
        return;
    }

    TH1F *hEta = (TH1F *)f->Get("Muons/MC_Eta");
    TH1F *hP   = (TH1F *)f->Get("Muons/MC_P");
    if (!hEta || !hP) {
        std::cerr << "[ERROR] Muons/MC_Eta or Muons/MC_P not found in " << inFile << std::endl;
        return;
    }

    hEta->SetTitle(";#eta;Entries");
    hP->SetTitle(";p [GeV];Entries");

    DrawAndSave(hEta, "cMuonEta", "Plots/MuonMC_Eta.pdf", false);
    DrawAndSave(hP,   "cMuonP",   "Plots/MuonMC_P.pdf",   true);

    f->Close();
}