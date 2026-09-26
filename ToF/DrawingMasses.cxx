#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TColor.h>
#include <TStyle.h>
#include <TGraph.h>
#include <algorithm>

// Helper that converts a TH2F to a TGraph using only non-empty bins
TGraph* ConvertTH2ToGraph(TH2F* h2) {
    TGraph* g = new TGraph();
    int pointIdx = 0;
    for (int x = 1; x <= h2->GetNbinsX(); ++x) {
        for (int y = 1; y <= h2->GetNbinsY(); ++y) {
            if (h2->GetBinContent(x, y) > 0) {
                double xVal = h2->GetXaxis()->GetBinCenter(x);
                double yVal = h2->GetYaxis()->GetBinCenter(y);
                g->SetPoint(pointIdx++, xVal, yVal);
            }
        }
    }
    return g;
}

void DrawingMasses() {
    // 1. Open ROOT files
    TFile *fPions = TFile::Open("Plots/CaloToF/Diag_Pions.root", "READ");
    TFile *fMuons = TFile::Open("Plots/CaloToF/Diag_Muons.root", "READ");

    if (!fPions || fPions->IsZombie() || !fMuons || fMuons->IsZombie()) {
        printf("Error opening ROOT files!\n");
        return;
    }

    // Retrieve histograms from the files
    TH2F *h2_BetaMom_Pions = (TH2F*)fPions->Get("BetaVsMom_Pions");
    TH2F *h2_BetaMom_Muons = (TH2F*)fMuons->Get("BetaVsMom_Muons"); 

    TH1F *h1_Mass_Pions = (TH1F*)fPions->Get("Mass_Pions");
    TH1F *h1_Mass_Muons = (TH1F*)fMuons->Get("Mass_Muons");       

    // ==========================================
    // Style settings (global axis scaling)
    // ==========================================
    gStyle->SetOptStat(0);
    gStyle->SetLabelSize(0.045, "XYZ"); // Larger axis tick labels (default is about 0.035)
    gStyle->SetTitleSize(0.05, "XYZ");  // Larger axis titles
    gStyle->SetTitleOffset(0.9, "X");   // Offset the X-axis title
    gStyle->SetTitleOffset(0.9, "Y");   // Offset the Y-axis title

    // ==========================================
    // PLOT 1: BetaVsMom (2D using TGraph)
    // ==========================================
    
    TCanvas *c1 = new TCanvas("c1", "Beta vs Mom", 1200, 600);
    c1->SetBottomMargin(0.13); // Increase margin for the larger X-axis title
    c1->SetLeftMargin(0.12);   // Increase margin for the larger Y-axis title

    Int_t colPions = TColor::GetColorTransparent(kRed, 0.7);
    Int_t colMuons = TColor::GetColorTransparent(kBlue, 0.7);

    TGraph *gPions = ConvertTH2ToGraph(h2_BetaMom_Pions);
    TGraph *gMuons = ConvertTH2ToGraph(h2_BetaMom_Muons);

    gPions->SetMarkerStyle(20);
    gPions->SetMarkerSize(1.2);
    gPions->SetMarkerColor(colPions);

    gMuons->SetMarkerStyle(20);
    gMuons->SetMarkerSize(1.2);
    gMuons->SetMarkerColor(colMuons);
    h2_BetaMom_Pions->SetTitle("");


    // Configure axes on the main histogram
    h2_BetaMom_Pions->GetXaxis()->SetRangeUser(0.1, 1.1);
    h2_BetaMom_Pions->GetYaxis()->SetRangeUser(0.8, 1.1);
    
    // Set dedicated axis sizes for PLOT 1
    h2_BetaMom_Pions->GetXaxis()->SetLabelSize(0.05);
    h2_BetaMom_Pions->GetYaxis()->SetLabelSize(0.05);
    h2_BetaMom_Pions->GetXaxis()->SetTitleSize(0.06);
    h2_BetaMom_Pions->GetYaxis()->SetTitleSize(0.06);

    h2_BetaMom_Pions->Draw("AXIS");

    gPions->Draw("P SAME");
    gMuons->Draw("P SAME");

    TLegend *leg1 = new TLegend(0.7, 0.75, 0.88, 0.88);
    leg1->SetBorderSize(0);
    leg1->SetNColumns(1);
    leg1->SetColumnSeparation(0.1);
    leg1->SetEntrySeparation(0.1);
    leg1->SetMargin(0.15);
    leg1->SetTextFont(42);
    leg1->SetTextSize(0.045); // Larger legend font
    
    TGraph *gLegPions = (TGraph*)gPions->Clone();
    TGraph *gLegMuons = (TGraph*)gMuons->Clone();
    gLegPions->SetMarkerColor(kRed);
    gLegMuons->SetMarkerColor(kBlue);
    
    leg1->AddEntry(gLegPions, "Pions (#pi)", "p");
    leg1->AddEntry(gLegMuons, "Muons (#mu)", "p");
    leg1->Draw();

    c1->SaveAs("Plots/CaloToF/BetaVsMom.png");

    // ==========================================
    // PLOT 2: Mass Distribution
    // ==========================================

    TCanvas *c2 = new TCanvas("c2", "Mass Distribution", 1200, 600);
    c2->SetBottomMargin(0.13); // Increase margin for the larger X-axis title
    c2->SetLeftMargin(0.12);   // Increase margin for the larger Y-axis title

    TAxis *oldAxis = h1_Mass_Pions->GetXaxis();
    int nbins = oldAxis->GetNbins();
    double xmin = oldAxis->GetXmin() * 1000.0;
    double xmax = oldAxis->GetXmax() * 1000.0;

    TH1F *h1_Mass_Pions_MeV = new TH1F("h1_Mass_Pions_MeV", "Mass Distribution;m [MeV];Counts", nbins, xmin, xmax);
    TH1F *h1_Mass_Muons_MeV = new TH1F("h1_Mass_Muons_MeV", "Mass Distribution;m [MeV];Counts", nbins, xmin, xmax);

    for (int i = 1; i <= nbins; ++i) {
        h1_Mass_Pions_MeV->SetBinContent(i, h1_Mass_Pions->GetBinContent(i));
        h1_Mass_Pions_MeV->SetBinError(i, h1_Mass_Pions->GetBinError(i));
        h1_Mass_Muons_MeV->SetBinContent(i, h1_Mass_Muons->GetBinContent(i));
        h1_Mass_Muons_MeV->SetBinError(i, h1_Mass_Muons->GetBinError(i));
    }

    // Style for pions and muons
    h1_Mass_Pions_MeV->SetLineColor(kRed);
    h1_Mass_Pions_MeV->SetLineWidth(2);
    h1_Mass_Pions_MeV->SetFillColor(TColor::GetColorTransparent(kRed, 0.2));

    h1_Mass_Muons_MeV->SetLineColor(kBlue);
    h1_Mass_Muons_MeV->SetLineWidth(2);
    h1_Mass_Muons_MeV->SetFillColor(TColor::GetColorTransparent(kBlue, 0.2));

    // Set dedicated axis sizes for PLOT 2
    h1_Mass_Pions_MeV->GetXaxis()->SetLabelSize(0.05); // X-axis tick-label size
    h1_Mass_Pions_MeV->GetYaxis()->SetLabelSize(0.05); // Y-axis tick-label size
    h1_Mass_Pions_MeV->GetXaxis()->SetTitleSize(0.06);  // Size of the "m [MeV]" title
    h1_Mass_Pions_MeV->GetYaxis()->SetTitleSize(0.06);  // Size of the "Counts" title
    h1_Mass_Pions_MeV->GetYaxis()->SetTitleOffset(1.1);  // Offset the Y-axis title from tick labels

    // Adjust the Y scale and X range
    float max_y = std::max(h1_Mass_Pions_MeV->GetMaximum(), h1_Mass_Muons_MeV->GetMaximum());
    h1_Mass_Pions_MeV->SetMaximum(max_y * 1.15);
    h1_Mass_Pions_MeV->GetXaxis()->SetRangeUser(50.0, 200.0);
    h1_Mass_Pions_MeV->SetTitle("");

    h1_Mass_Pions_MeV->Draw("HIST");
    h1_Mass_Muons_MeV->Draw("HIST SAME");

    double meanPionsMeV = h1_Mass_Pions->GetMean() * 1000.0;
    double meanMuonsMeV = h1_Mass_Muons->GetMean() * 1000.0;

    TString labelPions = Form("Pions (<m> = %.1f MeV)", meanPionsMeV);
    TString labelMuons = Form("Muons (<m> = %.1f MeV)", meanMuonsMeV);

    TLegend *leg2 = new TLegend(0.2, 0.73, 0.55, 0.88);
    leg2->SetBorderSize(0);
    leg2->SetNColumns(1);
    leg2->SetColumnSeparation(0.1);
    leg2->SetEntrySeparation(0.1);
    leg2->SetMargin(0.15);
    leg2->SetTextFont(42);
    leg2->SetTextSize(0.04); // Larger legend text
    leg2->AddEntry(h1_Mass_Pions_MeV, labelPions, "f");
    leg2->AddEntry(h1_Mass_Muons_MeV, labelMuons, "f");
    leg2->Draw();

    c2->SaveAs("Plots/CaloToF/MassDistribution.png");
}