// Plotter.cxx
// Usage: root -l -b -q Plotter.cxx+

#include <TFile.h>
#include <TH1F.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLatex.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

// Helper for drawing the ePIC label
void DrawEPICLabel() {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.035);
    latex.SetTextFont(42);
    latex.DrawLatex(0.18, 0.84, "#bf{ePIC simulation}");
    latex.DrawLatex(0.18, 0.79, "single particle");
}

void Plotter() {
    gStyle->SetOptStat(0);
    gStyle->SetPadRightMargin(0.05);
    gStyle->SetPadLeftMargin(0.12);
    gStyle->SetPadBottomMargin(0.12);

    const string filename = "TestingMuonID.root";
    const string pdfName = "Plots/MuonID_plots.pdf";

    TFile* file = TFile::Open(filename.c_str(), "READ");
    if (!file || file->IsZombie()) {
        file = TFile::Open("Plots/TestingMuonID.root", "READ");
        if (!file || file->IsZombie()) {
            cerr << "[ERROR] Cannot open TestingMuonID.root!" << endl;
            return;
        }
    }

    bool isFirstPage = true;

    // =========================================================================
    // 1. RESPONSE HISTOGRAMS (log Y)
    // =========================================================================
    struct ResponseConfig {
        string muonName;
        string pionName;
        string title;
    };

    vector<ResponseConfig> respConfigs = {
        {"h_Response_Muon_LowP",  "h_Response_Pion_LowP",  "Response (Low p)"},
        {"h_Response_Muon_HighP", "h_Response_Pion_HighP", "Response (High p)"},
        {"h_Response_Muon",       "h_Response_Pion",       "Response (Inclusive)"}
    };

    for (const auto& cfg : respConfigs) {
        TH1D* h_muon = (TH1D*)file->Get(cfg.muonName.c_str());
        TH1D* h_pion = (TH1D*)file->Get(cfg.pionName.c_str());

        if (!h_muon || !h_pion) {
            cerr << "[WARNING] Missing histogram: " << cfg.muonName << " or " << cfg.pionName << endl;
            continue;
        }

        TCanvas* c1 = new TCanvas(("c_" + cfg.muonName).c_str(), cfg.title.c_str(), 1000, 700);
        c1->SetGrid();
        gPad->SetLogy(1);
        
        h_muon->Scale(1.0 / h_muon->GetEntries());
        h_pion->Scale(1.0 / h_pion->GetEntries());

        h_muon->SetLineColor(kBlue + 1);    
        h_muon->SetLineWidth(2);
        h_pion->SetLineColor(kRed + 1);
        h_pion->SetLineWidth(2);

        double max_y = max(h_muon->GetMaximum(), h_pion->GetMaximum()) * 2.0; // Extra headroom for the logarithmic scale
        h_muon->SetMaximum(max_y);
        h_muon->SetTitle((cfg.title + ";Classifier Response;Normalized Counts").c_str());

        h_muon->Draw("HIST");
        h_pion->Draw("HIST SAME");

        TLegend* leg = new TLegend(0.68, 0.75, 0.88, 0.88);
        leg->SetBorderSize(1);
        leg->SetFillColor(kWhite);
        leg->AddEntry(h_muon, "Muon", "l");
        leg->AddEntry(h_pion, "Pion", "l");
        leg->Draw();

        DrawEPICLabel();

        if (isFirstPage) {
            c1->SaveAs((pdfName + "(").c_str());
            isFirstPage = false;
        } else {
            c1->SaveAs(pdfName.c_str());
        }

        delete leg;
        delete c1;
    }

    // =========================================================================
    // 2. EFFICIENCY / REJECTION PLOTS (TH1F)
    // =========================================================================
    struct EffConfig {
        string effName;
        string title;
        double yMin; // Easily adjust the Y range
        double yMax; // Easily adjust the Y range
    };

    vector<EffConfig> effConfigs = {
        {"h_Muon_Efficiency_vs_Pt",            "Muon Efficiency vs p_{T};p_{T} [GeV/c];Efficiency",     0.75, 1.05},
        {"h_Muon_Efficiency_vs_P",             "Muon Efficiency vs p;p [GeV/c];Efficiency",              0.75, 1.05},
        {"h_Muon_Efficiency_vs_Eta",           "Muon Efficiency vs #eta;#eta;Efficiency",                 0.75, 1.05},
        {"h_Pion_Rejection_vs_Pt",             "Pion Rejection vs p_{T};p_{T} [GeV/c];Rejection Efficiency",   0.75, 1.05},
        {"h_Pion_Rejection_vs_P",              "Pion Rejection vs p;p [GeV/c];Rejection Efficiency",          0.75, 1.05},
        {"h_Pion_Rejection_vs_Eta",            "Pion Rejection vs #eta;#eta;Rejection Efficiency",             0.75, 1.05},
        {"h_Muon_Efficiency_vs_P_LowP",        "Muon Efficiency vs p (Low p);p [GeV/c];Efficiency",        0.75, 1.05},
        {"h_Muon_Efficiency_vs_P_HighP",       "Muon Efficiency vs p (High p);p [GeV/c];Efficiency",       0.75, 1.05},
        {"h_Pion_Rejection_vs_P_LowP",         "Pion Rejection vs p (Low p);p [GeV/c];Rejection Efficiency",    0.5, 1.05},
        {"h_Pion_Rejection_vs_P_HighP",        "Pion Rejection vs p (High p);p [GeV/c];Rejection Efficiency",   0.75, 1.05},
        {"h_Muon_Efficiency_vs_P_LowP_withToF", "Muon Efficiency vs p (with ToF);p [GeV/c];Efficiency",    0.75, 1.05},
        {"h_Muon_Efficiency_vs_P_LowP_noToF",   "Muon Efficiency vs p (no ToF);p [GeV/c];Efficiency",      0.5, 1.05}
    };

    for (const auto& cfg : effConfigs) {
        TH1F* h_eff = (TH1F*)file->Get(cfg.effName.c_str());

        if (!h_eff) {
            cerr << "[WARNING] TH1F not found: " << cfg.effName << endl;
            continue;
        }

        TCanvas* c2 = new TCanvas(("c_" + cfg.effName).c_str(), cfg.title.c_str(), 1000, 700);
        c2->SetGrid();

        // Stylizacja histogramu
        h_eff->SetTitle(cfg.title.c_str());
        h_eff->SetLineColor(kBlue + 2);
        h_eff->SetMarkerColor(kBlue + 2);
        h_eff->SetMarkerStyle(20);
        h_eff->SetMarkerSize(0.9);
        
        // Set Y limits directly for TH1F
        h_eff->SetMinimum(cfg.yMin);
        h_eff->SetMaximum(cfg.yMax);

        // "E1" draws points with vertical error bars. If there are no errors, use "HIST P".
        h_eff->Draw("E1"); 

        DrawEPICLabel();

        if (isFirstPage) {
            c2->SaveAs((pdfName + "(").c_str());
            isFirstPage = false;
        } else {
            c2->SaveAs(pdfName.c_str());
        }

        delete c2;
    }

    // Close the PDF file
    if (!isFirstPage) {
        TCanvas* cClose = new TCanvas("cClose", "", 1000, 700);
        cClose->SaveAs((pdfName + ")").c_str());
        delete cClose;
        cout << "[OK] Saved plots to: " << pdfName << endl;
    } else {
        cerr << "[ERROR] No objects were drawn." << endl;
    }

    file->Close();
    delete file;
}