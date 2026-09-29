// plotEffRej.C
// Draw two plots:
//   1) Muon Efficiency + Pion Rejection vs Momentum (P)
//   2) Muon Efficiency + Pion Rejection vs Eta
//
// Usage in ROOT:
//   root -l 'plotEffRej.C("plik_wejsciowy.root")'

void DrawingresultsLowMom() {

    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(1);

    // Larger axis fonts
    gStyle->SetLabelFont(62, "XYZ");
    gStyle->SetTitleFont(62, "XYZ");
    gStyle->SetLabelSize(0.045, "XYZ");
    gStyle->SetTitleSize(0.05, "XYZ");
    gStyle->SetTitleOffset(1.15, "X");
    gStyle->SetTitleOffset(1.15, "Y");
    gStyle->SetTitleFontSize(0.05);

    TFile infile="TestingMuonID_Performance_CombinedLow.root";
    TFile *f = TFile::Open(infile);
    if (!f || f->IsZombie()) {
        printf("Cannot open file: %s\n", infile);
        return;
    }

    // --- Histograms vs. pT ---
    TH1D *h_muEff_p  = (TH1D*)f->Get("h_Muon_Efficiency_vs_Pt");
    TH1D *h_piRej_p  = (TH1D*)f->Get("h_Pion_Rejection_vs_Pt");

    // --- Histograms vs. eta ---
    TH1D *h_muEff_eta = (TH1D*)f->Get("h_Muon_Efficiency_vs_Eta");
    TH1D *h_piRej_eta = (TH1D*)f->Get("h_Pion_Rejection_vs_Eta");

    if (!h_muEff_p || !h_piRej_p || !h_muEff_eta || !h_piRej_eta) {
        printf("One or more histograms are missing from the file; check their names.\n");
        return;
    }

    auto styleHist = [](TH1D* h, int color, int marker, const char* xtitle) {
        h->SetMarkerStyle(marker);
        h->SetMarkerColor(color);
        h->SetLineColor(color);
        h->SetMarkerSize(1.1);
        h->GetXaxis()->SetTitle(xtitle);
        h->GetYaxis()->SetTitle("Efficiency / Rejection");
        h->GetYaxis()->SetRangeUser(0.0, 1.05);
    };

    // ================= Canvas 1: vs P =================
    TCanvas *c1 = new TCanvas("c1", "Efficiency and Rejection vs P", 900, 700);
    c1->SetGrid();

    styleHist(h_muEff_p, kRed + 1, 20, "Momentum [GeV/c]");
    styleHist(h_piRej_p, kGreen+ 1, 21, "Momentum [GeV/c]");

    h_muEff_p->SetTitle("");
    h_muEff_p->Draw("P");
    h_piRej_p->Draw("P SAME");

    TLegend *leg1 = new TLegend(0.62, 0.18, 0.88, 0.35);
    leg1->SetTextSize(0.04);
    leg1->SetBorderSize(0);
    leg1->AddEntry(h_muEff_p, "Muon efficiency", "lp");
    leg1->AddEntry(h_piRej_p, "Pion rejection", "lp");
    leg1->Draw();

    c1->SaveAs("EffRej_vs_P.png");

    // ================= Canvas 2: vs Eta =================
    TCanvas *c2 = new TCanvas("c2", "Efficiency and Rejection vs Eta", 900, 700);
    c2->SetGrid();

    styleHist(h_muEff_eta, kRed + 1, 20, "#eta");
    styleHist(h_piRej_eta, kGreen + 1, 21, "#eta");

    h_muEff_eta->SetTitle("");
    h_muEff_eta->Draw("P");
    h_piRej_eta->Draw("P SAME");

    TLegend *leg2 = new TLegend(0.62, 0.18, 0.88, 0.35);
    leg2->SetTextSize(0.04);
    leg2->SetBorderSize(0);
    leg2->AddEntry(h_muEff_eta, "Muon efficiency", "lp");
    leg2->AddEntry(h_piRej_eta, "Pion rejection", "lp");
    leg2->Draw();

    c2->SaveAs("EffRej_vs_Eta.png");
}