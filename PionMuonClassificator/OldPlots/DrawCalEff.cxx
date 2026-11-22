void DrawSomething() {
    // Global ROOT style settings
    gStyle->SetOptStat(0);
    gStyle->SetPadTickX(1);             // Ticks at the top and bottom
    gStyle->SetPadTickY(1);             // Ticks on the left and right
    gStyle->SetGridStyle(3);            // Dotted grid
    gStyle->SetGridColor(kGray);        // Subtle grid color
    gStyle->SetTitleFont(42, "xyz");
    gStyle->SetLabelFont(42, "xyz");

    // Enlarge the main plot title
    gStyle->SetTitleFontSize(0.055);

    TFile *file = TFile::Open("MuonID_Histograms.root", "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Cannot open MuonID_Histograms.root!" << std::endl;
        return;
    }

    TH1 *h_found_mom = (TH1*)file->Get("h_eff_found_vs_mc_mom");
    TH1 *h_found_eta = (TH1*)file->Get("h_eff_found_vs_mc_eta");

    TCanvas *c1 = new TCanvas("c1", "Muon ID & Tracking Efficiencies", 1000, 600);
    c1->SetLeftMargin(0.14);   // Margin for larger tick labels and the Y-axis title
    c1->SetBottomMargin(0.14); // Margin for the larger X-axis title

    // Helper to enlarge axis elements
    auto setCustomFonts = [](TH1* h) {
        if (!h) return;
        
        // Axis titles (X and Y)
        h->GetXaxis()->SetTitleSize(0.045);
        h->GetYaxis()->SetTitleSize(0.045);
        h->GetXaxis()->SetTitleOffset(1.1);
        h->GetYaxis()->SetTitleOffset(1.3);

        // Axis tick labels (X and Y)
        h->GetXaxis()->SetLabelSize(0.040);
        h->GetYaxis()->SetLabelSize(0.040);
    };

    // 1. Plot as a function of momentum (p)
    gPad->SetGrid(1, 1);
    if (h_found_mom) {
        h_found_mom->SetTitle("Muon Calorimeter Efficiency;Momentum [GeV/c];Efficiency / Rejection");
        h_found_mom->GetYaxis()->SetRangeUser(0.0, 1.02);
        
        setCustomFonts(h_found_mom); // Apply enlarged fonts

        // Stylizacja Found vs MC (Czerwony)
        h_found_mom->SetMarkerStyle(20);
        h_found_mom->SetMarkerColor(kRed+1);
        h_found_mom->SetLineColor(kRed+1);
        h_found_mom->Draw("PE");
    }
    c1->SaveAs("Results/MuonID_Eff_Mom.png");

    // 2. Plot as a function of pseudorapidity (eta)
    c1->Clear();
    
    gPad->SetGrid(1, 1);
    if (h_found_eta) {
        h_found_eta->SetTitle("Muon Calorimeter Efficiency;#eta;Efficiency / Rejection");
        h_found_eta->GetYaxis()->SetRangeUser(0.0, 1.02);

        setCustomFonts(h_found_eta); // Apply enlarged fonts

        // Stylizacja Found vs MC (Czerwony)
        h_found_eta->SetMarkerStyle(20);
        h_found_eta->SetMarkerColor(kRed+1);
        h_found_eta->SetLineColor(kRed+1);
        h_found_eta->Draw("PE");
    }

    c1->SaveAs("Results/MuonID_Eff_Eta.png");
}