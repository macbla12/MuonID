void BetaVsMomPlot(TH1D* BetaVsMom)
{
    TCanvas *c1 = new TCanvas("c1", "Beta vs Momentum", 800, 600);
    c1->Clear();
    gPad->SetLeftMargin(0.13);   
    gPad->SetBottomMargin(0.13); 
    gPad->SetRightMargin(0.14);  
    gPad->SetLogz(1);
    BetaVsMom->GetXaxis()->SetTitle("Momentum [GeV/c]");
    BetaVsMom->GetXaxis()->SetTitleSize(0.05);
    BetaVsMom->GetXaxis()->SetLabelSize(0.045);
    BetaVsMom->GetXaxis()->SetTitleOffset(1.1);
    BetaVsMom->GetXaxis()->SetRangeUser(0.0, 1);

    BetaVsMom->GetYaxis()->SetTitle("#beta");
    BetaVsMom->GetYaxis()->SetTitleSize(0.05);
    BetaVsMom->GetYaxis()->SetLabelSize(0.045);
    BetaVsMom->GetYaxis()->SetTitleOffset(1.2);
    BetaVsMom->GetYaxis()->SetRangeUser(0.7, 1.05);
    BetaVsMom->Draw("COLZ");

    TLatex tex;
    tex.SetTextSize(0.045); 
    tex.SetNDC(kFALSE);    
    tex.DrawLatex(0.5, 0.95, "e^{#pm}/#mu^{#pm}/#pi^{#pm}");
    tex.DrawLatex(0.7, 0.8, "K^{#pm}");
    tex.DrawLatex(1, 0.75, "p^{#pm}");
    c1->SaveAs("Plots/betamom.png");
    gPad->SetLogz(0);
}
void PlottingToF()
{
    TFile *PaperFile = new TFile("Plots/ToFPlots.root","READ");
    string name;

    static constexpr int NumOfFiles=5;
    TH1D *BToFEtaRangeHist[NumOfFiles], *BToFPhiRangeHist[NumOfFiles], *EToFEtaRangeHist[NumOfFiles], *EToFPhiRangeHist[NumOfFiles];
    TH1D *ToFBTimeHist[NumOfFiles], *ToFETimeHist[NumOfFiles];
    TH1D *NumberOfBHits[NumOfFiles], *NumberOfEHits[NumOfFiles];
    TH1D *ToFTimeHist[NumOfFiles], *ToFMassHist[NumOfFiles];
    TH1D *DoubleToFMassHist[NumOfFiles];
      
    for(int i=0; i<5; i++)
    {
        if(i==0) name="Muons";
        if(i==1) name="Pions";
        if(i==2) name="Electrons";
        if(i==3) name="Kaons";
        if(i==4) name="Protons";
        BToFEtaRangeHist[i]= (TH1D*)PaperFile->Get(Form("BToFEtaRangeHist%s",name.c_str()));
        BToFPhiRangeHist[i]= (TH1D*)PaperFile->Get(Form("BToFPhiRangeHist%s",name.c_str()));

        EToFEtaRangeHist[i]= (TH1D*)PaperFile->Get(Form("EToFEtaRangeHist%s",name.c_str()));
        EToFPhiRangeHist[i]= (TH1D*)PaperFile->Get(Form("EToFPhiRangeHist%s",name.c_str()));

        ToFBTimeHist[i]= (TH1D*)PaperFile->Get(Form("ToFBarrelTimeHist%s",name.c_str()));
        ToFETimeHist[i]= (TH1D*)PaperFile->Get(Form("ToFEndcapTimeHist%s",name.c_str()));

        NumberOfBHits[i]= (TH1D*)PaperFile->Get(Form("NumberOfBarrelHits%s",name.c_str()));
        NumberOfEHits[i]= (TH1D*)PaperFile->Get(Form("NumberOfEndcapHits%s",name.c_str()));

        ToFMassHist[i]= (TH1D*)PaperFile->Get(Form("ToFMassHist%s",name.c_str()));
        DoubleToFMassHist[i]= (TH1D*)PaperFile->Get(Form("DoubleToFMassHist%s",name.c_str()));
    }
    
    TH1D *DistanceHist = (TH1D*)PaperFile->Get("DistanceHist");
    TH1D *LengthDiffrance = (TH1D*)PaperFile->Get("LengthDiffrance");
    TH1D *BetaVsMom = (TH1D*)PaperFile->Get("BetaVsMom");

    BetaVsMomPlot(BetaVsMom);
    
    TCanvas c1;
    c1.SaveAs("Plots/FirstAnalysis.pdf[");


    c1.Clear();
    NumberOfBHits[0]->GetXaxis()->SetTitle("Number of Barrel Hits");
    NumberOfBHits[0]->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");
    c1.Clear();
    NumberOfEHits[0]->GetXaxis()->SetTitle("Number of Endcap Hits");
    NumberOfEHits[0]->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");

    c1.Clear();
    BToFEtaRangeHist[0]->GetXaxis()->SetTitle("#eta");
    BToFEtaRangeHist[0]->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");

    c1.Clear();
    BToFPhiRangeHist[0]->GetXaxis()->SetTitle("#phi");
    BToFPhiRangeHist[0]->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");

        c1.Clear();
    EToFEtaRangeHist[0]->GetXaxis()->SetTitle("#eta");
    EToFEtaRangeHist[0]->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");

    c1.Clear();
    EToFPhiRangeHist[0]->GetXaxis()->SetTitle("#phi");
    EToFPhiRangeHist[0]->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");

    c1.Clear();
    LengthDiffrance->GetXaxis()->SetTitle("Length Difference [cm]");
    LengthDiffrance->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");

    c1.Clear();
    DistanceHist->GetXaxis()->SetTitle("Distance to ToF [cm]");
    DistanceHist->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");


    c1.Clear();
    gPad->SetLogz(1);
    BetaVsMom->GetXaxis()->SetTitle("Momentum [GeV/c]");
    BetaVsMom->GetYaxis()->SetTitle("#beta");
    BetaVsMom->Draw("COLZ");
    c1.SaveAs("Plots/FirstAnalysis.pdf");
    gPad->SetLogz(0);
    
    c1.Clear();
    ToFBTimeHist[0]->Scale(1./ToFBTimeHist[0]->Integral());
    ToFBTimeHist[1]->Scale(1./ToFBTimeHist[1]->Integral());
    ToFBTimeHist[2]->Scale(1./ToFBTimeHist[2]->Integral());
    ToFBTimeHist[0]->GetXaxis()->SetTitle("time [ns]");
    ToFBTimeHist[1]->GetXaxis()->SetTitle("time [ns]");
    ToFBTimeHist[2]->GetXaxis()->SetTitle("time [ns]");


    ToFBTimeHist[0]->SetLineColor(kRed);
    ToFBTimeHist[1]->SetLineColor(kBlue);
    ToFBTimeHist[2]->SetLineColor(kGreen);

    ToFBTimeHist[1]->Draw("HIST");
    ToFBTimeHist[0]->Draw("HIST SAME");
    ToFBTimeHist[2]->Draw("HIST SAME");
    c1.SaveAs("Plots/FirstAnalysis.pdf");

    c1.Clear();
    ToFETimeHist[0]->Scale(1./ToFETimeHist[0]->Integral());
    ToFETimeHist[1]->Scale(1./ToFETimeHist[1]->Integral());
    ToFETimeHist[2]->Scale(1./ToFETimeHist[2]->Integral());
    ToFETimeHist[0]->GetXaxis()->SetTitle("time [ns]");
    ToFETimeHist[1]->GetXaxis()->SetTitle("time [ns]");
    ToFETimeHist[2]->GetXaxis()->SetTitle("time [ns]");


    ToFETimeHist[0]->SetLineColor(kRed);
    ToFETimeHist[1]->SetLineColor(kBlue);
    ToFETimeHist[2]->SetLineColor(kGreen);

    ToFETimeHist[1]->Draw("HIST");
    ToFETimeHist[0]->Draw("HIST SAME");
    ToFETimeHist[2]->Draw("HIST SAME");
    c1.SaveAs("Plots/FirstAnalysis.pdf");
    /*
    c1.Clear();
    ToFMassHist[0]->Scale(1./ToFMassHist[0]->Integral());
    ToFMassHist[1]->Scale(1./ToFMassHist[1]->Integral());
    ToFMassHist[2]->Scale(1./ToFMassHist[2]->Integral());
    ToFMassHist[0]->SetTitle("Resonstructed Mass from ToF");

    ToFMassHist[0]->GetYaxis()->SetTitle("1/N dN/dm_{tof} [1/GeV]");
    

    ToFMassHist[0]->GetXaxis()->SetTitle("m_{tof} [GeV]");
    ToFMassHist[1]->GetXaxis()->SetTitle("m_{tof} [GeV]");
    ToFMassHist[2]->GetXaxis()->SetTitle("m_{tof} [GeV]");

    ToFMassHist[0]->SetLineColor(kRed);
    ToFMassHist[1]->SetLineColor(kBlue);
    ToFMassHist[2]->SetLineColor(kGreen);

    ToFMassHist[0]->Draw("HIST");
    ToFMassHist[1]->Draw("HIST SAME");
    ToFMassHist[2]->Draw("HIST SAME");
    leg->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");
    
    c1.Clear();
    DoubleToFMassHist[0]->Scale(1./DoubleToFMassHist[0]->Integral());
    DoubleToFMassHist[1]->Scale(1./DoubleToFMassHist[1]->Integral());
    DoubleToFMassHist[2]->Scale(1./DoubleToFMassHist[2]->Integral());
    DoubleToFMassHist[0]->SetTitle("Resonstructed Mass from ToF");

    DoubleToFMassHist[0]->GetYaxis()->SetTitle("1/N dN/dm_{tof} [1/GeV]");

    DoubleToFMassHist[0]->GetXaxis()->SetTitle("m_{tof} [GeV]");
    DoubleToFMassHist[1]->GetXaxis()->SetTitle("m_{tof} [GeV]");
    DoubleToFMassHist[2]->GetXaxis()->SetTitle("m_{tof} [GeV]");

    DoubleToFMassHist[0]->SetLineColor(kRed);
    DoubleToFMassHist[1]->SetLineColor(kBlue);
    DoubleToFMassHist[2]->SetLineColor(kGreen);

    DoubleToFMassHist[0]->Draw("HIST");
    DoubleToFMassHist[1]->Draw("HIST SAME");
    leg3->Draw();
    c1.SaveAs("Plots/FirstAnalysis.pdf");
    */
    
    c1.SaveAs("Plots/FirstAnalysis.pdf]");
}