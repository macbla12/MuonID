// ComparePlots.cxx
// Porownanie histogramow: HitFeatures_Branches.root vs HitFeatures_Podio.root
// Uruchom: root -l -q ComparePlots.cxx
//
// Dla kazdego TH1 z pliku Branches:
//   - gorny pad: nalozone rozklady (bez normalizacji - te same eventy, wiec liczby maja sie zgadzac)
//   - dolny pad: stosunek Podio / Branches (1.0 = zgodnosc)
// Do konsoli wypisuje tabelke: entries, integral, liczba roznych binow.

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

using namespace std;

void DrawEPICLabel(double x = 0.18, double y = 0.84, double textSize = 0.040) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(textSize);
    latex.SetTextFont(42);
    latex.DrawLatex(x, y, "#bf{ePIC simulation}");
    latex.DrawLatex(x, y - textSize * 1.25, "single particle");
}

void ComparePlots(const char *branchFile = "HitFeatures_Branches.root",
                  const char *podioFile  = "HitFeatures_Podio.root")
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TFile *fBr = TFile::Open(branchFile, "READ");
    TFile *fPo = TFile::Open(podioFile, "READ");
    if (!fBr || fBr->IsZombie() || !fPo || fPo->IsZombie()) {
        cerr << "[ERROR] Cannot open " << branchFile << " or " << podioFile << endl;
        return;
    }

    const string pdfName = "Compare_Podio_vs_Branches.pdf";
    bool isFirstPage = true;

    cout << left << setw(48) << "histogram"
         << right << setw(12) << "N_branch" << setw(12) << "N_podio"
         << setw(14) << "Int_branch" << setw(14) << "Int_podio"
         << setw(10) << "diffBins" << endl;

    TIter next(fBr->GetListOfKeys());
    TKey *key = nullptr;

    while ((key = (TKey *)next())) {
        TClass *cl = gROOT->GetClass(key->GetClassName());
        if (!cl || !cl->InheritsFrom("TH1") || cl->InheritsFrom("TH2")) continue;

        string hName = key->GetName();
        TH1 *hBrOrig = (TH1 *)fBr->Get(hName.c_str());
        TH1 *hPoOrig = (TH1 *)fPo->Get(hName.c_str());
        if (!hBrOrig || !hPoOrig) {
            cout << left << setw(48) << hName << "  MISSING in one of the files" << endl;
            continue;
        }

        TH1 *hBr = (TH1 *)hBrOrig->Clone((hName + "_br").c_str());
        TH1 *hPo = (TH1 *)hPoOrig->Clone((hName + "_po").c_str());
        hBr->SetDirectory(nullptr);
        hPo->SetDirectory(nullptr);

        // ---- podsumowanie w konsoli ----
        int diffBins = 0;
        for (int b = 1; b <= hBr->GetNbinsX(); ++b)
            if (fabs(hBr->GetBinContent(b) - hPo->GetBinContent(b)) > 1e-9) diffBins++;

        cout << left << setw(48) << hName
             << right << setw(12) << (long long)hBr->GetEntries() << setw(12) << (long long)hPo->GetEntries()
             << setw(14) << setprecision(6) << hBr->Integral() << setw(14) << hPo->Integral()
             << setw(10) << diffBins << endl;

        // ---- kanwa z dwoma padami ----
        TCanvas *c = new TCanvas(("c_" + hName).c_str(), hName.c_str(), 1000, 900);
        TPad *pTop = new TPad("pTop", "", 0, 0.30, 1, 1.0);
        TPad *pBot = new TPad("pBot", "", 0, 0.00, 1, 0.30);
        pTop->SetBottomMargin(0.02); pTop->SetLeftMargin(0.16); pTop->SetRightMargin(0.05); pTop->SetTopMargin(0.08);
        pBot->SetTopMargin(0.03);    pBot->SetBottomMargin(0.36); pBot->SetLeftMargin(0.16); pBot->SetRightMargin(0.05);
        pTop->SetGrid(); pBot->SetGrid();
        pTop->Draw(); pBot->Draw();

        // ---- gorny pad ----
        pTop->cd();

        bool logY = (hName.find("HitE") != string::npos) || (hName.find("HitT") != string::npos) ||
                    (hName.find("SumE") != string::npos) || (hName.find("NHits") != string::npos);
        pTop->SetLogy(logY);

        double maxVal = max(hBr->GetMaximum(), hPo->GetMaximum());
        if (maxVal <= 0) maxVal = 1.0;
        hBr->SetMinimum(logY ? 0.5 : 0.0);
        hBr->SetMaximum(logY ? maxVal * 20 : maxVal * 1.35);

        hBr->SetLineColor(kBlue + 1);  hBr->SetLineWidth(3);
        hPo->SetLineColor(kRed + 1);   hPo->SetLineWidth(2); hPo->SetLineStyle(2);
        hPo->SetMarkerColor(kRed + 1); hPo->SetMarkerStyle(24); hPo->SetMarkerSize(0.8);

        hBr->GetXaxis()->SetLabelSize(0);
        hBr->GetXaxis()->SetTitleSize(0);
        hBr->GetYaxis()->SetTitle("Counts");
        hBr->GetYaxis()->SetTitleSize(0.055);
        hBr->GetYaxis()->SetLabelSize(0.048);
        hBr->GetYaxis()->SetTitleOffset(1.25);
        hBr->GetYaxis()->SetTitleFont(42);
        hBr->GetYaxis()->SetLabelFont(42);

        hBr->Draw("HIST");
        hPo->Draw("HIST P SAME");

        TLatex title;
        title.SetNDC(); title.SetTextFont(42); title.SetTextSize(0.045);
        title.DrawLatex(0.16, 0.935, hName.c_str());

        TLegend *leg = new TLegend(0.62, 0.68, 0.92, 0.88);
        leg->SetBorderSize(1); leg->SetFillColor(kWhite);
        leg->SetTextSize(0.045); leg->SetTextFont(42);
        leg->AddEntry(hBr, "Branches (TTreeReader)", "l");
        leg->AddEntry(hPo, "Podio (EDM4eic)", "lp");
        leg->Draw();

        DrawEPICLabel(0.20, 0.82, 0.045);

        // ---- dolny pad: ratio Podio / Branches ----
        pBot->cd();
        TH1 *hRatio = (TH1 *)hPo->Clone((hName + "_ratio").c_str());
        hRatio->SetDirectory(nullptr);
        hRatio->Divide(hBr);   // biny z pustym mianownikiem -> 0 (nie rysowane jako punkt)

        hRatio->SetMinimum(0.5);
        hRatio->SetMaximum(1.5);
        hRatio->SetMarkerStyle(20);
        hRatio->SetMarkerSize(0.7);
        hRatio->SetMarkerColor(kBlack);
        hRatio->SetLineColor(kBlack);
        hRatio->SetLineStyle(1);

        const double s = 0.70 / 0.30;  // skala fontow dla mniejszego padu
        hRatio->GetXaxis()->SetTitle(hBrOrig->GetXaxis()->GetTitle());
        hRatio->GetXaxis()->SetTitleSize(0.055 * 0.70 * s / 1.0 * 0.50);
        hRatio->GetXaxis()->SetLabelSize(0.048 * s * 0.70);
        hRatio->GetXaxis()->SetTitleOffset(1.10);
        hRatio->GetYaxis()->SetTitle("Podio / Branches");
        hRatio->GetYaxis()->SetTitleSize(0.055 * s * 0.70);
        hRatio->GetYaxis()->SetLabelSize(0.048 * s * 0.70);
        hRatio->GetYaxis()->SetTitleOffset(0.60);
        hRatio->GetYaxis()->SetNdivisions(505);
        hRatio->GetXaxis()->SetTitleFont(42); hRatio->GetYaxis()->SetTitleFont(42);
        hRatio->GetXaxis()->SetLabelFont(42); hRatio->GetYaxis()->SetLabelFont(42);

        hRatio->Draw("P");

        TLine *one = new TLine(hRatio->GetXaxis()->GetXmin(), 1.0, hRatio->GetXaxis()->GetXmax(), 1.0);
        one->SetLineColor(kRed + 1); one->SetLineStyle(7); one->SetLineWidth(2);
        one->Draw();

        // ---- zapis ----
        if (isFirstPage) { c->SaveAs((pdfName + "(").c_str()); isFirstPage = false; }
        else             { c->SaveAs(pdfName.c_str()); }

        delete one; delete leg; delete hRatio; delete hBr; delete hPo; delete c;
    }

    if (!isFirstPage) {
        TCanvas *cClose = new TCanvas("cClose", "", 1000, 900);
        cClose->SaveAs((pdfName + ")").c_str());
        delete cClose;
        cout << "[OK] Saved comparison to: " << pdfName << endl;
    }

    fBr->Close(); fPo->Close();
    delete fBr; delete fPo;
}