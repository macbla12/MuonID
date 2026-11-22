#include <utility>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <TSystem.h>

// Helper for drawing the ePIC label
void DrawEPICLabel(double x = 0.18, double y = 0.84, double textSize = 0.040) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(textSize);
    latex.SetTextFont(42);
    latex.DrawLatex(x, y, "#bf{ePIC simulation}");
    latex.DrawLatex(x, y - textSize * 1.25, "single particle");
}

std::pair<double, double> GetTailCuts(TH1D* h, double lowQuantile = 0.005, double highQuantile = 0.995) {
    if (!h || h->Integral() <= 0) {
        return {h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax()};
    }

    double totalIntegral = h->Integral();
    double runningSum = 0.0;
    int nbins = h->GetNbinsX();

    double xMinCut = h->GetXaxis()->GetXmin();
    double xMaxCut = h->GetXaxis()->GetXmax();
    bool foundMin = false;

    for (int i = 1; i <= nbins; ++i) {
        runningSum += h->GetBinContent(i);
        double fraction = runningSum / totalIntegral;

        if (!foundMin && fraction >= lowQuantile) {
            int safeBinMin = max(i - 2, 1);
            xMinCut = h->GetXaxis()->GetBinLowEdge(safeBinMin);
            foundMin = true;
        }

        if (fraction >= highQuantile) {
            int safeBinMax = min(i + 2, nbins);
            xMaxCut = h->GetXaxis()->GetBinUpEdge(safeBinMax);
            break;
        }
    }

    return {xMinCut, xMaxCut};
}

// Struktura przechowująca informacje o cząstce
struct ParticleInfo {
    string name;
    string fileName;
    TFile* file;
    Color_t color;
};

void PlotAllFeaturesHighP() {
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(1);              
    gStyle->SetTitleBorderSize(0);       
    gStyle->SetTitleFontSize(0.050);     

    gStyle->SetPadLeftMargin(0.12);
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadRightMargin(0.02);
    gStyle->SetPadTopMargin(0.02);

    // Utworzenie folderu na pojedyncze pliki PDF, jeśli nie istnieje
    gSystem->mkdir("Plots/HighP", kTRUE);

    std::vector<ParticleInfo> particles = {
        {"Electron", "Plots/FeaturesHighP_Electrons.root", nullptr, kOrange + 4},
        {"Muon",     "Plots/FeaturesHighP_Muons.root",     nullptr, kBlue + 1},
        {"Pion",     "Plots/FeaturesHighP_Pions.root",     nullptr, kRed + 1},
        {"Kaon",     "Plots/FeaturesHighP_Kaons.root",     nullptr, kGreen + 2},
        {"Proton",   "Plots/FeaturesHighP_Protons.root",   nullptr, kMagenta + 1}
    };
   
    // Otwieranie plików z uwzględnieniem folderu zapasowego Plots/Hits/
    bool allFilesOpened = true;
    for (auto& p : particles) {
        string path = p.fileName;
        p.file = TFile::Open(path.c_str(), "READ");
        if (!p.file || p.file->IsZombie()) {
            path = p.fileName;
            p.file = TFile::Open(path.c_str(), "READ");
        }
        if (!p.file || p.file->IsZombie()) {
            cerr << "[ERROR] Cannot open " << p.fileName << " (nor in Plots/Hits/)!" << endl;
            allFilesOpened = false;
        }
    }

    if (!allFilesOpened) {
        cerr << "[ERROR] One or more ROOT files could not be opened. Aborting." << endl;
        for (auto& p : particles) {
            if (p.file) { p.file->Close(); delete p.file; }
        }
        return;
    }

    const string pdfName = "Plots/FeaturesAll_5Particle_Comparison_HighP.pdf";
    bool isFirstPage = true;

    // Używamy pierwszego pliku (np. Electron) jako wzorca kluczy/histogramów
    TIter next(particles[0].file->GetListOfKeys());
    TKey* key = nullptr;

    while ((key = (TKey*)next())) {
        TClass* cl = gROOT->GetClass(key->GetClassName());
        
        // Pomijamy pliki 2D oraz obiekty inne niż TH1
        if (!cl || !cl->InheritsFrom("TH1") || cl->InheritsFrom("TH2")) continue;

        string hName = key->GetName();
        TString name = hName;
        
        // Pobieranie i klonowanie histogramów dla każdej cząstki
        std::vector<TH1D*> hists;
        bool missingHisto = false;

        for (size_t i = 0; i < particles.size(); ++i) {
            TH1D* h_orig = (TH1D*)particles[i].file->Get(hName.c_str());
            if (!h_orig) {
                missingHisto = true;
                break;
            }
            TH1D* h_norm = (TH1D*)h_orig->Clone((hName + "_norm_" + particles[i].name).c_str());
            
            // 1. Normalizacja do 1
            if (h_norm->Integral() > 0) h_norm->Scale(1.0 / h_norm->Integral());
            hists.push_back(h_norm);
        }

        if (missingHisto) {
            // Jeśli brakuje histogramu w którymś pliku, czyścimy pamięć i pomijamy
            for (auto h : hists) delete h;
            continue;
        }

        // 2. Automatyczne przycinanie ogonów dla wszystkich cząstek i znalezienie wspólnego zakresu X
        double xMinSmart = hists[0]->GetXaxis()->GetXmin();
        double xMaxSmart = hists[0]->GetXaxis()->GetXmax();

        for (size_t i = 0; i < hists.size(); ++i) {
            auto [pMin, pMax] = GetTailCuts(hists[i], 0.01, 0.96);
            if (i == 0) {
                xMinSmart = pMin;
                xMaxSmart = pMax;
            } else {
                xMinSmart = min(xMinSmart, pMin);
                xMaxSmart = max(xMaxSmart, pMax);
            }
        }

        for (auto h : hists) {
            h->GetXaxis()->SetRangeUser(xMinSmart, xMaxSmart);
        }

        // 3. Maksymalna wartość Y w znormalizowanym widoku spośród wszystkich cząstek
        double maxVal = 0.0;
        int binMin = hists[0]->GetXaxis()->FindBin(xMinSmart);
        int binMax = hists[0]->GetXaxis()->FindBin(xMaxSmart);

        for (auto h : hists) {
            for (int b = binMin; b <= binMax; ++b) {
                maxVal = max(maxVal, h->GetBinContent(b));
            }
        }

        if (maxVal <= 0) maxVal = 1.0;

        hists[0]->SetMinimum(0.0);
        hists[0]->SetMaximum(maxVal * 1.30); // 30% zapasu nad najwyższym punktem

        // Tworzenie Canvas
        TCanvas* c = new TCanvas(("c_" + hName).c_str(), hName.c_str(), 1000, 800);
        c->SetGrid();
        
        if (name.EndsWith("_h_SpreadR") || name.EndsWith("_h_R_Disp") || name.EndsWith("_h_R_DispWeighted") || name.EndsWith("_h_EnergyStdDev")) {
            hists[0]->SetMinimum(0.001);
            hists[0]->SetMaximum(maxVal * 15); 
            gPad->SetLogy(1);
        } else {
            gPad->SetLogy(0);
        }

        hists[0]->SetTitle("");

        // Stylowanie osi i pierwszego histogramu
        hists[0]->SetLineColor(particles[0].color);
        hists[0]->SetLineWidth(3);

        hists[0]->GetXaxis()->SetTitleSize(0.050);
        hists[0]->GetYaxis()->SetTitleSize(0.050);
        hists[0]->GetXaxis()->SetTitleOffset(1.15);
        hists[0]->GetYaxis()->SetTitleOffset(1.30);
        hists[0]->GetYaxis()->SetTitle("Normalized Counts");

        hists[0]->GetXaxis()->SetLabelSize(0.045);
        hists[0]->GetYaxis()->SetLabelSize(0.045);
        
        hists[0]->GetXaxis()->SetTitleFont(42);
        hists[0]->GetYaxis()->SetTitleFont(42);
        hists[0]->GetXaxis()->SetLabelFont(42);
        hists[0]->GetYaxis()->SetLabelFont(42);

        // Rysowanie pierwszego histogramu
        hists[0]->Draw("HIST");

        // Rysowanie kolejnych histogramów oraz konfiguracja stylów linii
        for (size_t i = 1; i < hists.size(); ++i) {
            hists[i]->SetLineColor(particles[i].color);
            hists[i]->SetLineWidth(3);
            hists[i]->Draw("HIST SAME");
        }

        // Legenda dla 5 cząstek
        TLegend* leg = new TLegend(0.65, 0.70, 0.88, 0.90);
        leg->SetBorderSize(1);
        leg->SetFillColor(kWhite);
        leg->SetTextSize(0.035);
        leg->SetTextFont(42);
        
        for (size_t i = 0; i < particles.size(); ++i) {
            leg->AddEntry(hists[i], particles[i].name.c_str(), "l");
        }
        leg->Draw();

        DrawEPICLabel(0.15, 0.92, 0.040);

        // 1. Zapis do osobnego pliku PDF w folderze Plots/HighP/
        c->SaveAs(("Plots/HighP/" + hName + ".pdf").c_str());

        // 2. Zapis do głównego pliku PDF wielostronicowego
        if (isFirstPage) {
            c->SaveAs((pdfName + "(").c_str());
            isFirstPage = false;
        } else {
            c->SaveAs(pdfName.c_str());
        }

        // Czyszczenie pamięci dla danej strony
        delete leg;
        for (auto h : hists) delete h;
        delete c;
    }

    if (!isFirstPage) {
        TCanvas* cClose = new TCanvas("cClose", "", 1000, 800);
        cClose->SaveAs((pdfName + ")").c_str());
        delete cClose;
        cout << "[OK] Saved 5-particle comparison plots to: " << pdfName << " and individual PDFs to Plots/HighP/" << endl;
    }

    // Zamknięcie i zwolnienie plików ROOT
    for (auto& p : particles) {
        if (p.file) {
            p.file->Close();
            delete p.file;
        }
    }
}