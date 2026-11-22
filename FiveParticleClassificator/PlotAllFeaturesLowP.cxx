// PlotAllDiagFeatures.cxx
// Usage: root -l -b -q PlotAllDiagFeatures.cxx+

#include <TFile.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TKey.h>
#include <TClass.h>
#include <TROOT.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

// Struktura przechowująca informacje o cząstce
struct ParticleInfo {
    string name;       // np. "Electron", "Muon", ...
    string filePath;   // ścieżka do pliku ROOT
    TFile* file;       // wskaźnik na otwarty plik
    Color_t color;     // kolor wykresu
};

// Helper for drawing the ePIC label
void DrawEPICLabel(double x = 0.18, double y = 0.95, double textSize = 0.040) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(textSize);
    latex.SetTextFont(42);
    latex.DrawLatex(x, y, "#bf{ePIC simulation}");
    latex.DrawLatex(x, y - textSize * 1.25, "single particle");
}

// Determine automatic tail cutoffs (0.5% on each side by default)
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

        // Cut the left tail
        if (!foundMin && fraction >= lowQuantile) {
            int safeBinMin = max(i - 2, 1);
            xMinCut = h->GetXaxis()->GetBinLowEdge(safeBinMin);
            foundMin = true;
        }

        // Cut the right tail
        if (fraction >= highQuantile) {
            int safeBinMax = min(i + 2, nbins);
            xMaxCut = h->GetXaxis()->GetBinUpEdge(safeBinMax);
            break;
        }
    }

    return {xMinCut, xMaxCut};
}

void PlotAllFeaturesLowP() {
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(1);              
    gStyle->SetTitleBorderSize(0);       
    gStyle->SetTitleFontSize(0.050);     

    gStyle->SetPadLeftMargin(0.12);
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadRightMargin(0.02);
    gStyle->SetPadTopMargin(0.02);

    // ==========================================
    // KONFIGURACJA: Wpisz tutaj nazwy (lub fragmenty nazw) 
    // histogramów HCal, dla których chcesz włączyć Rebin(2)
    // ==========================================
    vector<string> hcalRebinHistos = {
        "HCal_h_EnergyStdDev",    
        "HCal_h_Energy",
        "HCal_h_Eta_DispWeighted",
        "HCal_h_Phi_DispWeighted",

    };

    // Definicja cząstek do porównania
    vector<ParticleInfo> particles = {
        {"Electron", "Plots/FeaturesLowP_Electrons.root", nullptr, kOrange + 4},
        {"Muon",     "Plots/FeaturesLowP_Muons.root",     nullptr, kBlue + 1},
        {"Pion",     "Plots/FeaturesLowP_Pions.root",     nullptr, kRed + 1},
        {"Kaon",     "Plots/FeaturesLowP_Kaons.root",     nullptr, kGreen + 2},
        {"Proton",   "Plots/FeaturesLowP_Protons.root",   nullptr, kMagenta + 1}
    };

    // Otwieranie wszystkich plików
    for (auto& p : particles) {
        p.file = TFile::Open(p.filePath.c_str(), "READ");
        if (!p.file || p.file->IsZombie()) {
            cerr << "[WARNING] Cannot open file: " << p.filePath << endl;
        }
    }

    // Znajdź pierwszy poprawnie otwarty plik jako wzorzec (referencję kluczy)
    TFile* f_ref = nullptr;
    string refParticleName = "";
    for (auto& p : particles) {
        if (p.file && !p.file->IsZombie()) {
            f_ref = p.file;
            refParticleName = p.name;
            break;
        }
    }

    if (!f_ref) {
        cerr << "[ERROR] None of the particle ROOT files could be opened!" << endl;
        return;
    }

    const string pdfName = "Plots/FeaturesAll_5Particle_Comparison_LowP.pdf";
    bool isFirstPage = true;

    TIter next(f_ref->GetListOfKeys());
    TKey* key = nullptr;

    while ((key = (TKey*)next())) {
        TClass* cl = gROOT->GetClass(key->GetClassName());
        
        // Skip objects other than TH1 and exclude TH2 plots (2D)
        if (!cl || !cl->InheritsFrom("TH1") || cl->InheritsFrom("TH2")) continue;

        string refKeyName = key->GetName();
        TString name = refKeyName;

        // Sprawdź, czy ten konkretny histogram ma mieć Rebin(2)
        bool applyRebin = false;
        for (const auto& pattern : hcalRebinHistos) {
            if (refKeyName.find(pattern) != string::npos) {
                applyRebin = true;
                break;
            }
        }

        // Wektor do trzymania przygotowanych par: (informacje o cząstce, znormalizowany histogram)
        vector<pair<ParticleInfo, TH1D*>> validHistos;
        double xMinSmart = 1e9;
        double xMaxSmart = -1e9;

        // Pobranie i przetworzenie histogramów dla każdej cząstki
        for (auto& p : particles) {
            if (!p.file || p.file->IsZombie()) continue;

            string hName = refKeyName;
            
            // Dynamiczna zamiana nazwy cząstki referencyjnej na bieżącą
            size_t pos = hName.find(refParticleName);
            if (pos != string::npos) {
                hName.replace(pos, refParticleName.length(), p.name);
            } else {
                size_t posPlural = hName.find(refParticleName + "s");
                if (posPlural != string::npos) {
                    hName.replace(posPlural, refParticleName.length() + 1, p.name + "s");
                }
            }

            TH1D* h_orig = (TH1D*)p.file->Get(hName.c_str());
            if (!h_orig) {
                h_orig = (TH1D*)p.file->Get(refKeyName.c_str());
            }

            if (!h_orig) continue;

            TH1D* h_clone = (TH1D*)h_orig->Clone(("h_" + p.name + "_" + refKeyName).c_str());

            // Wykonaj Rebin(2) jeśli wybrano ten histogram na liście
            if (applyRebin) {
                h_clone->Rebin(2);
            }

            // 1. Normalizacja do 1
            if (h_clone->Integral() > 0) h_clone->Scale(1.0 / h_clone->Integral());

            // 2. Automatyczne przycinanie ogonów
            auto [minC, maxC] = GetTailCuts(h_clone, 0.01, 0.96);
            xMinSmart = min(xMinSmart, minC);
            xMaxSmart = max(xMaxSmart, maxC);

            validHistos.push_back({p, h_clone});
        }

        if (validHistos.empty()) continue;

        // Zabezpieczenie zakresu
        if (xMinSmart >= xMaxSmart) {
            xMinSmart = validHistos[0].second->GetXaxis()->GetXmin();
            xMaxSmart = validHistos[0].second->GetXaxis()->GetXmax();
        }

        // Ustawienie zakresu X dla wszystkich i znalezienie globalnego maksimum Y
        double maxVal = 0.0;
        for (auto& item : validHistos) {
            TH1D* h = item.second;
            h->GetXaxis()->SetRangeUser(xMinSmart, xMaxSmart);
            int binMin = h->GetXaxis()->FindBin(xMinSmart);
            int binMax = h->GetXaxis()->FindBin(xMaxSmart);
            for (int b = binMin; b <= binMax; ++b) {
                maxVal = max(maxVal, h->GetBinContent(b));
            }
        }

        if (maxVal <= 0) maxVal = 1.0;

        // Create canvas
        TCanvas* c = new TCanvas(("c_" + refKeyName).c_str(), refKeyName.c_str(), 1000, 800);
        c->SetGrid();
        
        if (name.EndsWith("_h_SpreadR") || name.EndsWith("_h_R_Disp") || name.EndsWith("_h_R_DispWeighted") || name.EndsWith("_h_EnergyStdDev")) {
            maxVal *= 15.0; // Increase the maximum for log scale
            gPad->SetLogy(1);
        } else {
            gPad->SetLogy(0);
        }

        // Konfiguracja pierwszego histogramu (bazy)
        TH1D* h_first = validHistos[0].second;
        h_first->SetMinimum(gPad->GetLogy() ? 0.001 : 0.0);
        h_first->SetMaximum(maxVal * 1.30);

        string baseTitle = refKeyName;
        size_t pPos = baseTitle.find(refParticleName);
        if (pPos != string::npos) baseTitle.erase(pPos, refParticleName.length());
        size_t psPos = baseTitle.find(refParticleName + "s");
        if (psPos != string::npos) baseTitle.erase(psPos, refParticleName.length() + 1);
        
        h_first->SetTitle("");

        // Styling osi
        h_first->GetXaxis()->SetTitleSize(0.050);
        h_first->GetYaxis()->SetTitleSize(0.050);
        h_first->GetXaxis()->SetTitleOffset(1.15);
        h_first->GetYaxis()->SetTitleOffset(1.30);
        h_first->GetYaxis()->SetTitle("Normalized Counts");

        h_first->GetXaxis()->SetLabelSize(0.045);
        h_first->GetYaxis()->SetLabelSize(0.045);
        
        h_first->GetXaxis()->SetTitleFont(42);
        h_first->GetYaxis()->SetTitleFont(42);
        h_first->GetXaxis()->SetLabelFont(42);
        h_first->GetYaxis()->SetLabelFont(42);

        // Tworzenie legendy dopasowanej do liczby cząstek
        TLegend* leg = new TLegend(0.65, 0.70, 0.88, 0.90);
        leg->SetBorderSize(1);
        leg->SetFillColor(kWhite);
        leg->SetTextSize(0.035);
        leg->SetTextFont(42);

        // Rysowanie wszystkich histogramów w pętli
        bool firstDraw = true;
        for (auto& item : validHistos) {
            ParticleInfo pInfo = item.first;
            TH1D* h = item.second;

            h->SetLineColor(pInfo.color);
            h->SetLineWidth(3);

            if (firstDraw) {
                h->Draw("HIST");
                firstDraw = false;
            } else {
                h->Draw("HIST SAME");
            }

            leg->AddEntry(h, pInfo.name.c_str(), "l");
        }

        leg->Draw();
        DrawEPICLabel(0.15, 0.92, 0.040);
        c->SaveAs(("Plots/LowP/" + refKeyName + ".pdf").c_str());
        
        if (isFirstPage) {
            c->SaveAs((pdfName + "(").c_str());
            isFirstPage = false;
        } else {
            c->SaveAs(pdfName.c_str());
        }

        delete leg;
        for (auto& item : validHistos) {
            delete item.second;
        }
        delete c;
    }

    if (!isFirstPage) {
        TCanvas* cClose = new TCanvas("cClose", "", 1000, 800);
        cClose->SaveAs((pdfName + ")").c_str());
        delete cClose;
        cout << "[OK] Generated all TH1 histograms in: " << pdfName << endl;
    }

    // Zamknięcie i zwolnienie zasobów plików
    for (auto& p : particles) {
        if (p.file) {
            p.file->Close();
            delete p.file;
        }
    }
}