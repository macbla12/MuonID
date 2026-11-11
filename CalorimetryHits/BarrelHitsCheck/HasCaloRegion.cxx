// hasCalo_region.cxx
// Execute in ROOT:
//   root -l -q 'hasCalo_region.cxx()'

#include <glob.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "podio/ROOTReader.h"
#include "podio/Frame.h"
#include "edm4hep/MCParticleCollection.h"
#include "edm4eic/ClusterCollection.h"
#include "edm4eic/MCRecoParticleAssociationCollection.h"
#include "edm4eic/MCRecoClusterParticleAssociationCollection.h"

#include <TFile.h>
#include <TCanvas.h>
#include <TEfficiency.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TVector3.h>
#include <TMath.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TPad.h>

static const double T_CUT = 20.0;
static const double ETA_LO = -1.0;
static const double ETA_HI = 1.0;

static bool InEta(double eta) {
    return (eta >= ETA_LO && eta < ETA_HI);
}

struct Calo { double E = 0; int n = 0, ncl = 0, nAll = 0, nNoT = 0; };

// Collect hits from associated clusters taking time cut T_CUT into account
Calo Collect(const podio::Frame& ev, const std::vector<std::string>& names,
             const edm4hep::MCParticle& sim, TH1F* hT) {
    Calo c;
    for (const auto& nm : names) {
        std::string assocName = nm.substr(0, nm.size() - 1) + "Associations";

        auto& as = ev.get<edm4eic::MCRecoClusterParticleAssociationCollection>(assocName);
        for (auto a : as) {
            if (a.getSim() != sim) continue;
            c.ncl++;
            for (auto h : a.getRec().getHits()) {
                c.nNoT++; // hits without time cut
                hT->Fill(h.getTime());
                if (h.getTime() <= T_CUT) { 
                    c.n++; 
                    c.E += h.getEnergy(); 
                }
            }
        }
        c.nAll += ev.get<edm4eic::ClusterCollection>(nm).size();
    }
    return c;
}

void HasCaloRegion() {
    gStyle->SetOptStat(0); // Clean style without stat box for overlays

    // Calorimeter collections for eta range [-1.0, 3.5]
    std::vector<std::string> ECAL = {"EcalBarrelImagingClusters", "EcalBarrelScFiClusters", "EcalEndcapPClusters"};
    std::vector<std::string> HCAL = {"HcalBarrelClusters", "LFHCALClusters"};

    glob_t g; 
    glob("/run/media/epic/Data/Background/Muons/Continuous/reco_*.root", GLOB_TILDE, nullptr, &g);

    std::vector<std::string> files(g.gl_pathv, g.gl_pathv + g.gl_pathc); 
    globfree(&g);
    
    if (files.empty()) { 
        std::cerr << "Error: No input files found!\n"; 
        return; 
    }

    podio::ROOTReader reader; 
    reader.openFiles(files);
    const size_t nEv = std::min<size_t>(reader.getEntries("events"), 10000);

    // ---- TEfficiency & Histograms ----
    // Conditions: 
    // 0: track, 1: hasCalo (n>0), 2: hasECal, 3: hasHCal, 
    // 4: hasCalo_ncl (ncl>0), 5: hasCalo_noTcut (nNoT>0), 6: hasCalo_nAll (nAll>0)
    const char* cond[] = {"track", "hasCalo", "hasECal", "hasHCal", "hasCalo_ncl", "hasCalo_noTcut", "hasCalo_nAll"};
    const char* var[]  = {"pt", "eta", "phi", "p"};
    const char* ttl[]  = {"p_{T} [GeV]", "#eta", "#phi", "p [GeV]"};
    const int nb[]     = {40, 45, 36, 40};
    const double lo[]  = {0, ETA_LO, -TMath::Pi(), 0};
    const double hi[]  = {4, ETA_HI, TMath::Pi(), 4};

    TEfficiency* eff[7][4];
    TH1F* hVarAll[4];     // Kinematic distributions for ALL reconstructed tracks
    TH1F* hVarHasCalo[4]; // Kinematic distributions for tracks WITH calorimeter hit

    for (int v = 0; v < 4; v++) {
        hVarAll[v]     = new TH1F(Form("h_%s_all", var[v]), Form(";%s;Events / Tracks", ttl[v]), nb[v], lo[v], hi[v]);
        hVarHasCalo[v] = new TH1F(Form("h_%s_hasCalo", var[v]), Form(";%s;Events / Tracks", ttl[v]), nb[v], lo[v], hi[v]);
        
        for (int c = 0; c < 7; c++) {
            eff[c][v] = new TEfficiency(Form("eff_%s_%s", cond[c], var[v]),
                                        Form("%s;%s;Efficiency", cond[c], ttl[v]), nb[v], lo[v], hi[v]);
        }
    }

    // Time distribution histograms
    auto* hTe = new TH1F("hT_ecal", ";t_{hit} ECal [ns];Hits", 200, -50, 200);
    auto* hTh = new TH1F("hT_hcal", ";t_{hit} HCal [ns];Hits", 200, -50, 200);

    // 2D Histograms for Page 5 (|eta| < 3.5)
    auto* h2_pt_eta = new TH2F("h2_pt_eta", "p_{T} vs #eta;#eta;p_{T} [GeV]", 70, -3.5, 3.5, 40, 0, 4);
    auto* h2_p_eta  = new TH2F("h2_p_eta",  "p vs #eta;#eta;p [GeV]",         70, -3.5, 3.5, 40, 0, 4);

    // Event Loop
    for (size_t i = 0; i < nEv; i++) {
        const auto ev = podio::Frame(reader.readNextEntry("events"));
        const auto& mcs   = ev.get<edm4hep::MCParticleCollection>("MCParticles");
        const auto& assoc = ev.get<edm4eic::MCRecoParticleAssociationCollection>("ReconstructedChargedParticleAssociations");

        auto kin = [](auto m, double& pt, double& p, double& eta, double& phi) {
            TVector3 v(m.x, m.y, m.z); 
            pt = v.Pt(); 
            p = v.Mag(); 
            eta = v.Eta(); 
            phi = v.Phi(); 
        };

        auto fillEff = [&](int c, bool pass, double pt, double eta, double phi, double p) {
            double x[4] = {pt, eta, phi, p};
            for (int v = 0; v < 4; v++) eff[c][v]->Fill(pass, x[v]);
        };

        // (1) Track Reconstruction Efficiency
        for (auto mcp : mcs) {
            double pt, p, eta, phi; 
            kin(mcp.getMomentum(), pt, p, eta, phi);
            if (!InEta(eta)) continue;

            bool found = false;
            for (auto a : assoc) {
                if (a.getSim() == mcp) { 
                    found = true; 
                    break; 
                }
            }
            fillEff(0, found, pt, eta, phi, p);
        }

        // (2) Calorimeter Matching for Reconstructed Tracks
        for (auto a : assoc) {
            auto sim = a.getSim(); 

            double pt, p, eta, phi; 
            kin(sim.getMomentum(), pt, p, eta, phi);
            
            // Fill 2D histograms for broader eta range
            if (std::abs(eta) <= 3.5) {
                h2_pt_eta->Fill(eta, pt);
                h2_p_eta->Fill(eta, p);
            }

            if (!InEta(eta)) continue;

            double x[4] = {pt, eta, phi, p};
            for (int v = 0; v < 4; v++) hVarAll[v]->Fill(x[v]);

            Calo e = Collect(ev, ECAL, sim, hTe);
            Calo h = Collect(ev, HCAL, sim, hTh);

            bool hasCalo      = (e.n > 0 || h.n > 0);
            bool hasCalo_ncl  = (e.ncl > 0 || h.ncl > 0);
            bool hasCalo_noT  = (e.nNoT > 0 || h.nNoT > 0);
            bool hasCalo_nAll = (e.nAll > 0 || h.nAll > 0);

            if (hasCalo) {
                for (int v = 0; v < 4; v++) hVarHasCalo[v]->Fill(x[v]);
            }

            fillEff(1, hasCalo, pt, eta, phi, p);
            fillEff(2, e.n > 0, pt, eta, phi, p);
            fillEff(3, h.n > 0, pt, eta, phi, p);
            fillEff(4, hasCalo_ncl, pt, eta, phi, p);
            fillEff(5, hasCalo_noT, pt, eta, phi, p);
            fillEff(6, hasCalo_nAll, pt, eta, phi, p);
        }
    }

    // ---- Save to ROOT file ----
    TFile out("hasCalo_results.root", "RECREATE");
    for (int c = 0; c < 7; c++) {
        for (int v = 0; v < 4; v++) eff[c][v]->Write();
    }
    for (int v = 0; v < 4; v++) {
        hVarAll[v]->Write();
        hVarHasCalo[v]->Write();
    }
    h2_pt_eta->Write();
    h2_p_eta->Write();
    hTe->Write();
    hTh->Write();

    // ---- PDF Generation ----
    const char* pdfName = "hasCalo_summary.pdf";
    TCanvas cPdf("cPdf", "Summary", 1200, 600);

    cPdf.Print(Form("%s[", pdfName)); // Open PDF

    // Pages 1-4: Left -> Overlaid Distributions (All vs hasCalo), Right -> hasCalo Efficiency
    for (int v = 0; v < 4; v++) {
        cPdf.Clear();
        cPdf.Divide(2, 1);

        // Left Panel: Overlaid Kinematic Distributions
        cPdf.cd(1);
        gPad->SetRightMargin(0.10);

        hVarAll[v]->SetLineColor(kBlue);
        hVarAll[v]->SetLineWidth(2);
        
        hVarHasCalo[v]->SetLineColor(kRed + 1);
        hVarHasCalo[v]->SetFillColorAlpha(kRed + 1, 0.35);
        hVarHasCalo[v]->SetLineWidth(2);

        double maxVal = std::max(hVarAll[v]->GetMaximum(), hVarHasCalo[v]->GetMaximum());
        hVarAll[v]->SetMaximum(maxVal * 1.25);
        hVarAll[v]->SetMinimum(0.0);

        hVarAll[v]->Draw("HIST");
        hVarHasCalo[v]->Draw("HIST SAME");

        TLegend* leg = new TLegend(0.60, 0.75, 0.88, 0.88);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->AddEntry(hVarAll[v], "All tracks", "l");
        leg->AddEntry(hVarHasCalo[v], "hasCalo tracks", "f");
        leg->Draw();

        // Right Panel: Efficiency Curve
        cPdf.cd(2);
        gPad->SetRightMargin(0.10);
        eff[1][v]->SetLineColor(kRed + 1);
        eff[1][v]->SetMarkerColor(kRed + 1);
        eff[1][v]->SetMarkerStyle(20);
        eff[1][v]->Draw("AP");

        cPdf.Print(pdfName);
    }

    // Page 5: 2D Distributions (pT vs eta, p vs eta) for |eta| < 3.5
    cPdf.Clear();
    cPdf.Divide(2, 1);

    cPdf.cd(1);
    gPad->SetRightMargin(0.15);
    h2_pt_eta->Draw("COLZ");

    cPdf.cd(2);
    gPad->SetRightMargin(0.15);
    h2_p_eta->Draw("COLZ");

    cPdf.Print(pdfName);

    // Page 6: pT Efficiency for hasECal and hasHCal
    cPdf.Clear();
    cPdf.Divide(2, 1);

    cPdf.cd(1);
    gPad->SetRightMargin(0.10);
    eff[2][0]->SetLineColor(kGreen + 2);
    eff[2][0]->SetMarkerColor(kGreen + 2);
    eff[2][0]->SetMarkerStyle(21);
    eff[2][0]->Draw("AP");

    cPdf.cd(2);
    gPad->SetRightMargin(0.10);
    eff[3][0]->SetLineColor(kMagenta + 1);
    eff[3][0]->SetMarkerColor(kMagenta + 1);
    eff[3][0]->SetMarkerStyle(22);
    eff[3][0]->Draw("AP");

    cPdf.Print(pdfName);

    // Page 7: p Efficiency for hasECal and hasHCal
    cPdf.Clear();
    cPdf.Divide(2, 1);

    cPdf.cd(1);
    gPad->SetRightMargin(0.10);
    eff[2][3]->SetLineColor(kGreen + 2);
    eff[2][3]->SetMarkerColor(kGreen + 2);
    eff[2][3]->SetMarkerStyle(21);
    eff[2][3]->Draw("AP");

    cPdf.cd(2);
    gPad->SetRightMargin(0.10);
    eff[3][3]->SetLineColor(kMagenta + 1);
    eff[3][3]->SetMarkerColor(kMagenta + 1);
    eff[3][3]->SetMarkerStyle(22);
    eff[3][3]->Draw("AP");

    cPdf.Print(pdfName);

    // Page 8: Diagnostic Comparison (nAll vs ncl vs nNoT vs n) for pT and p
    cPdf.Clear();
    cPdf.Divide(2, 1);

    int diagConds[] = {6, 4, 5, 1}; // nAll, ncl, nNoT, n (hasCalo)
    int diagColors[] = {kBlack, kBlue, kOrange + 7, kRed + 1};
    int diagStyles[] = {24, 25, 26, 20};
    const char* diagLabels[] = {"nAll > 0 (All clusters)", "ncl > 0 (Assoc. clusters)", "nNoT > 0 (Hits, no t-cut)", "n > 0 (Hits, t <= 20ns)"};

    // Panel 1: Diagnostic vs pT
    cPdf.cd(1);
    gPad->SetRightMargin(0.10);
    TLegend* legDiag1 = new TLegend(0.45, 0.15, 0.88, 0.38);
    legDiag1->SetBorderSize(0);
    legDiag1->SetFillStyle(0);

    for (int k = 0; k < 4; k++) {
        int cIdx = diagConds[k];
        eff[cIdx][0]->SetLineColor(diagColors[k]);
        eff[cIdx][0]->SetMarkerColor(diagColors[k]);
        eff[cIdx][0]->SetMarkerStyle(diagStyles[k]);
        eff[cIdx][0]->Draw(k == 0 ? "AP" : "P SAME");
        legDiag1->AddEntry(eff[cIdx][0], diagLabels[k], "lp");
    }
    legDiag1->Draw();

    // Panel 2: Diagnostic vs p
    cPdf.cd(2);
    gPad->SetRightMargin(0.10);
    TLegend* legDiag2 = new TLegend(0.45, 0.15, 0.88, 0.38);
    legDiag2->SetBorderSize(0);
    legDiag2->SetFillStyle(0);

    for (int k = 0; k < 4; k++) {
        int cIdx = diagConds[k];
        eff[cIdx][3]->SetLineColor(diagColors[k]);
        eff[cIdx][3]->SetMarkerColor(diagColors[k]);
        eff[cIdx][3]->SetMarkerStyle(diagStyles[k]);
        eff[cIdx][3]->Draw(k == 0 ? "AP" : "P SAME");
        legDiag2->AddEntry(eff[cIdx][3], diagLabels[k], "lp");
    }
    legDiag2->Draw();

    cPdf.Print(pdfName);

    cPdf.Print(Form("%s]", pdfName)); // Close PDF

    out.Close();
    std::cout << "Successfully finished. Results saved to hasCalo_summary.pdf and hasCalo_results.root\n";
}