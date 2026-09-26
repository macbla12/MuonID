#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TMath.h>
#include <string>
#include <TLegend.h>
#include <vector>
#include <tuple>
#include <TProfile.h>
#include <TGraph.h>

// Brings in ToFSim(), ToFResults, computeExpectedTime(), field-map loading.
// (ToFSim.cxx must be in the same directory / include path.)
#include "ToFSim.cxx"

// =============================================================================
// Combine several (time, length) measurements belonging to the SAME track
// into one beta estimate via UNWEIGHTED least squares through the origin:
//
//   model:  t_i = (L_i / c) * u          with u = 1/beta
//
// No sigma_t is assumed anywhere -- every hit contributes equally. This is
// the simplest defensible combination once you don't trust a guessed timing
// resolution; if you later measure a real sigma_t (e.g. from the electron
// sample residuals), swap the weights back in.
//
//   u_hat    = sum(x_i * t_i) / sum(x_i^2)     x_i = L_i / c
//   beta_hat = 1 / u_hat
// =============================================================================
struct BetaEstimate { double beta; bool valid; };

BetaEstimate CombineBeta(const std::vector<std::pair<double,double>> &hits)
// hits: {time, length}
{
    double sumXX = 0.0, sumXT = 0.0;
    for (auto &h : hits) {
        double t = h.first;
        double L = h.second;
        double x = L / c_light;

        sumXX += x * x;
        sumXT += x * t;
    }

    if (sumXX <= 0.0) return {0.0, false};

    double u_hat = sumXT / sumXX;   // = 1/beta
    if (u_hat <= 0.0) return {0.0, false};

    return {1.0 / u_hat, true};
}

void IDAnalysis()
{
   //////////////////////
   //Setting up constants (PDG values -- do not alter)
   //////////////////////
   static double MuonMass     = 0.1056583;      // Expected value; temporary, do not change
   static double PionMass     = 0.13957039;      // Expected value; temporary, do not change
   static double ElectronMass = 0.00051099895;
   static double KaonMass     = 0.493677;
   static double ProtonMass   = 0.93827208;

   std::vector<double> masses = {MuonMass, ElectronMass, PionMass, KaonMass, ProtonMass};
   std::vector<int>    pdgs   = {13, 11, 211, 321, 2212};
   std::vector<double> masses2(masses.size());
   for (size_t i=0;i<masses.size();++i) masses2[i] = masses[i]*masses[i]; // m^2 lookup

   gROOT->SetBatch(kTRUE);
   gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

   double DEG = 180/TMath::Pi();

   TFile *PaperFile = new TFile("Plots/ToFPlots.root", "RECREATE");

   //////////////////////
   //Setting up histograms
   //////////////////////
   static constexpr int NumOfFiles = 5;

   static constexpr int nPbins = 25;
   double totalTracks[NumOfFiles][nPbins]   = {0};
   double foundTracks[NumOfFiles][nPbins]   = {0};
   double goodTracksP[NumOfFiles][nPbins]   = {0};
   double badTracksP[NumOfFiles][nPbins]    = {0};

   TH1D *BToFEtaRangeHist[NumOfFiles], *BToFPhiRangeHist[NumOfFiles];
   TH1D *EToFEtaRangeHist[NumOfFiles], *EToFPhiRangeHist[NumOfFiles];
   TH1D *ToFBTimeHist[NumOfFiles], *ToFETimeHist[NumOfFiles];
   TH1D *NumberOfBHits[NumOfFiles], *NumberOfEHits[NumOfFiles];

   TH1D *EffVsP[NumOfFiles];
   TH1D *PurityVsP[NumOfFiles];

   TH2D *BetaVsMom[NumOfFiles];

   // Primary discriminating observable: mass-squared, one entry per track,
   // combining however many hits that track left. Lets you see ALL five
   // species at once in one histogram, no hypothesis test / no threshold.
   TH1D *MassSqHist[NumOfFiles];
   TH2D *MassSqVsMom[NumOfFiles];

   // dR (angular) matching diagnostics, per species, barrel/endcap.
   TH1D *dEtaBarrel[NumOfFiles], *dPhiBarrel[NumOfFiles], *dRBarrel[NumOfFiles], *DistanceToFBarrel[NumOfFiles];
   TH1D *dEtaEndcap[NumOfFiles], *dPhiEndcap[NumOfFiles], *dREndcap[NumOfFiles], *DistanceToFEndcap[NumOfFiles];

   TH1D *NMatchedHitsBarrel[NumOfFiles], *NMatchedHitsEndcap[NumOfFiles];

   // Diagnostic: RK4 path length vs. straight-line chord length, so you can
   // see how much curvature actually matters for these momenta/species.
   TH1D *PathLenMinusChord[NumOfFiles];
   TH1D *ToFSimFailedFrac[NumOfFiles]; // 1 if DistanceCheck false (fallback used)

   std::vector<TString> files(NumOfFiles);

   //files.at(0)="/run/media/epic/Data/Muons/Grape-10x275/Paper/RECO/*.root";
   //files.at(1)="/run/media/epic/Data/Tau/reco/Energy_10x275/double_pi/recoDoublePi.root";

   files.at(0)="/run/media/epic/Data/Background/Muons/Continuous/reco_aMuonsLow*.root";
   files.at(1)="/run/media/epic/Data/Background/Pions/Continuous/reco_piPlus*.root";
   files.at(2)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Electrons.root";
   files.at(3)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Kaons.root";
   files.at(4)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Protons.root";

   // Tune after looking at the dR plots from a first pass.
   double dR_cut_barrel = 0.8;
   double dR_cut_endcap = 0.8;
   double dist_cut_barrel = 6;
   double dist_cut_endcap = 6;



   for(int File=0; File<NumOfFiles;File++)
   {
      std::string name;
      if(File==0) name="Muons";
      if(File==1) name="Pions";
      if(File==2) name="Electrons";
      if(File==3) name="Kaons";
      if(File==4) name="Protons";

      TChain *mychain = new TChain("events");
      mychain->Add(files.at(File));

      TTreeReader tree_reader(mychain);

      TTreeReaderArray<float> trackMomX(tree_reader, "ReconstructedChargedParticles.momentum.x");
      TTreeReaderArray<float> trackMomY(tree_reader, "ReconstructedChargedParticles.momentum.y");
      TTreeReaderArray<float> trackMomZ(tree_reader, "ReconstructedChargedParticles.momentum.z");
      TTreeReaderArray<float> trackEng(tree_reader, "ReconstructedChargedParticles.energy");
      TTreeReaderArray<float> trackCharge(tree_reader, "ReconstructedChargedParticles.charge");

      TTreeReaderArray<float> BToFPosX(tree_reader, "TOFBarrelRecHits.position.x");
      TTreeReaderArray<float> BToFPosY(tree_reader, "TOFBarrelRecHits.position.y");
      TTreeReaderArray<float> BToFPosZ(tree_reader, "TOFBarrelRecHits.position.z");
      TTreeReaderArray<float> BToFTime(tree_reader, "TOFBarrelRecHits.time");

      TTreeReaderArray<float> EToFPosX(tree_reader, "TOFEndcapRecHits.position.x");
      TTreeReaderArray<float> EToFPosY(tree_reader, "TOFEndcapRecHits.position.y");
      TTreeReaderArray<float> EToFPosZ(tree_reader, "TOFEndcapRecHits.position.z");
      TTreeReaderArray<float> EToFTime(tree_reader, "TOFEndcapRecHits.time");

      //==================================//
      BToFEtaRangeHist[File]= new TH1D(Form("BToFEtaRangeHist%s",name.c_str()),
                                       Form("BToFEtaRangeHist%s",name.c_str()),
                                       50,-2.5,2.5);
      BToFPhiRangeHist[File]= new TH1D(Form("BToFPhiRangeHist%s",name.c_str()),
                                       Form("BToFPhiRangeHist%s",name.c_str()),
                                       50,-180,180);

      EToFEtaRangeHist[File]= new TH1D(Form("EToFEtaRangeHist%s",name.c_str()),
                                       Form("EToFEtaRangeHist%s",name.c_str()),
                                       50,1.5,3.5);
      EToFPhiRangeHist[File]= new TH1D(Form("EToFPhiRangeHist%s",name.c_str()),
                                       Form("EToFPhiRangeHist%s",name.c_str()),
                                       50,-180,180);

      ToFBTimeHist[File]= new TH1D(Form("ToFBarrelTimeHist%s",name.c_str()),
                                   Form("ToFBarrelTimeHist%s",name.c_str()),
                                   50,0,15);
      ToFETimeHist[File]= new TH1D(Form("ToFEndcapTimeHist%s",name.c_str()),
                                   Form("ToFEndcapTimeHist%s",name.c_str()),
                                   50,0,15);

      NumberOfBHits[File]= new TH1D(Form("NumberOfBarrelHits%s",name.c_str()),
                                    Form("NumberOfBarrelHits%s",name.c_str()),
                                    10,-0.5,9.5);
      NumberOfEHits[File]= new TH1D(Form("NumberOfEndcapHits%s",name.c_str()),
                                    Form("NumberOfEndcapHits%s",name.c_str()),
                                    10,-0.5,9.5);

      EffVsP[File]    = new TH1D(Form("EffVsP_%s",name.c_str()),
                                 Form("Efficiency vs P %s",name.c_str()),
                                 nPbins,0.0,2.5);
      PurityVsP[File] = new TH1D(Form("PurityVsP_%s",name.c_str()),
                                 Form("Purity vs P %s",name.c_str()),
                                 nPbins,0.0,2.5);

      BetaVsMom[File] = new TH2D(Form("BetaVsMom_%s",name.c_str()),
                                 Form("BetaVsMom_%s (combined per-track);p [GeV];#beta",name.c_str()),
                                 100,0,2,100,0.7,1.1);

      // m^2 in GeV^2: mu^2~0.0112, pi^2~0.0195, K^2~0.244, p^2~0.880
      MassSqHist[File]  = new TH1D(Form("MassHist_%s",name.c_str()),
                                   Form("Mass (combined per-track) %s;m [GeV];Counts",name.c_str()),
                                   200,-0.05,0.35);
      MassSqVsMom[File] = new TH2D(Form("MassSqVsMom_%s",name.c_str()),
                                   Form("MassSqVsMom_%s;p [GeV];m^{2} [GeV^{2}]",name.c_str()),
                                   100,0,2,150,-0.1,1.1);

      dEtaBarrel[File] = new TH1D(Form("dEtaBarrel_%s",name.c_str()),
                                  Form("dEta (track-hit) Barrel %s;#Delta#eta;Counts",name.c_str()),
                                  200,-1.0,1.0);
      dPhiBarrel[File] = new TH1D(Form("dPhiBarrel_%s",name.c_str()),
                                  Form("dPhi (track-hit) Barrel %s;#Delta#phi [rad];Counts",name.c_str()),
                                  200,-1.0,1.0);
      dRBarrel[File]   = new TH1D(Form("dRBarrel_%s",name.c_str()),
                                  Form("dR (track-hit) Barrel %s;#DeltaR;Counts",name.c_str()),
                                  200,0.0,2.0);

      DistanceToFBarrel[File]= new TH1D(Form("DistanceToFBarrel_%s",name.c_str()),
                                  Form("DistanceToFBarrel (track-hit) Barrel %s;DistanceToFBarrel [mm];Counts",name.c_str()),
                                  200,0.0,100.0);

      dEtaEndcap[File] = new TH1D(Form("dEtaEndcap_%s",name.c_str()),
                                  Form("dEta (track-hit) Endcap %s;#Delta#eta;Counts",name.c_str()),
                                  200,-1.0,1.0);
      dPhiEndcap[File] = new TH1D(Form("dPhiEndcap_%s",name.c_str()),
                                  Form("dPhi (track-hit) Endcap %s;#Delta#phi [rad];Counts",name.c_str()),
                                  200,-1.0,1.0);
      dREndcap[File]   = new TH1D(Form("dREndcap_%s",name.c_str()),
                                  Form("dR (track-hit) Endcap %s;#DeltaR;Counts",name.c_str()),
                                  200,0.0,2.0);
      DistanceToFEndcap[File]= new TH1D(Form("DistanceToFEndcap_%s",name.c_str()),
                                  Form("DistanceToFEndcap (track-hit) Endcap %s;DistanceToFEndcap [mm];Counts",name.c_str()),
                                  200,0.0,100.0);

      NMatchedHitsBarrel[File] = new TH1D(Form("NMatchedHitsBarrel_%s",name.c_str()),
                                          Form("Matched hits per track (Barrel) %s;N hits;Tracks",name.c_str()),
                                          10,-0.5,9.5);
      NMatchedHitsEndcap[File] = new TH1D(Form("NMatchedHitsEndcap_%s",name.c_str()),
                                          Form("Matched hits per track (Endcap) %s;N hits;Tracks",name.c_str()),
                                          10,-0.5,9.5);

      PathLenMinusChord[File] = new TH1D(Form("PathLenMinusChord_%s",name.c_str()),
                                         Form("RK4 path length minus straight-line chord %s;#Delta L [mm];Counts",name.c_str()),
                                         200,-5.0,50.0);

      ToFSimFailedFrac[File] = new TH1D(Form("ToFSimFailedFrac_%s",name.c_str()),
                                        Form("ToFSim DistanceCheck failed (1) vs ok (0) %s",name.c_str()),
                                        2,-0.5,1.5);

      int eventID      = 0;
      double FoundParticles = 0;
      double particscount   = 0;
      double goodcount      = 0;
      double badcount       = 0;

      while(tree_reader.Next()){
         eventID++;
         if(eventID>200000) break;

         if(eventID%20000==0) std::cout<<"File "<<name<<" and event number... "<<eventID<<std::endl;

         double FilePDG;
         if(File==0) FilePDG=13;
         if(File==1) FilePDG=211;
         if(File==2) FilePDG=11;
         if(File==3) FilePDG=321;
         if(File==4) FilePDG=2212;

         for(int particle=0; particle<trackEng.GetSize(); particle++)
         {
            TLorentzVector Partic;
            Partic.SetPxPyPzE(trackMomX[particle],
                              trackMomY[particle],
                              trackMomZ[particle],
                              trackEng[particle]);

            if(Partic.P()>1) continue;

            particscount++;

            double p = Partic.P();
            int pbin = int(p / 0.1);
            if (pbin >= 0 && pbin < nPbins) {
               totalTracks[File][pbin]++;
            }

            double trackEta = Partic.Eta();
            double trackPhi = Partic.Phi();
            int charge = (int)trackCharge[particle];

            // {time, length} per matched hit -- length now comes from the
            // RK4 field propagation (ToFSim), falling back to the straight
            // chord (ToFPos.Mag()) only if ToFSim couldn't confirm a close
            // approach (DistanceCheck==false), so a bad propagation never
            // silently corrupts the beta fit.
            std::vector<std::pair<double,double>> matchedBarrel;
            std::vector<std::pair<double,double>> matchedEndcap;

            // BARREL: dR scan over every hit in the event
            for(int hits=0; hits<BToFTime.GetSize(); hits++)
            {
               TVector3 ToFPos(BToFPosX[hits],BToFPosY[hits],BToFPosZ[hits]);

               BToFEtaRangeHist[File]->Fill(ToFPos.Eta());
               BToFPhiRangeHist[File]->Fill(ToFPos.Phi()*DEG);
               ToFBTimeHist[File]->Fill(BToFTime[hits]);

               double dEta = trackEta - ToFPos.Eta();
               double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
               double dR   = sqrt(dEta*dEta + dPhi*dPhi);

               dEtaBarrel[File]->Fill(dEta);
               dPhiBarrel[File]->Fill(dPhi);
               dRBarrel[File]->Fill(dR);

               if (dR < dR_cut_barrel && charge * dPhi < 0) {
                  ToFResults tof = ToFSim(Partic, charge, ToFPos);

                  double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                  ToFSimFailedFrac[File]->Fill(tof.DistanceCheck ? 0 : 1);
                  DistanceToFBarrel[File]->Fill(tof.distance_to_TOF);
                  if (tof.distance_to_TOF <dist_cut_barrel) {
                     if (tof.DistanceCheck) {
                        PathLenMinusChord[File]->Fill(tof.track_length - ToFPos.Mag());
                     }
                     matchedBarrel.push_back({BToFTime[hits], length});
                  }
               }
            }
            NumberOfBHits[File]->Fill(matchedBarrel.size());
            NMatchedHitsBarrel[File]->Fill((double)matchedBarrel.size());

            // ENDCAP: dR scan over every hit in the event
            for(int hits=0; hits<EToFTime.GetSize(); hits++)
            {
               TVector3 ToFPos(EToFPosX[hits],EToFPosY[hits],EToFPosZ[hits]);

               EToFEtaRangeHist[File]->Fill(ToFPos.Eta());
               EToFPhiRangeHist[File]->Fill(ToFPos.Phi()*DEG);
               ToFETimeHist[File]->Fill(EToFTime[hits]);

               double dEta = trackEta - ToFPos.Eta();
               double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
               double dR   = sqrt(dEta*dEta + dPhi*dPhi);

               dEtaEndcap[File]->Fill(dEta);
               dPhiEndcap[File]->Fill(dPhi);
               dREndcap[File]->Fill(dR);

               if (dR < dR_cut_endcap  && charge * dPhi < 0) {
                  ToFResults tof = ToFSim(Partic, charge, ToFPos);

                  double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                  ToFSimFailedFrac[File]->Fill(tof.DistanceCheck ? 0 : 1);
                  DistanceToFEndcap[File]->Fill(tof.distance_to_TOF);
                  if (tof.distance_to_TOF <dist_cut_barrel) {
                     if (tof.DistanceCheck) {
                        PathLenMinusChord[File]->Fill(tof.track_length - ToFPos.Mag());
                     }
                     matchedEndcap.push_back({EToFTime[hits], length});
                  }
               }
            }
            NumberOfEHits[File]->Fill(matchedEndcap.size());
            NMatchedHitsEndcap[File]->Fill((double)matchedEndcap.size());

            bool found = !matchedBarrel.empty() || !matchedEndcap.empty();
            if (found) FoundParticles++;
            if (!found) continue;

            // ---- one combined set of hits for this track ----------------
            std::vector<std::pair<double,double>> allMatched;
            allMatched.insert(allMatched.end(), matchedBarrel.begin(), matchedBarrel.end());
            allMatched.insert(allMatched.end(), matchedEndcap.begin(), matchedEndcap.end());

            // ---- combined beta, no sigma anywhere ------------------------
            BetaEstimate be = CombineBeta(allMatched);
            if (!be.valid) continue;

            BetaVsMom[File]->Fill(p, be.beta);

            // m^2 = p^2 (1/beta^2 - 1); can go negative for beta>=1 due to
            // resolution -- that's fine, it's still a usable continuous
            // observable, just plot it as-is.
            double msq = p*p * (1.0/(be.beta*be.beta) - 1.0);
            MassSqHist[File]->Fill(sqrt(msq));
            MassSqVsMom[File]->Fill(p, msq);

            // ---- classification: nearest true m^2, no hypothesis test,
            //      no threshold, works the same way for all 5 species -----
            int bestPDG = 0;
            double bestDiff = 1e18;
            for (size_t i=0;i<pdgs.size();++i) {
               double diff = std::fabs(msq - masses2[i]);
               if (diff < bestDiff) { bestDiff = diff; bestPDG = pdgs[i]; }
            }

            if (pbin >= 0 && pbin < nPbins) {
               foundTracks[File][pbin]++;
               if (bestPDG==FilePDG) goodTracksP[File][pbin]++;
               else                  badTracksP[File][pbin]++;
            }

            if (bestPDG==FilePDG) goodcount++;
            else                  badcount++;
         }
      }

      std::cout<<"==========================="<<std::endl;
      std::cout<<"End of "<< name << " file"<<std::endl;
      std::cout<<"Number of events: "<<eventID<<std::endl;
      std::cout<<"Found particles: "<<FoundParticles<<"   All particles: "<<particscount<<std::endl;
      std::cout<<"Found Ratio: "<<FoundParticles*100/particscount<<'%'<<std::endl;
      std::cout<<"Purity: "<<goodcount*100/(goodcount+badcount)<<'%'<<std::endl;
      std::cout<<"==========================="<<std::endl;

      for (int ib = 0; ib < nPbins; ++ib) {
         double eff = (totalTracks[File][ib] > 0)
                      ? foundTracks[File][ib] / totalTracks[File][ib]
                      : 0.0;
         double purity = (goodTracksP[File][ib] + badTracksP[File][ib] > 0)
                         ? goodTracksP[File][ib] /
                           (goodTracksP[File][ib] + badTracksP[File][ib])
                         : 0.0;

         EffVsP[File]->SetBinContent(ib+1, eff);
         PurityVsP[File]->SetBinContent(ib+1, purity);
      }

      BToFEtaRangeHist[File]->Write();
      BToFPhiRangeHist[File]->Write();
      EToFEtaRangeHist[File]->Write();
      EToFPhiRangeHist[File]->Write();
      ToFBTimeHist[File]->Write();
      ToFETimeHist[File]->Write();
      NumberOfBHits[File]->Write();
      NumberOfEHits[File]->Write();
      EffVsP[File]->Write();
      PurityVsP[File]->Write();
      BetaVsMom[File]->Write();
      MassSqHist[File]->Write();
      MassSqVsMom[File]->Write();

      dEtaBarrel[File]->Write();
      dPhiBarrel[File]->Write();
      dRBarrel[File]->Write();
      DistanceToFBarrel[File]->Write();

      dEtaEndcap[File]->Write();
      dPhiEndcap[File]->Write();
      dREndcap[File]->Write();
      DistanceToFEndcap[File]->Write();

      NMatchedHitsBarrel[File]->Write();
      NMatchedHitsEndcap[File]->Write();

      PathLenMinusChord[File]->Write();
      ToFSimFailedFrac[File]->Write();
   }

   // Theoretical beta(p) curves for five masses (still useful as a reference
   // on BetaVsMom, even though classification no longer tests hypotheses directly).
   const int nCurvePoints = 200;
   TGraph *betaMuon   = new TGraph(nCurvePoints);
   TGraph *betaElec   = new TGraph(nCurvePoints);
   TGraph *betaPion   = new TGraph(nCurvePoints);
   TGraph *betaKaon   = new TGraph(nCurvePoints);
   TGraph *betaProton = new TGraph(nCurvePoints);

   for (int i = 0; i < nCurvePoints; ++i) {
      double p = 5.0 * i / (nCurvePoints - 1);
      auto beta = [&](double m) {
         double E = std::sqrt(p*p + m*m);
         return p / E;
      };
      betaMuon  ->SetPoint(i, p, beta(MuonMass));
      betaElec  ->SetPoint(i, p, beta(ElectronMass));
      betaPion  ->SetPoint(i, p, beta(PionMass));
      betaKaon  ->SetPoint(i, p, beta(KaonMass));
      betaProton->SetPoint(i, p, beta(ProtonMass));
   }

   betaMuon  ->SetName("BetaCurve_Muon");
   betaElec  ->SetName("BetaCurve_Electron");
   betaPion  ->SetName("BetaCurve_Pion");
   betaKaon  ->SetName("BetaCurve_Kaon");
   betaProton->SetName("BetaCurve_Proton");

   betaMuon  ->Write();
   betaElec  ->Write();
   betaPion  ->Write();
   betaKaon  ->Write();
   betaProton->Write();

   PaperFile->Close();
}