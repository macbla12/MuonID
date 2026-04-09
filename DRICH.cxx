
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TMath.h>
#include <string>
#include <TLegend.h>
#include <vector>


void IDAnalysis()
{

   gROOT->SetBatch(kTRUE);
   gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");
   //gStyle->SetOptStat(0);

   double DEG=180/TMath::Pi();



   static constexpr int NumOfFiles=3;
   TH1D *ThetaPhotonAe[NumOfFiles],*ThetaPhotonGa[NumOfFiles];
   TH2D *ThetaPhotonMomAe[NumOfFiles],*ThetaPhotonMomGe[NumOfFiles];
   TH1D *PDG[NumOfFiles],*NumberParticles[NumOfFiles];
   TH1D *AllParticTheta[NumOfFiles], *AllParticPhi[NumOfFiles], *AllParticEnergy[NumOfFiles];
   TH1D *NotFoundParticTheta[NumOfFiles], *NotFoundParticPhi[NumOfFiles], *NotFoundParticEnergy[NumOfFiles];
   
   for(int i=0;i<NumOfFiles;i++)
   {
  
      //==================================//
      ThetaPhotonAe[i]= new TH1D(Form("ThetaPhotonAe%d",i+1),Form("ThetaPhotonAe%d",i+1),100,0,250);
      ThetaPhotonMomAe[i]= new TH2D(Form("ThetaPhotonMomAe%d",i+1),Form("ThetaPhotonMomAe%d",i+1),100,0,250,100,0,10);
      ThetaPhotonGa[i]= new TH1D(Form("ThetaPhotonGa%d",i+1),Form("ThetaPhotonGa%d",i+1),100,0,250);
      ThetaPhotonMomGa[i]= new TH2D(Form("ThetaPhotonMomGa%d",i+1),Form("ThetaPhotonMomGa%d",i+1),100,0,250,100,0,10);
      //==================================//
      PDG[i]= new TH1D(Form("PDG%d",i+1),Form("PDG%d",i+1),41,-230.5,230);
      NumberParticles[i]= new TH1D(Form("NumberParticles%d",i+1),Form("NumberParticles%d",i+1),9,-0.5,8.5);
      //==================================//
      AllParticTheta[i] = new TH1D(Form("AllParticTheta%d",i+1),Form("AllParticTheta%d",i+1),50,0,180);
      AllParticPhi[i]= new TH1D(Form("AllParticTheta%d",i+1),Form("AllParticTheta%d",i+1),30,-180,180);
      AllParticEnergy[i]= new TH1D(Form("AllParticTheta%d",i+1),Form("AllParticTheta%d",i+1),50,0,10);
      
      NotFoundParticTheta[i] = new TH1D(Form("NotFoundParticTheta%d",i+1),Form("NotFoundParticTheta%d",i+1),50,0,180);
      NotFoundParticPhi[i]= new TH1D(Form("NotFoundParticTheta%d",i+1),Form("NotFoundParticTheta%d",i+1),30,-180,180);
      NotFoundParticEnergy[i]= new TH1D(Form("NotFoundParticTheta%d",i+1),Form("NotFoundParticTheta%d",i+1),50,0,10);
   }
   
   vector<TString> files(NumOfFiles);
   
   files.at(0)="/Data/Muons/ToF/Electrons.root";
   files.at(1)="/Data/Muons/Grape-10x275/recoGL.root";
   files.at(2)="/Data/Tau/reco/double_pi/recoDoublePi.root";
   /*
   files.at(0)="/Data/Muons/ToF/Electrons.root";
   files.at(1)="/Data/Muons/Epic-10x275/recoEL0S.root";
   files.at(2)="/Data/Tau/EpIC/tcs/tau_tcs_hist.root";
   */
   
   for(int File=0; File<NumOfFiles;File++)
   {
         // Set up input file chain
         TChain *mychain = new TChain("events");
         mychain->Add(files.at(File));

         // Initialize reader
         TTreeReader tree_reader(mychain);

         // Get Particle Information
         TTreeReaderArray<int> partGenStat(tree_reader, "MCParticles.generatorStatus");
         TTreeReaderArray<double> partMomX(tree_reader, "MCParticles.momentum.x");
         TTreeReaderArray<double> partMomY(tree_reader, "MCParticles.momentum.y");
         TTreeReaderArray<double> partMomZ(tree_reader, "MCParticles.momentum.z");
         TTreeReaderArray<int> partPdg(tree_reader, "MCParticles.PDG");
         TTreeReaderArray<double> partMass(tree_reader, "MCParticles.mass");
         TTreeReaderArray<float> partCharge(tree_reader, "MCParticles.charge");
         TTreeReaderArray<unsigned int> partParb(tree_reader, "MCParticles.parents_begin");
         TTreeReaderArray<unsigned int> partPare(tree_reader, "MCParticles.parents_end");
         TTreeReaderArray<int> partParI(tree_reader, "_MCParticles_parents.index");

         // Get Reconstructed Track Information
         TTreeReaderArray<float> trackMomX(tree_reader, "ReconstructedChargedParticles.momentum.x");
         TTreeReaderArray<float> trackMomY(tree_reader, "ReconstructedChargedParticles.momentum.y");
         TTreeReaderArray<float> trackMomZ(tree_reader, "ReconstructedChargedParticles.momentum.z");
         TTreeReaderArray<int> trackPDG(tree_reader, "ReconstructedChargedParticles.PDG");
         TTreeReaderArray<float> trackMass(tree_reader, "ReconstructedChargedParticles.mass");
         TTreeReaderArray<float> trackCharge(tree_reader, "ReconstructedChargedParticles.charge");
         TTreeReaderArray<float> trackEng(tree_reader, "ReconstructedChargedParticles.energy");

         // Get Associations Between MCParticles and ReconstructedChargedParticles
         TTreeReaderArray<unsigned int> recoAssoc(tree_reader, "ReconstructedChargedParticleAssociations.recID");
         TTreeReaderArray<unsigned int> simuAssoc(tree_reader, "ReconstructedChargedParticleAssociations.simID");


         // Get Forward Detectors Information
         TTreeReaderArray<float> RPEng(tree_reader, "ForwardRomanPotRecParticles.energy");
         TTreeReaderArray<float> RPMomX(tree_reader, "ForwardRomanPotRecParticles.momentum.x");
         TTreeReaderArray<float> RPMomY(tree_reader, "ForwardRomanPotRecParticles.momentum.y");
         TTreeReaderArray<float> RPMomZ(tree_reader, "ForwardRomanPotRecParticles.momentum.z");

         // Get Gas Cherenkov Detectors Information
         TTreeReaderArray<float> DRGaNpe(tree_reader, "DRICHGasIrtCherenkovParticleID.npe");
         TTreeReaderArray<float> DRGaThB(tree_reader, "DRICHGasIrtCherenkovParticleID.thetaPhiPhotons_begin");
         TTreeReaderArray<float> DRGaThE(tree_reader, "DRICHGasIrtCherenkovParticleID.thetaPhiPhotons_end");

         // Get Aerogel Cherenkov Detectors Information
         TTreeReaderArray<float> DRAeNpe(tree_reader, "DRICHAerogelIrtCherenkovParticleID.npe");
         TTreeReaderArray<float> DRAeThB(tree_reader, "DRICHAerogelIrtCherenkovParticleID.thetaPhiPhotons_begin");
         TTreeReaderArray<float> DRAeThE(tree_reader, "DRICHAerogelIrtCherenkovParticleID.thetaPhiPhotons_end");


        int eventID=0;
        double FoundParticles=0;
        double particscount=0;
        double BadPDG=0;

        while(tree_reader.Next()){
            eventID++;


    
            for(int particle=0; particle<trackEng.GetSize();particle++)
            {
               //Obligatory Cuts 
               int Found=0;
               TLorentzVector Partic;
               Partic.SetPxPyPzE(trackMomX[particle],trackMomY[particle],trackMomZ[particle],trackEng[particle]);

               if(Partic.Theta()*DEG>178) continue;

               PDG[File]->Fill(trackPDG[particle]);

               

               AllParticTheta[File]->Fill(Partic.Theta()*DEG);
               AllParticPhi[File]->Fill(Partic.Phi()*DEG);
               AllParticEnergy[File]->Fill(Partic.Energy());
               if(Found==0)
               {
                  NotFoundParticTheta[File]->Fill(Partic.Theta()*DEG);
                  NotFoundParticPhi[File]->Fill(Partic.Phi()*DEG);
                  NotFoundParticEnergy[File]->Fill(Partic.Energy());
               }

            }
         }
        cout<<"==========================="<<endl;
        cout<<"End of "<< File+1 << " file"<<endl;
        cout<<"Number of events: "<<eventID<<endl;
        cout<<"Found particles: "<<FoundParticles<<"   All particles: "<<particscount<<endl;
        cout<<"Found Ratio: "<<FoundParticles*100/particscount<<'%'<<endl;
        cout<<"Reconstruction with good PDG "<<(1-(BadPDG/particscount))*100<<'%'<<endl;
        cout<<"==========================="<<endl;
    }
   gStyle->SetOptStat(111111);
   TCanvas c1;

   TLegend* leg = new TLegend(0.58, 0.6, 0.85, 0.85);
    leg->SetBorderSize(0);
    leg->SetNColumns(1);
    leg->SetColumnSeparation(0.1);
    leg->SetEntrySeparation(0.1);
    leg->SetMargin(0.15);
    leg->SetTextFont(42);
    leg->SetTextSize(0.05);
    leg->AddEntry(EnergyEcal[1],"Muons","l");
    leg->AddEntry(EnergyEcal[0],"Electrons","l");
    leg->AddEntry(EnergyEcal[2],"Pions","l");


   c1.SaveAs("Plots/DRICH.pdf[");  
   c1.Clear();
      EnergyEcal[0]->Scale(1./EnergyEcal[0]->Integral());
      EnergyEcal[1]->Scale(1./EnergyEcal[1]->Integral());
      EnergyEcal[2]->Scale(1./EnergyEcal[2]->Integral());

      EnergyEcal[0]->SetLineColor(kRed);
      EnergyEcal[1]->SetLineColor(kBlue);
      EnergyEcal[2]->SetLineColor(kGreen);

      EnergyEcal[1]->Draw("HIST");
      EnergyEcal[0]->Draw("HIST SAME");
      EnergyEcal[2]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/DRICH.pdf");

   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      AllParticTheta[0]->SetLineColor(kBlue);
      NotFoundParticTheta[0]->SetLineColor(kRed);
      
      AllParticTheta[0]->Draw();
      NotFoundParticTheta[0]->Draw("same");
   c1.cd(2);
      AllParticPhi[0]->SetLineColor(kBlue);
      NotFoundParticPhi[0]->SetLineColor(kRed);
      AllParticPhi[0]->SetMinimum(0);
      AllParticPhi[0]->Draw();
      NotFoundParticPhi[0]->Draw("same");
   c1.cd(3); 
      AllParticEnergy[0]->SetLineColor(kBlue);
      NotFoundParticEnergy[0]->SetLineColor(kRed);
      AllParticEnergy[0]->Draw();
      NotFoundParticEnergy[0]->Draw("same");   
   c1.SaveAs("Plots/DRICH.pdf");

   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      AllParticTheta[1]->SetLineColor(kBlue);
      NotFoundParticTheta[1]->SetLineColor(kRed);
      AllParticTheta[1]->Draw();
      NotFoundParticTheta[1]->Draw("same");
   c1.cd(2);
      AllParticPhi[1]->SetLineColor(kBlue);
      NotFoundParticPhi[1]->SetLineColor(kRed);
      AllParticPhi[1]->SetMinimum(0);

      AllParticPhi[1]->Draw();
      NotFoundParticPhi[1]->Draw("same");
   c1.cd(3); 
      AllParticEnergy[1]->SetLineColor(kBlue);
      NotFoundParticEnergy[1]->SetLineColor(kRed);
      AllParticEnergy[1]->Draw();
      NotFoundParticEnergy[1]->Draw("same");   
   c1.SaveAs("Plots/DRICH.pdf");

   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      AllParticTheta[2]->SetLineColor(kBlue);
      NotFoundParticTheta[2]->SetLineColor(kRed);
      AllParticTheta[2]->Draw();
      NotFoundParticTheta[2]->Draw("same");
   c1.cd(2);
      AllParticPhi[2]->SetLineColor(kBlue);
      NotFoundParticPhi[2]->SetLineColor(kRed);
      AllParticPhi[2]->SetMinimum(0);

      AllParticPhi[2]->Draw();
      NotFoundParticPhi[2]->Draw("same");
   c1.cd(3); 
      AllParticEnergy[2]->SetLineColor(kBlue);
      NotFoundParticEnergy[2]->SetLineColor(kRed);
      AllParticEnergy[2]->Draw();
      NotFoundParticEnergy[2]->Draw("same");   
   c1.SaveAs("Plots/DRICH.pdf");
   c1.SaveAs("Plots/DRICH.pdf]");

}

