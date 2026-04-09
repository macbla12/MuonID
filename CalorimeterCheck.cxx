
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
#include <tuple>

#include "ToFFastSim.cxx"
#include "Calorimeters.cxx"

void CalorimeterCheck()
{
   static double MuonMass=0.1056583;
   static double ElectronMass=0.00051099895;
   static double PionMass=0.13957039;

   gROOT->SetBatch(kTRUE);
   gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");
   //gStyle->SetOptStat(0);

   double DEG=180/TMath::Pi();


   static constexpr int NumOfFiles=3;
   TH1D *EnergyEcal[NumOfFiles],*EnergyHcal[NumOfFiles],*NumberEcal[NumOfFiles],*NumberHcal[NumOfFiles];
   TH1D *NumberEcalBarrel[NumOfFiles],*NumberEcalEndcapP[NumOfFiles],*NumberEcalEndcapN[NumOfFiles],*NumberHcalBarrel[NumOfFiles],
      *NumberHcalEndcapP[NumOfFiles],*NumberHcalEndcapN[NumOfFiles],*NumberLFHcal[NumOfFiles],*NumberB0Barrel[NumOfFiles];
   TH1D *PDG[NumOfFiles],*NumberParticles[NumOfFiles];
   TH1D *TripleECalParticTheta[NumOfFiles], *TripleECalParticPhi[NumOfFiles], *TripleECalParticEnergy[NumOfFiles],*TripleECalParticPt[NumOfFiles];
   TH1D *TripleHCalParticTheta[NumOfFiles], *TripleHCalParticPhi[NumOfFiles], *TripleHCalParticEnergy[NumOfFiles],*TripleHCalParticPt[NumOfFiles];
   TH1D *ParticTheta[NumOfFiles], *ParticPhi[NumOfFiles], *ParticEnergy[NumOfFiles],*ParticPt[NumOfFiles];

   TH1D *NotFoundParticTheta[NumOfFiles], *NotFoundParticPhi[NumOfFiles], *NotFoundParticEnergy[NumOfFiles];
   TH1D *ECalEnergyHist[NumOfFiles], *ECalEnergyMomHist[NumOfFiles],*HCalEnergyHist[NumOfFiles], *HCalEnergyMomHist[NumOfFiles];
   TH2D *ECalEnergyvsMomHist[NumOfFiles],*HCalEnergyvsMomHist[NumOfFiles];
   
   vector<TString> files(NumOfFiles);

   files.at(0)="/run/media/epic/Data/Muons/Grape-10x275/Paper/RECO/FULLRECO.root";
   files.at(1)="/run/media/epic/Data/Background/JPsi/Fulldata.root";
   files.at(2)="/run/media/epic/Data/Tau/reco/Energy_10x275/double_pi/recoDoublePi.root";
   TF1 *upperbondE = new TF1("upperbondE", "2/x", 0.5, 20.0);
   upperbondE->SetLineColor(kRed);
   upperbondE->SetLineWidth(1);

   TF1 *upperbondH = new TF1("upperbondH", "2.7/x", 0.5, 20.0); 
   upperbondH->SetLineColor(kRed);
   upperbondH->SetLineWidth(1);
      
   TF1 *lowerbondH = new TF1("lowerbondH", "0.35/x-0.25/(x*x)", 0.5, 20.0); 
   lowerbondH->SetLineColor(kRed);
   lowerbondH->SetLineWidth(1);

   
   for(int File=0; File<NumOfFiles;File++)
   {
      string name;
      if(File==0) name="Muons";
      if(File==1) name="Electrons";
      if(File==2) name="Pions";

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

      // Get B0 Information
      TTreeReaderArray<unsigned int> recoAssocB0(tree_reader, "B0ECalClusterAssociations.recID");
      TTreeReaderArray<unsigned int> simuAssocB0(tree_reader, "B0ECalClusterAssociations.simID");
      TTreeReaderArray<float> B0Eng(tree_reader, "B0ECalClusters.energy");
      TTreeReaderArray<unsigned int> B0ShPB(tree_reader, "B0ECalClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> B0ShPE(tree_reader, "B0ECalClusters.shapeParameters_end");
      TTreeReaderArray<float> B0ShParameters(tree_reader, "_B0ECalClusters_shapeParameters");



      TTreeReaderArray<float> B0z(tree_reader, "B0ECalClusters.position.z");

      // Get Forward Detectors Information
      TTreeReaderArray<float> RPEng(tree_reader, "ForwardRomanPotRecParticles.energy");
      TTreeReaderArray<float> RPMomX(tree_reader, "ForwardRomanPotRecParticles.momentum.x");
      TTreeReaderArray<float> RPMomY(tree_reader, "ForwardRomanPotRecParticles.momentum.y");
      TTreeReaderArray<float> RPMomZ(tree_reader, "ForwardRomanPotRecParticles.momentum.z");

      TTreeReaderArray<float> OffMEng(tree_reader, "ForwardOffMRecParticles.energy");

      // Ecal Information
      TTreeReaderArray<unsigned int> simuAssocEcalBarrel(tree_reader, "EcalBarrelClusterAssociations.simID");
      TTreeReaderArray<unsigned int> recoAssocEcalBarrel(tree_reader, "EcalBarrelClusterAssociations.recID");
      TTreeReaderArray<float> EcalBarrelEng(tree_reader, "EcalBarrelClusters.energy");


      TTreeReaderArray<unsigned int> simuAssocEcalBarrelImaging(tree_reader, "EcalBarrelImagingClusterAssociations.simID");
      TTreeReaderArray<unsigned int> recoAssocEcalBarrelImaging(tree_reader, "EcalBarrelImagingClusterAssociations.recID");
      TTreeReaderArray<float> EcalBarrelImagingEng(tree_reader, "EcalBarrelImagingClusters.energy");

      TTreeReaderArray<unsigned int> simuAssocEcalBarrelScFi(tree_reader, "EcalBarrelScFiClusterAssociations.simID");
      TTreeReaderArray<unsigned int> recoAssocEcalBarrelScFi(tree_reader, "EcalBarrelScFiClusterAssociations.recID");
      TTreeReaderArray<float> EcalBarrelScFiEng(tree_reader, "EcalBarrelScFiClusters.energy");

      TTreeReaderArray<unsigned int> simuAssocEcalEndcapP(tree_reader, "EcalEndcapPClusterAssociations.simID");
      TTreeReaderArray<unsigned int> recoAssocEcalEndcapP(tree_reader, "EcalEndcapPClusterAssociations.recID");    
      TTreeReaderArray<float> EcalEndcapPEng(tree_reader, "EcalEndcapPClusters.energy");

      TTreeReaderArray<unsigned int> simuAssocEcalEndcapN(tree_reader, "EcalEndcapNClusterAssociations.simID");
      TTreeReaderArray<unsigned int> recoAssocEcalEndcapN(tree_reader, "EcalEndcapNClusterAssociations.recID");
      TTreeReaderArray<float> EcalEndcapNEng(tree_reader, "EcalEndcapNClusters.energy");

      // Hcal Information
      TTreeReaderArray<unsigned int> simuAssocHcalBarrel(tree_reader, "HcalBarrelClusterAssociations.simID");
      TTreeReaderArray<unsigned int> recoAssocHcalBarrel(tree_reader, "HcalBarrelClusterAssociations.recID");
      TTreeReaderArray<float> HcalBarrelEng(tree_reader, "HcalBarrelClusters.energy");

      TTreeReaderArray<unsigned int> simuAssocHcalEndcapP(tree_reader, "HcalEndcapPInsertClusterAssociations.simID");
      TTreeReaderArray<unsigned int> recoAssocHcalEndcapP(tree_reader, "HcalEndcapPInsertClusterAssociations.recID");    
      TTreeReaderArray<float> HcalEndcapPEng(tree_reader, "HcalEndcapPInsertClusters.energy");

      TTreeReaderArray<unsigned int> simuAssocLFHcal(tree_reader, "LFHCALClusterAssociations.simID");
      TTreeReaderArray<unsigned int> recoAssocLFHcal(tree_reader, "LFHCALClusterAssociations.recID");    
      TTreeReaderArray<float> LFHcalEng(tree_reader, "LFHCALClusters.energy");

      TTreeReaderArray<unsigned int> simuAssocHcalEndcapN(tree_reader, "HcalEndcapNClusterAssociations.simID");
      TTreeReaderArray<unsigned int> recoAssocHcalEndcapN(tree_reader, "HcalEndcapNClusterAssociations.recID");
      TTreeReaderArray<float> HcalEndcapNEng(tree_reader, "HcalEndcapNClusters.energy");

      //==================================//
      NumberB0Barrel[File]= new TH1D(Form("NumberB0Barrel%s",name.c_str()),Form("NumberB0Barrel%s",name.c_str()),5,-0.5,4.5);
      NumberEcalBarrel[File]= new TH1D(Form("NumberEcalBarrel%s",name.c_str()),Form("NumberEcalBarrel%s",name.c_str()),5,-0.5,4.5);
      NumberEcalEndcapP[File]= new TH1D(Form("NumberEcalEndcapP%s",name.c_str()),Form("NumberEcalEndcapP%s",name.c_str()),5,-0.5,4.5);
      NumberEcalEndcapN[File]= new TH1D(Form("NumberEcalEndcapN%s",name.c_str()),Form("NumberEcalEndcapN%s",name.c_str()),5,-0.5,4.5);
      NumberHcalBarrel[File]= new TH1D(Form("NumberHcalBarrel%s",name.c_str()),Form("NumberHcalBarrel%s",name.c_str()),5,-0.5,4.5);
      NumberHcalEndcapP[File]= new TH1D(Form("NumberHcalEndcapP%s",name.c_str()),Form("NumberHcalEndcapP%s",name.c_str()),10,-0.5,9.5);
      NumberHcalEndcapN[File]= new TH1D(Form("NumberHcalEndcapN%s",name.c_str()),Form("NumberHcalEndcapN%s",name.c_str()),10,-0.5,9.5);
      NumberLFHcal[File]= new TH1D(Form("NumberLFHcal%s",name.c_str()),Form("NumberLFHcal%s",name.c_str()),10,-0.5,9.5);
     

      //==================================//
      NumberEcal[File]= new TH1D(Form("NumberEcal%s",name.c_str()),Form("NumberEcal%s",name.c_str()),10,-0.5,9.5);
      NumberHcal[File]= new TH1D(Form("NumberHcal%s",name.c_str()),Form("NumberHcal%s",name.c_str()),10,-0.5,9.5);
      EnergyEcal[File]= new TH1D(Form("EnergyEcal%s",name.c_str()),Form("EnergyEcal%s",name.c_str()),100,0,7);
      EnergyHcal[File]= new TH1D(Form("EnergyHcal%s",name.c_str()),Form("EnergyHcal%s",name.c_str()),100,0,7);
      //==================================//
      PDG[File]= new TH1D(Form("PDG%s",name.c_str()),Form("PDG%s",name.c_str()),41,-230.5,230);
      NumberParticles[File]= new TH1D(Form("NumberParticles%s",name.c_str()),Form("NumberParticles%s",name.c_str()),9,-0.5,8.5);
      //==================================//
      TripleHCalParticTheta[File] = new TH1D(Form("TripleHCalParticTheta%s",name.c_str()),Form("TripleHCalParticTheta%s",name.c_str()),50,0,180);
      TripleHCalParticPhi[File]= new TH1D(Form("TripleHCalParticPhi%s",name.c_str()),Form("TripleHCalParticPhi%s",name.c_str()),30,-180,180);
      TripleHCalParticEnergy[File]= new TH1D(Form("TripleHCalParticEnergy%s",name.c_str()),Form("TripleHCalParticEnergy%s",name.c_str()),50,0,10);
      TripleHCalParticPt[File]= new TH1D(Form("TripleHCalParticPt%s",name.c_str()),Form("TripleHCalParticPt%s",name.c_str()),50,0,3);
      
      TripleECalParticTheta[File] = new TH1D(Form("TripleECalParticTheta%s",name.c_str()),Form("TripleECalParticTheta%s",name.c_str()),50,0,180);
      TripleECalParticPhi[File]= new TH1D(Form("TripleECalParticPhi%s",name.c_str()),Form("TripleECalParticPhi%s",name.c_str()),30,-180,180);
      TripleECalParticEnergy[File]= new TH1D(Form("TripleECalParticEnergy%s",name.c_str()),Form("TripleECalParticEnergy%s",name.c_str()),50,0,10);
      TripleECalParticPt[File]= new TH1D(Form("TripleECalParticPt%s",name.c_str()),Form("TripleECalParticPt%s",name.c_str()),50,0,3);

      ParticTheta[File] = new TH1D(Form("ParticTheta%s",name.c_str()),Form("ParticTheta%s",name.c_str()),50,0,180);
      ParticPhi[File]= new TH1D(Form("ParticPhi%s",name.c_str()),Form("ParticPhi%s",name.c_str()),30,-180,180);
      ParticEnergy[File]= new TH1D(Form("ParticEnergy%s",name.c_str()),Form("ParticEnergy%s",name.c_str()),50,0,10);
      ParticPt[File]= new TH1D(Form("ParticPt%s",name.c_str()),Form("ParticPt%s",name.c_str()),50,0,3);

      //==================================//
      ECalEnergyHist[File]= new TH1D(Form("ECalEnergyHist%s",name.c_str()),Form("ECalEnergyHist%s",name.c_str()),50,0,1);
      ECalEnergyMomHist[File]= new TH1D(Form("ECalEnergyMomHist%s",name.c_str()),Form("ECalEnergyMomHist%s",name.c_str()),50,0,0.2);
      ECalEnergyvsMomHist[File]= new TH2D(Form("ECalEnergyvsMomHist%s",name.c_str()),Form("ECalEnergyvsMomHist%s",name.c_str()),50,0,22,50,0,6);

      HCalEnergyHist[File]= new TH1D(Form("HCalEnergyHist%s",name.c_str()),Form("HCalEnergyHist%s",name.c_str()),50,0,8);
      HCalEnergyMomHist[File]= new TH1D(Form("HCalEnergyMomHist%s",name.c_str()),Form("HCalEnergyMomHist%s",name.c_str()),50,0,4);
      HCalEnergyvsMomHist[File]= new TH2D(Form("HCalEnergyvsMomHist%s",name.c_str()),Form("HCalEnergyvsMomHist%s",name.c_str()),50,0,22,50,0,6);

      
      int eventID=0;
      double FoundParticles=0;
      double particscount=0;
      double BadPDG=0;
            int c=0;

      while(tree_reader.Next()){
         eventID++;
         //if(eventID>10) break;
         if(eventID>50000) break;
         if(eventID%20000==0) cout<<"File "<<name<<" and event number... "<<eventID<<endl;


         NumberB0Barrel[File]->Fill(B0Eng.GetSize());
         NumberEcalBarrel[File]->Fill(EcalBarrelEng.GetSize());
         NumberEcalEndcapP[File]->Fill(EcalEndcapPEng.GetSize());
         NumberEcalEndcapN[File]->Fill(EcalEndcapNEng.GetSize());
         NumberHcalBarrel[File]->Fill(HcalBarrelEng.GetSize());
         NumberHcalEndcapN[File]->Fill(HcalEndcapNEng.GetSize());
         NumberHcalEndcapP[File]->Fill(HcalEndcapPEng.GetSize());
         NumberLFHcal[File]->Fill(LFHcalEng.GetSize());
         

         int id=0;
         for(int particle=0; particle<trackEng.GetSize();particle++)
         {
            //Obligatory Cuts 
            double mass;
            if(File==0) mass=MuonMass;
            else if(File==1) mass=ElectronMass;
            else if(File==2) mass=PionMass;

            int Found=0;
            TLorentzVector Partic;
            Partic.SetPxPyPzE(trackMomX[particle],trackMomY[particle],trackMomZ[particle],trackEng[particle]);
           
            if(Partic.Theta()*DEG>178) continue;
         
            PDG[File]->Fill(trackPDG[particle]);
            if(!(trackPDG[particle]==0 || abs(trackPDG[particle])==13)) continue;
            
           //Ecal Energy Search
            NumberParticles[File]->Fill(simuAssocEcalBarrel.GetSize());
            int simuID = simuAssoc[particle];
            auto [ECalEnergy,ECalNumber,HCalEnergy,HCalNumber] = Calorimeters(Partic, simuID, EcalBarrelEng, EcalEndcapPEng, EcalEndcapNEng, HcalBarrelEng, HcalEndcapPEng, LFHcalEng, HcalEndcapNEng, B0Eng, EcalBarrelImagingEng, EcalBarrelScFiEng, 
               simuAssocEcalBarrel, simuAssocEcalEndcapP, simuAssocEcalEndcapN, simuAssocHcalBarrel, simuAssocHcalEndcapP, simuAssocLFHcal, simuAssocHcalEndcapN, simuAssocB0, simuAssocEcalBarrelImaging, simuAssocEcalBarrelScFi,B0ShPB,B0ShPE,B0ShParameters);
            if(ECalEnergy!=0)
            {
               EnergyEcal[File]->Fill(ECalEnergy);
               Found=1;
            }
            
           //Hcal Energy Search
           
            
            if(HCalEnergy!=0)
            {
               EnergyHcal[File]->Fill(HCalEnergy);
               Found=1;
            }
            NumberEcal[File]->Fill(ECalNumber);
            NumberHcal[File]->Fill(HCalNumber);

            particscount++;
            FoundParticles+=Found;

            //Track properties 
            double FullEnergy=HCalEnergy+ECalEnergy;
            if(FullEnergy==0) continue;
            double Momentum=Partic.P();
            double EcalEoverP=ECalEnergy/Momentum;
            double HcalEoverP=HCalEnergy/Momentum;

            ECalEnergyHist[File]->Fill(ECalEnergy);
            ECalEnergyMomHist[File]->Fill(EcalEoverP);
            ECalEnergyvsMomHist[File]->Fill(Momentum,EcalEoverP);
            HCalEnergyHist[File]->Fill(HCalEnergy);
            HCalEnergyMomHist[File]->Fill(HcalEoverP);
            HCalEnergyvsMomHist[File]->Fill(Momentum,HcalEoverP);
            
             
            ParticTheta[File]->Fill(Partic.Theta()*DEG);
            ParticPhi[File]->Fill(Partic.Phi()*DEG);
            ParticEnergy[File]->Fill(Partic.Energy());
            ParticPt[File]->Fill(Partic.Pt());
            if(HcalEoverP>upperbondH->Eval(Momentum)) 
            {
               TripleHCalParticTheta[File]->Fill(Partic.Theta()*DEG);
               TripleHCalParticPhi[File]->Fill(Partic.Phi()*DEG);
               TripleHCalParticEnergy[File]->Fill(Partic.Energy());
               TripleHCalParticPt[File]->Fill(Partic.Pt());

            }
            
            if(EcalEoverP>upperbondE->Eval(Momentum)) 
            {
               TripleECalParticTheta[File]->Fill(Partic.Theta()*DEG);
               TripleECalParticPhi[File]->Fill(Partic.Phi()*DEG);
               TripleECalParticEnergy[File]->Fill(Partic.Energy());
               TripleECalParticPt[File]->Fill(Partic.Pt());

            } 

         } 
      }
      

      cout<<"==========================="<<endl;
      cout<<"End of "<< name << " file"<<endl;
      cout<<"Number of events: "<<eventID<<endl;
      cout<<"Found particles: "<<FoundParticles<<"   All particles: "<<particscount<<endl;
      cout<<"Found Ratio: "<<FoundParticles*100/particscount<<'%'<<endl;
      cout<<"==========================="<<endl;
   }
   
   gStyle->SetOptStat(111111);
   //gStyle->SetOptStat(000000);
   TCanvas c1;

   TLegend* leg = new TLegend(0.58, 0.6, 0.85, 0.85);
    leg->SetBorderSize(0);
    leg->SetNColumns(1);
    leg->SetColumnSeparation(0.1);
    leg->SetEntrySeparation(0.1);
    leg->SetMargin(0.15);
    leg->SetTextFont(42);
    leg->SetTextSize(0.05);  
    leg->AddEntry(HCalEnergyMomHist[0],"Muons DP","l");
    leg->AddEntry(HCalEnergyMomHist[1],"Muons J/Psi","l");
    leg->AddEntry(HCalEnergyMomHist[2],"Pions","l");
   TLegend* leg2 = new TLegend(0.38, 0.6, 0.45, 0.85);
    leg2->SetBorderSize(0);
    leg2->SetNColumns(1);
    leg2->SetColumnSeparation(0.1);
    leg2->SetEntrySeparation(0.1);
    leg2->SetMargin(0.15);
    leg2->SetTextFont(42);
    leg2->SetTextSize(0.05);  
    leg2->AddEntry(ParticTheta[0],"All Paritcles","l");
    leg2->AddEntry(TripleHCalParticTheta[0],"Triple Hcal","l");
    leg2->AddEntry(TripleECalParticTheta[0],"Triple Ecal","l");
   c1.SaveAs("Plots/CalimeterCheck.pdf[");

    c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      ParticTheta[0]->SetLineColor(kBlue);
      TripleHCalParticTheta[0]->SetLineColor(kRed);
      TripleECalParticTheta[0]->SetLineColor(kGreen);
      
      ParticTheta[0]->Draw();
      TripleHCalParticTheta[0]->Draw("same");
      TripleECalParticTheta[0]->Draw("same");
      leg2->Draw();
   c1.cd(2);
      ParticPhi[0]->SetMinimum(0);

      ParticPhi[0]->SetLineColor(kBlue);
      TripleHCalParticPhi[0]->SetLineColor(kRed);
      TripleECalParticPhi[0]->SetLineColor(kGreen);
      
      ParticPhi[0]->Draw();
      TripleHCalParticPhi[0]->Draw("same");
      TripleECalParticPhi[0]->Draw("same");
      leg2->Draw();

   c1.cd(3); 
      ParticEnergy[0]->SetLineColor(kBlue);
      TripleHCalParticEnergy[0]->SetLineColor(kRed);
      TripleECalParticEnergy[0]->SetLineColor(kGreen);
      
      ParticEnergy[0]->Draw();
      TripleHCalParticEnergy[0]->Draw("same");
      TripleECalParticEnergy[0]->Draw("same");  
      leg2->Draw();

   c1.cd(4); 
      ParticPt[0]->SetLineColor(kBlue);
      TripleHCalParticPt[0]->SetLineColor(kRed);
      TripleECalParticPt[0]->SetLineColor(kGreen);
      
      ParticPt[0]->Draw();
      TripleHCalParticPt[0]->Draw("same");
      TripleECalParticPt[0]->Draw("same"); 
      leg2->Draw();

   c1.SaveAs("Plots/CalimeterCheck.pdf");
    c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      ParticTheta[1]->SetLineColor(kBlue);
      TripleHCalParticTheta[1]->SetLineColor(kRed);
      TripleECalParticTheta[1]->SetLineColor(kGreen);
      
      ParticTheta[1]->Draw();
      TripleHCalParticTheta[1]->Draw("same");
      TripleECalParticTheta[1]->Draw("same");
      leg2->Draw();

   c1.cd(2);
      ParticPhi[1]->SetMinimum(0);

      ParticPhi[1]->SetLineColor(kBlue);
      TripleHCalParticPhi[1]->SetLineColor(kRed);
      TripleECalParticPhi[1]->SetLineColor(kGreen);
      
      ParticPhi[1]->Draw();
      TripleHCalParticPhi[1]->Draw("same");
      TripleECalParticPhi[1]->Draw("same");
      leg2->Draw();

   c1.cd(3); 
      ParticEnergy[1]->SetLineColor(kBlue);
      TripleHCalParticEnergy[1]->SetLineColor(kRed);
      TripleECalParticEnergy[1]->SetLineColor(kGreen);
      
      ParticEnergy[1]->Draw();
      TripleHCalParticEnergy[1]->Draw("same");
      TripleECalParticEnergy[1]->Draw("same"); 
      leg2->Draw();

   c1.cd(4); 

      ParticPt[1]->SetLineColor(kBlue);
      TripleHCalParticPt[1]->SetLineColor(kRed);
      TripleECalParticPt[1]->SetLineColor(kGreen);
      
      ParticPt[1]->Draw();
      TripleHCalParticPt[1]->Draw("same");
      TripleECalParticPt[1]->Draw("same"); 
      leg2->Draw();

   c1.SaveAs("Plots/CalimeterCheck.pdf");
   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      ParticTheta[2]->SetLineColor(kBlue);
      TripleHCalParticTheta[2]->SetLineColor(kRed);
      TripleECalParticTheta[2]->SetLineColor(kGreen);
      
      ParticTheta[2]->Draw();
      TripleHCalParticTheta[2]->Draw("same");
      TripleECalParticTheta[2]->Draw("same");
      leg2->Draw();

   c1.cd(2);
      ParticPhi[2]->SetMinimum(0);

      ParticPhi[2]->SetLineColor(kBlue);
      TripleHCalParticPhi[2]->SetLineColor(kRed);
      TripleECalParticPhi[2]->SetLineColor(kGreen);
      
      ParticPhi[2]->Draw();
      TripleHCalParticPhi[2]->Draw("same");
      TripleECalParticPhi[2]->Draw("same");
      leg2->Draw();

   c1.cd(3); 
      ParticEnergy[2]->SetLineColor(kBlue);
      TripleHCalParticEnergy[2]->SetLineColor(kRed);
      TripleECalParticEnergy[2]->SetLineColor(kGreen);
      
      ParticEnergy[2]->Draw();
      TripleHCalParticEnergy[2]->Draw("same");
      TripleECalParticEnergy[2]->Draw("same");  
      leg2->Draw();

   c1.cd(4); 
      ParticPt[2]->SetLineColor(kBlue);
      TripleHCalParticPt[2]->SetLineColor(kRed);
      TripleECalParticPt[2]->SetLineColor(kGreen);
      
      ParticPt[2]->Draw();
      TripleHCalParticPt[2]->Draw("same");
      TripleECalParticPt[2]->Draw("same"); 
      leg2->Draw();

   c1.SaveAs("Plots/CalimeterCheck.pdf");

   c1.Clear();
   
      HCalEnergyMomHist[0]->Scale(1./HCalEnergyMomHist[0]->Integral());
      HCalEnergyMomHist[1]->Scale(1./HCalEnergyMomHist[1]->Integral());
      HCalEnergyMomHist[2]->Scale(1./HCalEnergyMomHist[2]->Integral());

      HCalEnergyMomHist[0]->SetLineColor(kRed);
      HCalEnergyMomHist[1]->SetLineColor(kBlue);
      HCalEnergyMomHist[2]->SetLineColor(kGreen);

      HCalEnergyMomHist[1]->Draw("HIST");
      HCalEnergyMomHist[0]->Draw("HIST SAME");
      HCalEnergyMomHist[2]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalimeterCheck.pdf");

   c1.Clear();
      ECalEnergyMomHist[0]->Scale(1./ECalEnergyMomHist[0]->Integral());
      ECalEnergyMomHist[1]->Scale(1./ECalEnergyMomHist[1]->Integral());
      ECalEnergyMomHist[2]->Scale(1./ECalEnergyMomHist[2]->Integral());

      ECalEnergyMomHist[0]->SetLineColor(kRed);
      ECalEnergyMomHist[1]->SetLineColor(kBlue);
      ECalEnergyMomHist[2]->SetLineColor(kGreen);

      ECalEnergyMomHist[1]->Draw("HIST");
      ECalEnergyMomHist[0]->Draw("HIST SAME");
      ECalEnergyMomHist[2]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalimeterCheck.pdf");
   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      gPad->SetLogz(1);
      ECalEnergyvsMomHist[0]->Draw("HIST");
      upperbondE->Draw("same");

   c1.cd(2);
      gPad->SetLogz(1);
      ECalEnergyvsMomHist[1]->Draw("HIST");
      upperbondE->Draw("same");

   c1.cd(3);
      gPad->SetLogz(1);
      ECalEnergyvsMomHist[2]->Draw("HIST");
      upperbondE->Draw("same");

   c1.SaveAs("Plots/CalimeterCheck.pdf");
   c1.Clear();
   c1.Divide(2,2);
   gPad->SetLogz(0);
     c1.cd(1);
      gPad->SetLogz(1);
      HCalEnergyvsMomHist[0]->Draw("HIST");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
   c1.cd(2);
      gPad->SetLogz(1);
      HCalEnergyvsMomHist[1]->Draw("HIST");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
   c1.cd(3);
      gPad->SetLogz(1);
      HCalEnergyvsMomHist[2]->Draw("HIST");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
   c1.SaveAs("Plots/CalimeterCheck.pdf");
   gPad->SetLogz(0);

   c1.SaveAs("Plots/CalimeterCheck.pdf]");

    
}

