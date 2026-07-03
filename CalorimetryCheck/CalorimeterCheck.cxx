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

//#include "ToFFastSim.cxx"
#include "CalorimeterValues.cxx"

void CalorimeterCheck()
{
   static double MuonMass=0.1056583;
   static double ElectronMass=0.00051099895;
   static double PionMass=0.13957039;

   gROOT->SetBatch(kTRUE);
   gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");
   //gStyle->SetOptStat(0);

   double DEG=180/TMath::Pi();


   static constexpr int NumOfFiles=2;
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

   files.at(0)="/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
   files.at(1)="/run/media/epic/Data/Muons/Grape-10x275/Current/reco*.root";

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
      TTreeReaderArray<int> simuAssoc(tree_reader, "_ReconstructedChargedParticleAssociations_sim.index");

      // Get B0 Information
      TTreeReaderArray<int> simuAssocB0(tree_reader, "_B0ECalClusterAssociations_sim.index");
      TTreeReaderArray<float> B0x(tree_reader, "B0ECalClusters.position.x");
      TTreeReaderArray<float> B0y(tree_reader, "B0ECalClusters.position.y");
      TTreeReaderArray<float> B0z(tree_reader, "B0ECalClusters.position.z");
      TTreeReaderArray<float> B0Eng(tree_reader, "B0ECalClusters.energy");
      TTreeReaderArray<unsigned int> B0ShPB(tree_reader, "B0ECalClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> B0ShPE(tree_reader, "B0ECalClusters.shapeParameters_end");
      TTreeReaderArray<float> B0ShParameters(tree_reader, "_B0ECalClusters_shapeParameters");




      // Ecal Information
      TTreeReaderArray<int> simuAssocEcalBarrel(tree_reader, "_EcalBarrelClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalBarrelEng(tree_reader, "EcalBarrelClusters.energy");
      TTreeReaderArray<float> EcalBarrelx(tree_reader, "EcalBarrelClusters.position.x");
      TTreeReaderArray<float> EcalBarrely(tree_reader, "EcalBarrelClusters.position.y");
      TTreeReaderArray<float> EcalBarrelz(tree_reader, "EcalBarrelClusters.position.z");
      TTreeReaderArray<unsigned int> EcalBarrelShPB(tree_reader, "EcalBarrelClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalBarrelShPE(tree_reader, "EcalBarrelClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalBarrelShParameters(tree_reader, "_EcalBarrelClusters_shapeParameters");


      TTreeReaderArray<int> simuAssocEcalBarrelImaging(tree_reader, "_EcalBarrelImagingClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalBarrelImagingEng(tree_reader, "EcalBarrelImagingClusters.energy");
      TTreeReaderArray<float> EcalBarrelImagingx(tree_reader, "EcalBarrelImagingClusters.position.x");
      TTreeReaderArray<float> EcalBarrelImagingy(tree_reader, "EcalBarrelImagingClusters.position.y");
      TTreeReaderArray<float> EcalBarrelImagingz(tree_reader, "EcalBarrelImagingClusters.position.z");
      TTreeReaderArray<unsigned int> EcalBarrelImagingShPB(tree_reader, "EcalBarrelImagingClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalBarrelImagingShPE(tree_reader, "EcalBarrelImagingClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalBarrelImagingShParameters(tree_reader, "_EcalBarrelImagingClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocEcalBarrelScFi(tree_reader, "_EcalBarrelScFiClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalBarrelScFiEng(tree_reader, "EcalBarrelScFiClusters.energy");
      TTreeReaderArray<float> EcalBarrelScFix(tree_reader, "EcalBarrelScFiClusters.position.x");
      TTreeReaderArray<float> EcalBarrelScFiy(tree_reader, "EcalBarrelScFiClusters.position.y");
      TTreeReaderArray<float> EcalBarrelScFiz(tree_reader, "EcalBarrelScFiClusters.position.z");
      TTreeReaderArray<unsigned int> EcalBarrelScFiShPB(tree_reader, "EcalBarrelScFiClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalBarrelScFiShPE(tree_reader, "EcalBarrelScFiClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalBarrelScFiShParameters(tree_reader, "_EcalBarrelScFiClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocEcalEndcapP(tree_reader, "_EcalEndcapPClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalEndcapPEng(tree_reader, "EcalEndcapPClusters.energy");
      TTreeReaderArray<float> EcalEndcapPx(tree_reader, "EcalEndcapPClusters.position.x");
      TTreeReaderArray<float> EcalEndcapPy(tree_reader, "EcalEndcapPClusters.position.y");
      TTreeReaderArray<float> EcalEndcapPz(tree_reader, "EcalEndcapPClusters.position.z");
      TTreeReaderArray<unsigned int> EcalEndcapPShPB(tree_reader, "EcalEndcapPClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalEndcapPShPE(tree_reader, "EcalEndcapPClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalEndcapPShParameters(tree_reader, "_EcalEndcapPClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocEcalEndcapN(tree_reader, "_EcalEndcapNClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalEndcapNEng(tree_reader, "EcalEndcapNClusters.energy");
      TTreeReaderArray<float> EcalEndcapNx(tree_reader, "EcalEndcapNClusters.position.x");
      TTreeReaderArray<float> EcalEndcapNy(tree_reader, "EcalEndcapNClusters.position.y");
      TTreeReaderArray<float> EcalEndcapNz(tree_reader, "EcalEndcapNClusters.position.z");
      TTreeReaderArray<unsigned int> EcalEndcapNShPB(tree_reader, "EcalEndcapNClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalEndcapNShPE(tree_reader, "EcalEndcapNClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalEndcapNShParameters(tree_reader, "_EcalEndcapNClusters_shapeParameters");

      // Hcal Information
      TTreeReaderArray<int> simuAssocHcalBarrel(tree_reader, "_HcalBarrelClusterAssociations_sim.index");
      TTreeReaderArray<float> HcalBarrelEng(tree_reader, "HcalBarrelClusters.energy");
      TTreeReaderArray<float> HcalBarrelx(tree_reader, "HcalBarrelClusters.position.x");
      TTreeReaderArray<float> HcalBarrely(tree_reader, "HcalBarrelClusters.position.y");
      TTreeReaderArray<float> HcalBarrelz(tree_reader, "HcalBarrelClusters.position.z");
      TTreeReaderArray<unsigned int> HcalBarrelShPB(tree_reader, "HcalBarrelClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> HcalBarrelShPE(tree_reader, "HcalBarrelClusters.shapeParameters_end");
      TTreeReaderArray<float> HcalBarrelShParameters(tree_reader, "_HcalBarrelClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocHcalEndcapP(tree_reader, "_HcalEndcapPInsertClusterAssociations_sim.index");
      TTreeReaderArray<float> HcalEndcapPEng(tree_reader, "HcalEndcapPInsertClusters.energy");
      TTreeReaderArray<float> HcalEndcapPx(tree_reader, "HcalEndcapPInsertClusters.position.x");
      TTreeReaderArray<float> HcalEndcapPy(tree_reader, "HcalEndcapPInsertClusters.position.y");
      TTreeReaderArray<float> HcalEndcapPz(tree_reader, "HcalEndcapPInsertClusters.position.z");
      TTreeReaderArray<unsigned int> HcalEndcapPShPB(tree_reader, "HcalEndcapPInsertClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> HcalEndcapPShPE(tree_reader, "HcalEndcapPInsertClusters.shapeParameters_end");
      TTreeReaderArray<float> HcalEndcapPShParameters(tree_reader, "_HcalEndcapPInsertClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocLFHcal(tree_reader, "_LFHCALClusterAssociations_sim.index");
      TTreeReaderArray<float> LFHcalEng(tree_reader, "LFHCALClusters.energy");
      TTreeReaderArray<float> LFHcalx(tree_reader, "LFHCALClusters.position.x");
      TTreeReaderArray<float> LFHcaly(tree_reader, "LFHCALClusters.position.y");
      TTreeReaderArray<float> LFHcalz(tree_reader, "LFHCALClusters.position.z");
      TTreeReaderArray<unsigned int> LFHcalShPB(tree_reader, "LFHCALClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> LFHcalShPE(tree_reader, "LFHCALClusters.shapeParameters_end");
      TTreeReaderArray<float> LFHcalShParameters(tree_reader, "_LFHCALClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocHcalEndcapN(tree_reader, "_HcalEndcapNClusterAssociations_sim.index");
      TTreeReaderArray<float> HcalEndcapNEng(tree_reader, "HcalEndcapNClusters.energy");
      TTreeReaderArray<float> HcalEndcapNx(tree_reader, "HcalEndcapNClusters.position.x");
      TTreeReaderArray<float> HcalEndcapNy(tree_reader, "HcalEndcapNClusters.position.y");
      TTreeReaderArray<float> HcalEndcapNz(tree_reader, "HcalEndcapNClusters.position.z");
      TTreeReaderArray<unsigned int> HcalEndcapNShPB(tree_reader, "HcalEndcapNClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> HcalEndcapNShPE(tree_reader, "HcalEndcapNClusters.shapeParameters_end");
      TTreeReaderArray<float> HcalEndcapNShParameters(tree_reader, "_HcalEndcapNClusters_shapeParameters");

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
            if(Partic.Eta()<-1.25) continue;


         
            PDG[File]->Fill(trackPDG[particle]);
            //if(!(trackPDG[particle]==0 || abs(trackPDG[particle])==13)) continue;
            
           //Ecal Energy Search
            NumberParticles[File]->Fill(simuAssocEcalBarrel.GetSize());
            int simuID = simuAssoc[particle];
            auto [ECalEnergy,ECalNumber,HCalEnergy,HCalNumber] = CalorimeterValues(Partic, simuID, EcalBarrelEng, EcalEndcapPEng, EcalEndcapNEng, HcalBarrelEng, HcalEndcapPEng, LFHcalEng, HcalEndcapNEng, B0Eng, EcalBarrelImagingEng, EcalBarrelScFiEng, 
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
   c1.SaveAs("CalimeterCheck.pdf[");

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

   c1.SaveAs("CalimeterCheck.pdf");
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

   c1.SaveAs("CalimeterCheck.pdf");
   
   c1.Clear();
   
      HCalEnergyMomHist[0]->Scale(1./HCalEnergyMomHist[0]->Integral());
      HCalEnergyMomHist[1]->Scale(1./HCalEnergyMomHist[1]->Integral());

      HCalEnergyMomHist[0]->SetLineColor(kRed);
      HCalEnergyMomHist[1]->SetLineColor(kBlue);

      HCalEnergyMomHist[1]->Draw("HIST");
      HCalEnergyMomHist[0]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("CalimeterCheck.pdf");

   c1.Clear();
      ECalEnergyMomHist[0]->Scale(1./ECalEnergyMomHist[0]->Integral());
      ECalEnergyMomHist[1]->Scale(1./ECalEnergyMomHist[1]->Integral());

      ECalEnergyMomHist[0]->SetLineColor(kRed);
      ECalEnergyMomHist[1]->SetLineColor(kBlue);

      ECalEnergyMomHist[1]->Draw("HIST");
      ECalEnergyMomHist[0]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("CalimeterCheck.pdf");
   c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      gPad->SetLogz(1);
      ECalEnergyvsMomHist[0]->Draw("HIST");
      upperbondE->Draw("same");

   c1.cd(2);
      gPad->SetLogz(1);
      ECalEnergyvsMomHist[1]->Draw("HIST");
      upperbondE->Draw("same");

   c1.SaveAs("CalimeterCheck.pdf");
   c1.Clear();
   c1.Divide(2,1);
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

   c1.SaveAs("CalimeterCheck.pdf");
   gPad->SetLogz(0);

   c1.SaveAs("CalimeterCheck.pdf]");

    
}

