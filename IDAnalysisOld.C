
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
//#include "Calorimeters.cxx"
#include "Calorimeternew.cxx"


void IDAnalysisOld()
{
   //////////////////////
   //Setting up constants
   //////////////////////

   static double MuonMass=0.1056583;
   static double ElectronMass=0.00051099895;
   static double PionMass=0.13957039;

   gROOT->SetBatch(kTRUE);
   gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");
   //gStyle->SetOptStat(0);

   double DEG=180/TMath::Pi();

   //////////////////////
   //Setting up data.root for ML
   //////////////////////

   float ECalEnergy;
   float HCalEnergy;
   float ECalNumber;
   float HCalNumber;
   float EoverP;
   float Momentum;
   float IsMuon;
   vector<float> EcalShape;
   vector<float> HcalShape;

   TFile *file = new TFile("MLDataContinuous.root", "RECREATE");
   TTree *MLDataTree = new TTree("MLDataTree", "MLDataTree");

   MLDataTree->Branch("ECalEnergy", &ECalEnergy, "ECalEnergy/F");
   MLDataTree->Branch("HCalEnergy", &HCalEnergy, "HCalEnergy/F");
   MLDataTree->Branch("ECalNumber", &ECalNumber, "ECalNumber/F");
   MLDataTree->Branch("HCalNumber", &HCalNumber, "HCalNumber/F");
   MLDataTree->Branch("EoverP", &EoverP, "EoverP/F");
   MLDataTree->Branch("Momentum", &Momentum, "Momentum/F");
   MLDataTree->Branch("EcalShape", &EcalShape);
   MLDataTree->Branch("HcalShape", &HcalShape);
   MLDataTree->Branch("IsMuon", &IsMuon, "IsMuon/F");

   //////////////////////
   //Setting up histograms
   //////////////////////
   static constexpr int NumOfFiles=3;
   TH1D *EnergyEcal[NumOfFiles],*EnergyHcal[NumOfFiles],*NumberEcal[NumOfFiles],*NumberHcal[NumOfFiles];
   TH1D *NumberEcalBarrel[NumOfFiles],*NumberEcalEndcapP[NumOfFiles],*NumberEcalEndcapN[NumOfFiles],*NumberHcalBarrel[NumOfFiles],
      *NumberHcalEndcapP[NumOfFiles],*NumberHcalEndcapN[NumOfFiles],*NumberLFHcal[NumOfFiles],*NumberB0Barrel[NumOfFiles];
   TH1D *PDG[NumOfFiles],*NumberParticles[NumOfFiles];
   TH1D *AllParticTheta[NumOfFiles], *AllParticPhi[NumOfFiles], *AllParticEnergy[NumOfFiles];
   TH1D *NotFoundParticTheta[NumOfFiles], *NotFoundParticPhi[NumOfFiles], *NotFoundParticEnergy[NumOfFiles];
   TH1D *ECalEnergyHist[NumOfFiles], *ECalEnergyMomHist[NumOfFiles],*HCalEnergyHist[NumOfFiles], *HCalEnergyMomHist[NumOfFiles];
   TH2D *ECalEnergyvsMomHist[NumOfFiles],*HCalEnergyvsMomHist[NumOfFiles];
   TH1D *ToFTimeHist[NumOfFiles], *ToFMassHist[NumOfFiles], *ToFMassHist2[NumOfFiles], *ToFMassHist10[NumOfFiles], *ToFMassHist40[NumOfFiles];
   TH1D *DoubleToFMassHist[NumOfFiles], *DoubleToFMassHist2[NumOfFiles], *DoubleToFMassHist10[NumOfFiles], *DoubleToFMassHist40[NumOfFiles];
   TH1D *EcalShapeHist[7][NumOfFiles], *HcalShapeHist[7][NumOfFiles];
   
   vector<TString> files(NumOfFiles);
   /*
   files.at(0)="/run/media/epic/Data/Muons/Grape-10x100/Paper/recoGL.root";
   files.at(1)="/run/media/epic/Data/Muons/ToF/Electrons.root";
   files.at(2)="/run/media/epic/Data/Tau/reco/Energy_10x275/double_pi/recoDoublePi.root";
   
   files.at(0)="/run/media/epic/Data/Muons/Grape-10x100/Paper/recoGL.root";
   */
   //files.at(0)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Muons.root";
   
   files.at(0)="/run/media/epic/Data/Background/JPsi/OLD/*.root";
   files.at(1)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Electrons.root";
   files.at(2)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Pions.root";

   //files.at(2)="/run/media/epic/Data/Background/Pions/*.root";

   
   /*
   files.at(0)="/Data/Muons/ToF/Electrons.root";
   files.at(1)="/Data/Muons/Epic-10x275/recoEL0S.root";
   files.at(2)="/Data/Tau/EpIC/tcs/tau_tcs_hist.root";
   */
   TF1 *upperbondE = new TF1("upperbondE", "5/(x+2)", 0.001, 24.0);
   upperbondE->SetLineColor(kRed);
   upperbondE->SetLineWidth(1);

   TF1 *upperbondH = new TF1("upperbondH", "3/x+0.1",  0.001, 24.0); 
   upperbondH->SetLineColor(kRed);
   upperbondH->SetLineWidth(1);
      
   TF1 *lowerbondH = new TF1("lowerbondH", "0.35/x-0.25/(x*x)",  0.001, 24.0); 
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
      TTreeReaderArray<float> EcalBarrelEng(tree_reader, "EcalBarrelClusters.energy");
      TTreeReaderArray<unsigned int> EcalBarrelShPB(tree_reader, "EcalBarrelClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalBarrelShPE(tree_reader, "EcalBarrelClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalBarrelShParameters(tree_reader, "_EcalBarrelClusters_shapeParameters");


      TTreeReaderArray<unsigned int> simuAssocEcalBarrelImaging(tree_reader, "EcalBarrelImagingClusterAssociations.simID");
      TTreeReaderArray<float> EcalBarrelImagingEng(tree_reader, "EcalBarrelImagingClusters.energy");
      TTreeReaderArray<unsigned int> EcalBarrelImagingShPB(tree_reader, "EcalBarrelImagingClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalBarrelImagingShPE(tree_reader, "EcalBarrelImagingClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalBarrelImagingShParameters(tree_reader, "_EcalBarrelImagingClusters_shapeParameters");

      TTreeReaderArray<unsigned int> simuAssocEcalBarrelScFi(tree_reader, "EcalBarrelScFiClusterAssociations.simID");
      TTreeReaderArray<float> EcalBarrelScFiEng(tree_reader, "EcalBarrelScFiClusters.energy");
      TTreeReaderArray<unsigned int> EcalBarrelScFiShPB(tree_reader, "EcalBarrelScFiClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalBarrelScFiShPE(tree_reader, "EcalBarrelScFiClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalBarrelScFiShParameters(tree_reader, "_EcalBarrelScFiClusters_shapeParameters");

      TTreeReaderArray<unsigned int> simuAssocEcalEndcapP(tree_reader, "EcalEndcapPClusterAssociations.simID");
      TTreeReaderArray<float> EcalEndcapPEng(tree_reader, "EcalEndcapPClusters.energy");
      TTreeReaderArray<unsigned int> EcalEndcapPShPB(tree_reader, "EcalEndcapPClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalEndcapPShPE(tree_reader, "EcalEndcapPClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalEndcapPShParameters(tree_reader, "_EcalEndcapPClusters_shapeParameters");

      TTreeReaderArray<unsigned int> simuAssocEcalEndcapN(tree_reader, "EcalEndcapNClusterAssociations.simID");
      TTreeReaderArray<float> EcalEndcapNEng(tree_reader, "EcalEndcapNClusters.energy");
      TTreeReaderArray<unsigned int> EcalEndcapNShPB(tree_reader, "EcalEndcapNClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalEndcapNShPE(tree_reader, "EcalEndcapNClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalEndcapNShParameters(tree_reader, "_EcalEndcapNClusters_shapeParameters");

      // Hcal Information
      TTreeReaderArray<unsigned int> simuAssocHcalBarrel(tree_reader, "HcalBarrelClusterAssociations.simID");
      TTreeReaderArray<float> HcalBarrelEng(tree_reader, "HcalBarrelClusters.energy");
      TTreeReaderArray<unsigned int> HcalBarrelShPB(tree_reader, "HcalBarrelClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> HcalBarrelShPE(tree_reader, "HcalBarrelClusters.shapeParameters_end");
      TTreeReaderArray<float> HcalBarrelShParameters(tree_reader, "_HcalBarrelClusters_shapeParameters");

      TTreeReaderArray<unsigned int> simuAssocHcalEndcapP(tree_reader, "HcalEndcapPInsertClusterAssociations.simID");
      TTreeReaderArray<float> HcalEndcapPEng(tree_reader, "HcalEndcapPInsertClusters.energy");
      TTreeReaderArray<unsigned int> HcalEndcapPShPB(tree_reader, "HcalEndcapPInsertClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> HcalEndcapPShPE(tree_reader, "HcalEndcapPInsertClusters.shapeParameters_end");
      TTreeReaderArray<float> HcalEndcapPShParameters(tree_reader, "_HcalEndcapPInsertClusters_shapeParameters");

      TTreeReaderArray<unsigned int> simuAssocLFHcal(tree_reader, "LFHCALClusterAssociations.simID");
      TTreeReaderArray<float> LFHcalEng(tree_reader, "LFHCALClusters.energy");
      TTreeReaderArray<unsigned int> LFHcalShPB(tree_reader, "LFHCALClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> LFHcalShPE(tree_reader, "LFHCALClusters.shapeParameters_end");
      TTreeReaderArray<float> LFHcalShParameters(tree_reader, "_LFHCALClusters_shapeParameters");

      TTreeReaderArray<unsigned int> simuAssocHcalEndcapN(tree_reader, "HcalEndcapNClusterAssociations.simID");
      TTreeReaderArray<float> HcalEndcapNEng(tree_reader, "HcalEndcapNClusters.energy");
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
      AllParticTheta[File] = new TH1D(Form("AllParticTheta%s",name.c_str()),Form("AllParticTheta%s",name.c_str()),50,0,180);
      AllParticPhi[File]= new TH1D(Form("AllParticPhi%s",name.c_str()),Form("AllParticPhi%s",name.c_str()),30,-180,180);
      AllParticEnergy[File]= new TH1D(Form("AllParticEnergy%s",name.c_str()),Form("AllParticEnergy%s",name.c_str()),50,0,10);
      
      NotFoundParticTheta[File] = new TH1D(Form("NotFoundParticTheta%s",name.c_str()),Form("NotFoundParticTheta%s",name.c_str()),50,0,180);
      NotFoundParticPhi[File]= new TH1D(Form("NotFoundParticPhi%s",name.c_str()),Form("NotFoundParticPhi%s",name.c_str()),30,-180,180);
      NotFoundParticEnergy[File]= new TH1D(Form("NotFoundParticEnergy%s",name.c_str()),Form("NotFoundParticEnergy%s",name.c_str()),50,0,10);
      //==================================//
      ECalEnergyHist[File]= new TH1D(Form("ECalEnergyHist%s",name.c_str()),Form("ECalEnergyHist%s",name.c_str()),50,0,15);
      ECalEnergyMomHist[File]= new TH1D(Form("ECalEnergyMomHist%s",name.c_str()),Form("ECalEnergyMomHist%s",name.c_str()),50,0,0.2);
      ECalEnergyvsMomHist[File]= new TH2D(Form("ECalEnergyvsMomHist%s",name.c_str()),Form("ECalEnergyvsMomHist%s",name.c_str()),50,0,22,50,0,2);

      HCalEnergyHist[File]= new TH1D(Form("HCalEnergyHist%s",name.c_str()),Form("HCalEnergyHist%s",name.c_str()),50,0,15);
      HCalEnergyMomHist[File]= new TH1D(Form("HCalEnergyMomHist%s",name.c_str()),Form("HCalEnergyMomHist%s",name.c_str()),50,0,4);
      HCalEnergyvsMomHist[File]= new TH2D(Form("HCalEnergyvsMomHist%s",name.c_str()),Form("HCalEnergyvsMomHist%s",name.c_str()),50,0,22,50,0,2);
      //==================================//
      ToFTimeHist[File]= new TH1D(Form("ToFTimeHist%s",name.c_str()),Form("ToFTimeHist%s",name.c_str()),50,0,15);
      ToFMassHist[File]= new TH1D(Form("ToFMassHist%s",name.c_str()),Form("ToFMassHist%s",name.c_str()),50,0,0.3);
      ToFMassHist2[File]= new TH1D(Form("ToFMassHist2%s",name.c_str()),Form("ToFMassHist2%s",name.c_str()),50,0,0.3);
      ToFMassHist10[File]= new TH1D(Form("ToFMassHist10%s",name.c_str()),Form("ToFMassHist10%s",name.c_str()),50,0,0.3);
      ToFMassHist40[File]= new TH1D(Form("ToFMassHist40%s",name.c_str()),Form("ToFMassHist40%s",name.c_str()),50,0,0.3);
      //==================================//
      DoubleToFMassHist[File]= new TH1D(Form("DoubleToFMassHist%s",name.c_str()),Form("ToFMassHist%s",name.c_str()),50,0,0.3);
      DoubleToFMassHist2[File]= new TH1D(Form("DoubleToFMassHist2%s",name.c_str()),Form("ToFMassHist2%s",name.c_str()),50,0,0.3);
      DoubleToFMassHist10[File]= new TH1D(Form("DoubleToFMassHist10%s",name.c_str()),Form("ToFMassHist10%s",name.c_str()),50,0,0.3);
      DoubleToFMassHist40[File]= new TH1D(Form("DoubleToFMassHist40%s",name.c_str()),Form("ToFMassHist40%s",name.c_str()),50,0,0.3);
      vector<vector<float>> ParamsVector= {{100,10,40},{100,7,35},{100,-0.001,0.001},{100,-0.001,0.004},{100,-0.2,0.2},{100,-1.5,1.5},{50,-3.5,3.5}};
      for(int i=0; i<7; i++){
         EcalShapeHist[i][File] = new TH1D(Form("EcalShape%d%s", i, name.c_str()), Form("EcalShape%d%s", i, name.c_str()), ParamsVector[i][0], ParamsVector[i][1], ParamsVector[i][2]);
         HcalShapeHist[i][File] = new TH1D(Form("HcalShape%d%s", i, name.c_str()), Form("HcalShape%d%s", i, name.c_str()), ParamsVector[i][0], ParamsVector[i][1], ParamsVector[i][2]);
      }




      int eventID=0;
      double FoundParticles=0;
      double particscount=0;
      double BadPDG=0;
      double aftercuts=0;

      while(tree_reader.Next()){
         eventID++;
         if(eventID>10) break;
         
         //if(File==0) if(eventID>3800) break;

         

         if(eventID%20000==0) cout<<"File "<<name<<" and event number... "<<eventID<<endl;
         ECalEnergy=0;
         HCalEnergy=0;
         ECalNumber=0;
         HCalNumber=0;
         EoverP=0;
         Momentum=0;
         EcalShape.clear();
         HcalShape.clear();

         //////////////////////
         //Number of clusters in the event
         //////////////////////

         NumberB0Barrel[File]->Fill(B0Eng.GetSize());
         NumberEcalBarrel[File]->Fill(EcalBarrelEng.GetSize());
         NumberEcalEndcapP[File]->Fill(EcalEndcapPEng.GetSize());
         NumberEcalEndcapN[File]->Fill(EcalEndcapNEng.GetSize());
         NumberHcalBarrel[File]->Fill(HcalBarrelEng.GetSize());
         NumberHcalEndcapN[File]->Fill(HcalEndcapNEng.GetSize());
         NumberHcalEndcapP[File]->Fill(HcalEndcapPEng.GetSize());
         NumberLFHcal[File]->Fill(LFHcalEng.GetSize());
         
         //ECalNumber=EcalBarrelEng.GetSize()+EcalEndcapPEng.GetSize()+EcalEndcapNEng.GetSize();
         //HCalNumber=HcalBarrelEng.GetSize()+HcalEndcapNEng.GetSize()+HcalEndcapPEng.GetSize()+LFHcalEng.GetSize();
         /*
         vector<double> timeToF(2),timeToF2(2),timeToF10(2),timeToF40(2);
         vector<double> LengthToF(2);
         vector<double> Moment(2);
         */
         int id=0;
         for(int particle=0; particle<trackEng.GetSize();particle++)
         {
            particscount++;
            //Obligatory Cuts 
            double mass;
            if(File==0) mass=MuonMass;
            else if(File==1) mass=ElectronMass;
            else if(File==2) mass=PionMass;

            int Found=0;
            TLorentzVector Partic;
            Partic.SetPxPyPzE(trackMomX[particle],trackMomY[particle],trackMomZ[particle],trackEng[particle]);
            /*
               auto [t,t2,t10,t40,l]=ToFFastSim(Partic,trackCharge[particle],mass);
               ToFTimeHist[File]->Fill(t);
               timeToF[id]=t;
               timeToF2[id]=t2;
               timeToF10[id]=t10;
               timeToF40[id]=t40;

               LengthToF[id]=l;
               Moment[id]=Partic.P();
               if(timeToF[0]!=0 && id==0) id++;
               else if(timeToF[1]!=0 && id==1 && File!=2)
               {
                  double masslepton=DiLeptonMass(LengthToF[0], LengthToF[1], Moment[0], Moment[1],timeToF[0]-timeToF[1]);
                  DoubleToFMassHist[File]->Fill(sqrt(masslepton));
                  double masslepton2=DiLeptonMass(LengthToF[0], LengthToF[1], Moment[0], Moment[1],timeToF2[0]-timeToF2[1]);
                  DoubleToFMassHist2[File]->Fill(sqrt(masslepton2));

                  double masslepton10=DiLeptonMass(LengthToF[0], LengthToF[1], Moment[0], Moment[1],timeToF10[0]-timeToF10[1]);
                  DoubleToFMassHist10[File]->Fill(sqrt(masslepton10));
                  double masslepton40=DiLeptonMass(LengthToF[0], LengthToF[1], Moment[0], Moment[1],timeToF40[0]-timeToF40[1]);
                  DoubleToFMassHist40[File]->Fill(sqrt(masslepton40));

                  //cout<<masslepton40<<endl;
               } 
               
               if(timeToF[0]!=0)
               {
                  double masslepton=LeptonMass(l, Partic.P(), t);
                  ToFMassHist[File]->Fill(sqrt(masslepton));

                  double masslepton2=LeptonMass(l, Partic.P(), t2);
                  ToFMassHist2[File]->Fill(sqrt(masslepton2));

                  double masslepton10=LeptonMass(l, Partic.P(), t10);
                  ToFMassHist10[File]->Fill(sqrt(masslepton10));

                  double masslepton40=LeptonMass(l, Partic.P(), t40);
                  ToFMassHist40[File]->Fill(sqrt(masslepton40));

               }
            */
            if(Partic.Theta()*DEG>178) continue;
         
            PDG[File]->Fill(trackPDG[particle]);
            AllParticEnergy[File]->Fill(Partic.E());
            AllParticTheta[File]->Fill(Partic.Theta()*DEG);
            AllParticPhi[File]->Fill(Partic.Phi()*DEG);
            
           
           //Ecal Energy Search
            NumberParticles[File]->Fill(simuAssocEcalBarrel.GetSize());
            int simuID = simuAssoc[particle];
             cout<<simuID<<" blabla "<<endl;

            /*
               auto [ECalEnergy,ECalNumber,HCalEnergy,HCalNumber] = Calorimeters(Partic, simuID, EcalBarrelEng, EcalEndcapPEng, EcalEndcapNEng, HcalBarrelEng, HcalEndcapPEng, LFHcalEng, HcalEndcapNEng, B0Eng, EcalBarrelImagingEng, EcalBarrelScFiEng, 
               simuAssocEcalBarrel, simuAssocEcalEndcapP, simuAssocEcalEndcapN, simuAssocHcalBarrel, simuAssocHcalEndcapP, simuAssocLFHcal, simuAssocHcalEndcapN, simuAssocB0, simuAssocEcalBarrelImaging, simuAssocEcalBarrelScFi,B0ShPB,B0ShPE,B0ShParameters);
            */
            //////////////////////
            // Collect energies and shapes from all ECal detectors
            //////////////////////

            double maxEcalEnergy = 0;
            vector<float> maxEcalShape;
            
            auto [Energy,Number,Shape] = Calorimeternew( simuID, EcalBarrelEng, simuAssocEcalBarrel,  EcalBarrelShPB, EcalBarrelShPE,EcalBarrelShParameters);
            ECalEnergy+=Energy;
            ECalNumber+=Number;
            if(Energy > maxEcalEnergy && !Shape.empty() && Shape[0] != 0) {
                maxEcalEnergy = Energy;
                maxEcalShape = Shape;
            }

            auto [EnergyImaging,NumberImaging,ShapeImaging] = Calorimeternew( simuID, EcalBarrelImagingEng, simuAssocEcalBarrelImaging,  EcalBarrelImagingShPB, EcalBarrelImagingShPE,EcalBarrelImagingShParameters);
            ECalEnergy+=EnergyImaging;
            ECalNumber+=NumberImaging;
            if(EnergyImaging > maxEcalEnergy && !ShapeImaging.empty() && ShapeImaging[0] != 0) {
                maxEcalEnergy = EnergyImaging;
                maxEcalShape = ShapeImaging;
            }

            auto [EnergyScFi,NumberScFi,ShapeScFi] = Calorimeternew( simuID, EcalBarrelScFiEng, simuAssocEcalBarrelScFi,  EcalBarrelScFiShPB, EcalBarrelScFiShPE,EcalBarrelScFiShParameters);
            ECalEnergy+=EnergyScFi;
            ECalNumber+=NumberScFi;
            if(EnergyScFi > maxEcalEnergy && !ShapeScFi.empty() && ShapeScFi[0] != 0) {
                maxEcalEnergy = EnergyScFi;
                maxEcalShape = ShapeScFi;
            }

            auto [EnergyEndcapP,NumberEndcapP,ShapeEndcapP] = Calorimeternew( simuID, EcalEndcapPEng, simuAssocEcalEndcapP,  EcalEndcapPShPB, EcalEndcapPShPE,EcalEndcapPShParameters);
            ECalEnergy+=EnergyEndcapP;
            ECalNumber+=NumberEndcapP;
            if(EnergyEndcapP > maxEcalEnergy && !ShapeEndcapP.empty() && ShapeEndcapP[0] != 0) {
                maxEcalEnergy = EnergyEndcapP;
                maxEcalShape = ShapeEndcapP;
            }

            auto [EnergyEndcapN,NumberEndcapN,ShapeEndcapN] = Calorimeternew( simuID, EcalEndcapNEng, simuAssocEcalEndcapN,  EcalEndcapNShPB, EcalEndcapNShPE,EcalEndcapNShParameters);
            ECalEnergy+=EnergyEndcapN;
            ECalNumber+=NumberEndcapN;
            if(EnergyEndcapN > maxEcalEnergy && !ShapeEndcapN.empty() && ShapeEndcapN[0] != 0) {
                maxEcalEnergy = EnergyEndcapN;
                maxEcalShape = ShapeEndcapN;
            }

            auto [EnergyB0,NumberB0,ShapeB0] = Calorimeternew( simuID, B0Eng, simuAssocB0,  B0ShPB, B0ShPE,B0ShParameters);
            ECalEnergy+=EnergyB0;
            ECalNumber+=NumberB0;
            if(EnergyB0 > maxEcalEnergy && !ShapeB0.empty() && ShapeB0[0] != 0) {
                maxEcalEnergy = EnergyB0;
                maxEcalShape = ShapeB0;
            }
            
            // Assign shape from detector with highest energy
            if(!maxEcalShape.empty()) EcalShape = maxEcalShape;


            if(ECalEnergy!=0)
            {
               EnergyEcal[File]->Fill(ECalEnergy);
               Found=1;
            }
            //////////////////////           
            //Hcal Energy Search
            //////////////////////

            double maxHcalEnergy = 0;
            vector<float> maxHcalShape;
            
            auto [EnergyHcalBarrel,NumberHcalBarrel,ShapeHcalBarrel] = Calorimeternew( simuID, HcalBarrelEng, simuAssocHcalBarrel,  HcalBarrelShPB, HcalBarrelShPE,HcalBarrelShParameters);
            HCalEnergy+=EnergyHcalBarrel;
            HCalNumber+=NumberHcalBarrel;
            if(EnergyHcalBarrel > maxHcalEnergy && !ShapeHcalBarrel.empty() && ShapeHcalBarrel[0] != 0) {
                maxHcalEnergy = EnergyHcalBarrel;
                maxHcalShape = ShapeHcalBarrel;
            }

            auto [EnergyHcalEndcapP,NumberHcalEndcapP,ShapeHcalEndcapP] = Calorimeternew( simuID, HcalEndcapPEng, simuAssocHcalEndcapP,  HcalEndcapPShPB, HcalEndcapPShPE,HcalEndcapPShParameters);
            HCalEnergy+=EnergyHcalEndcapP;
            HCalNumber+=NumberHcalEndcapP;
            if(EnergyHcalEndcapP > maxHcalEnergy && !ShapeHcalEndcapP.empty() && ShapeHcalEndcapP[0] != 0) {
                maxHcalEnergy = EnergyHcalEndcapP;
                maxHcalShape = ShapeHcalEndcapP;
            }

            auto [EnergyLFHcal,NumberLFHcal,ShapeLFHcal] = Calorimeternew( simuID, LFHcalEng, simuAssocLFHcal,  LFHcalShPB, LFHcalShPE,LFHcalShParameters);
            HCalEnergy+=EnergyLFHcal;
            HCalNumber+=NumberLFHcal;
            if(EnergyLFHcal > maxHcalEnergy && !ShapeLFHcal.empty() && ShapeLFHcal[0] != 0) {
                maxHcalEnergy = EnergyLFHcal;
                maxHcalShape = ShapeLFHcal;
            }

            auto [EnergyHcalEndcapN,NumberHcalEndcapN,ShapeHcalEndcapN] = Calorimeternew( simuID, HcalEndcapNEng, simuAssocHcalEndcapN,  HcalEndcapNShPB, HcalEndcapNShPE,HcalEndcapNShParameters);
            HCalEnergy+=EnergyHcalEndcapN;
            HCalNumber+=NumberHcalEndcapN;
            if(EnergyHcalEndcapN > maxHcalEnergy && !ShapeHcalEndcapN.empty() && ShapeHcalEndcapN[0] != 0) {
                maxHcalEnergy = EnergyHcalEndcapN;
                maxHcalShape = ShapeHcalEndcapN;
            }
            
            // Assign shape from detector with highest energy
            if(!maxHcalShape.empty()) HcalShape = maxHcalShape;
           
            
            if(HCalEnergy!=0)
            {
               EnergyHcal[File]->Fill(HCalEnergy);
               Found=1;
            }
            NumberEcal[File]->Fill(ECalNumber);
            NumberHcal[File]->Fill(HCalNumber);

            if(abs(Partic.Eta())<1.3 && abs(Partic.Eta())>1) continue;
            FoundParticles+=Found;
            if(!(trackPDG[particle]==0 || abs(trackPDG[particle])==13)) continue;
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
            
            for(int i=0; i<7; i++){
               if(!EcalShape.empty()) EcalShapeHist[i][File]->Fill(EcalShape[i]);
               if(!HcalShape.empty()) HcalShapeHist[i][File]->Fill(HcalShape[i]);
            }
            
            if(File==0) IsMuon=1;  
            else IsMuon=0; 
            
            if(HcalEoverP>upperbondH->Eval(Momentum)) continue;
            if(HcalEoverP<lowerbondH->Eval(Momentum)) continue;
            if(EcalEoverP>upperbondE->Eval(Momentum)) continue;
            
            aftercuts++;
            
            MLDataTree->Fill();  
            

         } 

      }
      

      cout<<"==========================="<<endl;
      cout<<"End of "<< name << " file"<<endl;
      cout<<"Number of events: "<<eventID<<endl;
      cout<<"Found particles: "<<FoundParticles<<"   All particles: "<<particscount<<endl;
      cout<<"Found particles: "<<FoundParticles<<"   After cuts particles: "<<aftercuts<<endl;

      cout<<"Found Ratio: "<<FoundParticles*100/particscount<<'%'<<endl;
      cout<<"After Cuts Ratio: "<<aftercuts*100/FoundParticles<<'%'<<endl;

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
    leg->AddEntry(EnergyEcal[0],"Muons","l");
    leg->AddEntry(EnergyEcal[1],"Electrons","l");
    leg->AddEntry(EnergyEcal[2],"Pions","l");




   TLegend* leg1 = new TLegend(0.78, 0.8, 0.95, 0.95);
    leg1->SetBorderSize(0);
    leg1->SetNColumns(1);
    leg1->SetColumnSeparation(0.1);
    leg1->SetEntrySeparation(0.1);
    leg1->SetMargin(0.15);
    leg1->SetTextFont(42);
    leg1->SetTextSize(0.05);
    leg1->AddEntry(EnergyEcal[0],"Muons ","l");
    leg1->AddEntry(EnergyEcal[1],"Electrons","l");
    leg1->AddEntry(EnergyEcal[2],"Pions","l");

    TLegend* leg2 = new TLegend(0.58, 0.2, 0.85, 0.45);
    leg2->SetBorderSize(0);
    leg2->SetNColumns(1);
    leg2->SetColumnSeparation(0.1);
    leg2->SetEntrySeparation(0.1);
    leg2->SetMargin(0.15);
    leg2->SetTextFont(42);
    leg2->SetTextSize(0.05);
    leg2->AddEntry(EnergyEcal[0],"Muons","l");
    leg2->AddEntry(EnergyEcal[1],"Electrons","l");

    TLegend* leg3 = new TLegend(0.58, 0.6, 0.85, 0.85);
    leg3->SetBorderSize(0);
    leg3->SetNColumns(1);
    leg3->SetColumnSeparation(0.1);
    leg3->SetEntrySeparation(0.1);
    leg3->SetMargin(0.15);
    leg3->SetTextFont(42);
    leg3->SetTextSize(0.05);
    leg3->AddEntry(EnergyEcal[0],"Muons","l");
    leg3->AddEntry(EnergyEcal[1],"Electrons","l");

   

   c1.SaveAs("Plots/UsefulToF.pdf[");


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
   c1.SaveAs("Plots/UsefulToF.pdf");

   

   c1.Clear();
      EnergyHcal[0]->Scale(1./EnergyHcal[0]->Integral());
      EnergyHcal[1]->Scale(1./EnergyHcal[1]->Integral());
      EnergyHcal[2]->Scale(1./EnergyHcal[2]->Integral());
      EnergyHcal[0]->SetLineColor(kRed);
      EnergyHcal[1]->SetLineColor(kBlue);
      EnergyHcal[2]->SetLineColor(kGreen);
      EnergyHcal[0]->Draw("HIST");
      EnergyHcal[1]->Draw("HIST SAME");
      EnergyHcal[2]->Draw("HIST SAME");
      leg->Draw();

   c1.SaveAs("Plots/UsefulToF.pdf");

   
   c1.Clear();
      NumberEcal[0]->Scale(1./NumberEcal[0]->Integral());
      NumberEcal[1]->Scale(1./NumberEcal[1]->Integral());
      NumberEcal[2]->Scale(1./NumberEcal[2]->Integral());
      NumberEcal[0]->SetLineColor(kRed);
      NumberEcal[1]->SetLineColor(kBlue);
      NumberEcal[2]->SetLineColor(kGreen);
      NumberEcal[1]->Draw("HIST");
      NumberEcal[0]->Draw("HIST SAME");
      NumberEcal[2]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/UsefulToF.pdf");

   c1.Clear();
      NumberHcal[0]->Scale(1./NumberHcal[0]->Integral());
      NumberHcal[1]->Scale(1./NumberHcal[1]->Integral());
      NumberHcal[2]->Scale(1./NumberHcal[2]->Integral());
      NumberHcal[0]->SetLineColor(kRed);
      NumberHcal[1]->SetLineColor(kBlue);
      NumberHcal[2]->SetLineColor(kGreen);
      NumberHcal[1]->Draw("HIST");
      NumberHcal[0]->Draw("HIST SAME");
      NumberHcal[2]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/UsefulToF.pdf");
   c1.SaveAs("Plots/UsefulToF.pdf]");


   c1.SaveAs("Plots/AdditionalToF.pdf[");

   

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
   c1.SaveAs("Plots/AdditionalToF.pdf");
   gPad->SetLogz(0);

   c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      ECalEnergyHist[0]->Scale(1./ECalEnergyHist[0]->Integral());
      ECalEnergyHist[1]->Scale(1./ECalEnergyHist[1]->Integral());
      ECalEnergyHist[2]->Scale(1./ECalEnergyHist[2]->Integral());

      ECalEnergyHist[0]->SetLineColor(kRed);
      ECalEnergyHist[1]->SetLineColor(kBlue);
      ECalEnergyHist[2]->SetLineColor(kGreen);

      ECalEnergyHist[1]->Draw("HIST");
      ECalEnergyHist[0]->Draw("HIST SAME");
      ECalEnergyHist[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(2);
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
   c1.SaveAs("Plots/AdditionalToF.pdf");
   c1.Clear();
   c1.Divide(2,2);
   
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
   c1.SaveAs("Plots/AdditionalToF.pdf");
   gPad->SetLogz(0);

   c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      HCalEnergyHist[0]->Scale(1./HCalEnergyHist[0]->Integral());
      HCalEnergyHist[1]->Scale(1./HCalEnergyHist[1]->Integral());
      HCalEnergyHist[2]->Scale(1./HCalEnergyHist[2]->Integral());

      HCalEnergyHist[0]->SetLineColor(kRed);
      HCalEnergyHist[1]->SetLineColor(kBlue);
      HCalEnergyHist[2]->SetLineColor(kGreen);

      HCalEnergyHist[1]->Draw("HIST");
      HCalEnergyHist[0]->Draw("HIST SAME");
      HCalEnergyHist[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(2);
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
   c1.SaveAs("Plots/AdditionalToF.pdf");


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
   c1.SaveAs("Plots/AdditionalToF.pdf");
   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      AllParticTheta[1]->SetLineColor(kBlue);
      NotFoundParticTheta[1]->SetLineColor(kRed);
      AllParticTheta[1]->Draw();
      NotFoundParticTheta[1]->Draw("colz");
   c1.cd(2);
      AllParticPhi[1]->SetLineColor(kBlue);
      NotFoundParticPhi[1]->SetLineColor(kRed);
      AllParticPhi[1]->SetMinimum(0);

      AllParticPhi[1]->Draw();
      NotFoundParticPhi[1]->Draw("colz");
   c1.cd(3); 
      AllParticEnergy[1]->SetLineColor(kBlue);
      NotFoundParticEnergy[1]->SetLineColor(kRed);
      AllParticEnergy[1]->Draw();
      NotFoundParticEnergy[1]->Draw("colz");   
   c1.SaveAs("Plots/AdditionalToF.pdf");
   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      AllParticTheta[2]->SetLineColor(kBlue);
      NotFoundParticTheta[2]->SetLineColor(kRed);
      AllParticTheta[2]->Draw();
      NotFoundParticTheta[2]->Draw("colz");
   c1.cd(2);
      AllParticPhi[2]->SetLineColor(kBlue);
      NotFoundParticPhi[2]->SetLineColor(kRed);
      AllParticPhi[2]->SetMinimum(0);

      AllParticPhi[2]->Draw();
      NotFoundParticPhi[2]->Draw("colz");
   c1.cd(3); 
      AllParticEnergy[2]->SetLineColor(kBlue);
      NotFoundParticEnergy[2]->SetLineColor(kRed);
      AllParticEnergy[2]->Draw();
      NotFoundParticEnergy[2]->Draw("colz");   
   c1.SaveAs("Plots/AdditionalToF.pdf");


   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      NumberEcalBarrel[0]->Scale(1./NumberEcalBarrel[0]->Integral());
      NumberEcalBarrel[1]->Scale(1./NumberEcalBarrel[1]->Integral());
      NumberEcalBarrel[0]->SetLineColor(kRed);
      NumberEcalBarrel[1]->SetLineColor(kBlue);
      NumberEcalBarrel[0]->Draw("HIST");
      NumberEcalBarrel[1]->Draw("HIST SAME");
      NumberEcalBarrel[2]->Scale(1./NumberEcalBarrel[2]->Integral());
      NumberEcalBarrel[2]->SetLineColor(kGreen);
      NumberEcalBarrel[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(2);
      NumberB0Barrel[0]->Scale(1./NumberB0Barrel[0]->Integral());
      NumberB0Barrel[1]->Scale(1./NumberB0Barrel[1]->Integral());
      NumberB0Barrel[0]->SetLineColor(kRed);
      NumberB0Barrel[1]->SetLineColor(kBlue);
      NumberB0Barrel[0]->Draw("HIST");
      NumberB0Barrel[1]->Draw("HIST SAME");
      NumberB0Barrel[2]->Scale(1./NumberB0Barrel[2]->Integral());
      NumberB0Barrel[2]->SetLineColor(kGreen);
      NumberB0Barrel[2]->Draw("HIST SAME");
   c1.cd(3);
      NumberEcalEndcapP[0]->Scale(1./NumberEcalEndcapP[0]->Integral());
      NumberEcalEndcapP[1]->Scale(1./NumberEcalEndcapP[1]->Integral());
      NumberEcalEndcapP[0]->SetLineColor(kRed);
      NumberEcalEndcapP[1]->SetLineColor(kBlue);
      NumberEcalEndcapP[0]->Draw("HIST");
      NumberEcalEndcapP[1]->Draw("HIST SAME");
      NumberEcalEndcapP[2]->Scale(1./NumberEcalEndcapP[2]->Integral());
      NumberEcalEndcapP[2]->SetLineColor(kGreen);
      NumberEcalEndcapP[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(4);
      NumberEcalEndcapN[0]->Scale(1./NumberEcalEndcapN[0]->Integral());
      NumberEcalEndcapN[1]->Scale(1./NumberEcalEndcapN[1]->Integral());
      NumberEcalEndcapN[0]->SetLineColor(kRed);
      NumberEcalEndcapN[1]->SetLineColor(kBlue);
      NumberEcalEndcapN[0]->Draw("HIST");
      NumberEcalEndcapN[1]->Draw("HIST SAME");
      NumberEcalEndcapN[2]->Scale(1./NumberEcalEndcapN[2]->Integral());
      NumberEcalEndcapN[2]->SetLineColor(kGreen);
      NumberEcalEndcapN[2]->Draw("HIST SAME");
      leg->Draw();      
   c1.SaveAs("Plots/AdditionalToF.pdf");

   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      NumberHcalBarrel[0]->Scale(1./NumberHcalBarrel[0]->Integral());
      NumberHcalBarrel[1]->Scale(1./NumberHcalBarrel[1]->Integral());
      NumberHcalBarrel[0]->SetLineColor(kRed);
      NumberHcalBarrel[1]->SetLineColor(kBlue);
      NumberHcalBarrel[0]->Draw("HIST");
      NumberHcalBarrel[1]->Draw("HIST SAME");
      NumberHcalBarrel[2]->Scale(1./NumberHcalBarrel[2]->Integral());
      NumberHcalBarrel[2]->SetLineColor(kGreen);
      NumberHcalBarrel[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(2);
      NumberLFHcal[0]->Scale(1./NumberLFHcal[0]->Integral());
      NumberLFHcal[1]->Scale(1./NumberLFHcal[1]->Integral());
      NumberLFHcal[0]->SetLineColor(kRed);
      NumberLFHcal[1]->SetLineColor(kBlue);
      NumberLFHcal[0]->Draw("HIST");
      NumberLFHcal[1]->Draw("HIST SAME");
      NumberLFHcal[2]->Scale(1./NumberLFHcal[2]->Integral());
      NumberLFHcal[2]->SetLineColor(kGreen);
      NumberLFHcal[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(3);
      NumberHcalEndcapP[0]->Scale(1./NumberHcalEndcapP[0]->Integral());
      NumberHcalEndcapP[1]->Scale(1./NumberHcalEndcapP[1]->Integral());
      NumberHcalEndcapP[0]->SetLineColor(kRed);
      NumberHcalEndcapP[1]->SetLineColor(kBlue);
      NumberHcalEndcapP[0]->Draw("HIST");
      NumberHcalEndcapP[1]->Draw("HIST SAME");
      NumberHcalEndcapP[2]->Scale(1./NumberHcalEndcapP[2]->Integral());
      NumberHcalEndcapP[2]->SetLineColor(kGreen);
      NumberHcalEndcapP[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(4);
      NumberHcalEndcapN[0]->Scale(1./NumberHcalEndcapN[0]->Integral());
      NumberHcalEndcapN[1]->Scale(1./NumberHcalEndcapN[1]->Integral());
      NumberHcalEndcapN[0]->SetLineColor(kRed);
      NumberHcalEndcapN[1]->SetLineColor(kBlue);
      NumberHcalEndcapN[0]->Draw("HIST");
      NumberHcalEndcapN[1]->Draw("HIST SAME");
      NumberHcalEndcapN[2]->Scale(1./NumberHcalEndcapN[2]->Integral());
      NumberHcalEndcapN[2]->SetLineColor(kGreen);
      NumberHcalEndcapN[2]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/AdditionalToF.pdf");


   c1.Clear();
      PDG[0]->Scale(1./PDG[0]->Integral());
      PDG[1]->Scale(1./PDG[1]->Integral());
      PDG[2]->Scale(1./PDG[2]->Integral());
      PDG[0]->SetLineColor(kRed);
      PDG[1]->SetLineColor(kBlue);
      PDG[2]->SetLineColor(kGreen);
      PDG[1]->Draw("HIST");
      PDG[0]->Draw("HIST SAME");
      PDG[2]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/AdditionalToF.pdf");

   c1.Clear();
      NumberParticles[0]->Scale(1./NumberParticles[0]->Integral());
      NumberParticles[1]->Scale(1./NumberParticles[1]->Integral());
      NumberParticles[2]->Scale(1./NumberParticles[2]->Integral());
      NumberParticles[0]->SetLineColor(kRed);
      NumberParticles[1]->SetLineColor(kBlue);
      NumberParticles[2]->SetLineColor(kGreen);
      NumberParticles[1]->Draw("HIST");
      NumberParticles[0]->Draw("HIST SAME");
      NumberParticles[2]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/AdditionalToF.pdf");

     // Shape histograms
   c1.Clear();
   c1.Divide(2,4);
   for(int i=0; i<7; i++){
      c1.cd(i+1);
      EcalShapeHist[i][0]->SetLineColor(kRed);
      EcalShapeHist[i][1]->SetLineColor(kBlue);
      EcalShapeHist[i][2]->SetLineColor(kGreen);
      EcalShapeHist[i][0]->Draw("HIST");
      EcalShapeHist[i][1]->Draw("HIST SAME");
      EcalShapeHist[i][2]->Draw("HIST SAME");
   }
   c1.SaveAs("Plots/AdditionalToF.pdf");
   
   c1.Clear();
   c1.Divide(2,4);
   for(int i=0; i<7; i++){
      c1.cd(i+1);
      HcalShapeHist[i][0]->SetLineColor(kRed);
      HcalShapeHist[i][1]->SetLineColor(kBlue);
      HcalShapeHist[i][2]->SetLineColor(kGreen);
      HcalShapeHist[i][0]->Draw("HIST");
      HcalShapeHist[i][1]->Draw("HIST SAME");
      HcalShapeHist[i][2]->Draw("HIST SAME");
   }
   c1.SaveAs("Plots/AdditionalToF.pdf");
   
   c1.Clear();
   ToFTimeHist[0]->Scale(1./ToFTimeHist[0]->Integral());
   ToFTimeHist[1]->Scale(1./ToFTimeHist[1]->Integral());
   ToFTimeHist[2]->Scale(1./ToFTimeHist[2]->Integral());
   ToFTimeHist[0]->GetXaxis()->SetTitle("time [ns]");
   ToFTimeHist[1]->GetXaxis()->SetTitle("time [ns]");
   ToFTimeHist[2]->GetXaxis()->SetTitle("time [ns]");


   ToFTimeHist[0]->SetLineColor(kRed);
   ToFTimeHist[1]->SetLineColor(kBlue);
   ToFTimeHist[2]->SetLineColor(kGreen);

   ToFTimeHist[1]->Draw("HIST");
   ToFTimeHist[0]->Draw("HIST SAME");
   ToFTimeHist[2]->Draw("HIST SAME");
   leg->Draw();
   c1.SaveAs("Plots/AdditionalToF.pdf");
   c1.SaveAs("Plots/AdditionalToF.pdf]");

   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
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
   c1.cd(2);
   ToFMassHist2[0]->Scale(1./ToFMassHist2[0]->Integral());
   ToFMassHist2[1]->Scale(1./ToFMassHist2[1]->Integral());
   ToFMassHist2[2]->Scale(1./ToFMassHist2[2]->Integral());
   ToFMassHist2[0]->SetTitle("Resonstructed Mass from ToF");

   ToFMassHist2[0]->GetYaxis()->SetTitle("1/N dN/dm_{tof} [1/GeV]");

   ToFMassHist2[0]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   ToFMassHist2[1]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   ToFMassHist2[2]->GetXaxis()->SetTitle("m_{tof} [GeV]");

   ToFMassHist2[0]->SetLineColor(kRed);
   ToFMassHist2[1]->SetLineColor(kBlue);
   ToFMassHist2[2]->SetLineColor(kGreen);

   ToFMassHist2[0]->Draw("HIST");
   ToFMassHist2[1]->Draw("HIST SAME");
   ToFMassHist2[2]->Draw("HIST SAME");
   leg->Draw();
   c1.cd(3);
   ToFMassHist10[0]->Scale(1./ToFMassHist10[0]->Integral());
   ToFMassHist10[1]->Scale(1./ToFMassHist10[1]->Integral());
   ToFMassHist10[2]->Scale(1./ToFMassHist10[2]->Integral());
   ToFMassHist10[0]->SetTitle("Resonstructed Mass from ToF");

   ToFMassHist10[0]->GetYaxis()->SetTitle("1/N dN/dm_{tof} [1/GeV]");


   ToFMassHist10[0]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   ToFMassHist10[1]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   ToFMassHist10[2]->GetXaxis()->SetTitle("m_{tof} [GeV]");

   ToFMassHist10[0]->SetLineColor(kRed);
   ToFMassHist10[1]->SetLineColor(kBlue);
   ToFMassHist10[2]->SetLineColor(kGreen);

   ToFMassHist10[0]->Draw("HIST");
   ToFMassHist10[1]->Draw("HIST SAME");
   ToFMassHist10[2]->Draw("HIST SAME");
   leg->Draw();
   c1.cd(4);
   ToFMassHist40[0]->Scale(1./ToFMassHist40[0]->Integral());
   ToFMassHist40[1]->Scale(1./ToFMassHist40[1]->Integral());
   ToFMassHist40[2]->Scale(1./ToFMassHist40[2]->Integral());
   ToFMassHist40[0]->SetTitle("Resonstructed Mass from ToF");

   ToFMassHist40[0]->GetYaxis()->SetTitle("1/N dN/dm_{tof} [1/GeV]");
   
   ToFMassHist40[0]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   ToFMassHist40[1]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   ToFMassHist40[2]->GetXaxis()->SetTitle("m_{tof} [GeV]");

   ToFMassHist40[0]->SetLineColor(kRed);
   ToFMassHist40[1]->SetLineColor(kBlue);
   ToFMassHist40[2]->SetLineColor(kGreen);

   ToFMassHist40[0]->Draw("HIST");
   ToFMassHist40[1]->Draw("HIST SAME");
   ToFMassHist40[2]->Draw("HIST SAME");
   leg->Draw();
  
   c1.SaveAs("Plots/ToF/Mass.pdf");
   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
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
   //DoubleToFMassHist[2]->Draw("HIST SAME");
   leg3->Draw();
   c1.cd(2);
   DoubleToFMassHist2[0]->Scale(1./DoubleToFMassHist2[0]->Integral());
   DoubleToFMassHist2[1]->Scale(1./DoubleToFMassHist2[1]->Integral());
   DoubleToFMassHist2[2]->Scale(1./DoubleToFMassHist2[2]->Integral());
   DoubleToFMassHist2[0]->SetTitle("Resonstructed Mass from ToF");

   DoubleToFMassHist2[0]->GetYaxis()->SetTitle("1/N dN/dm_{tof} [1/GeV]");

   DoubleToFMassHist2[0]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   DoubleToFMassHist2[1]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   DoubleToFMassHist2[2]->GetXaxis()->SetTitle("m_{tof} [GeV]");

   DoubleToFMassHist2[0]->SetLineColor(kRed);
   DoubleToFMassHist2[1]->SetLineColor(kBlue);
   DoubleToFMassHist2[2]->SetLineColor(kGreen);

   DoubleToFMassHist2[0]->Draw("HIST");
   DoubleToFMassHist2[1]->Draw("HIST SAME");
   //DoubleToFMassHist2[2]->Draw("HIST SAME");
   leg3->Draw();
   c1.cd(3);
   DoubleToFMassHist10[0]->Scale(1./DoubleToFMassHist10[0]->Integral());
   DoubleToFMassHist10[1]->Scale(1./DoubleToFMassHist10[1]->Integral());
   DoubleToFMassHist10[2]->Scale(1./DoubleToFMassHist10[2]->Integral());
   DoubleToFMassHist10[0]->SetTitle("Resonstructed Mass from ToF");

   DoubleToFMassHist10[0]->GetYaxis()->SetTitle("1/N dN/dm_{tof} [1/GeV]");

   DoubleToFMassHist10[0]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   DoubleToFMassHist10[1]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   DoubleToFMassHist10[2]->GetXaxis()->SetTitle("m_{tof} [GeV]");

   DoubleToFMassHist10[0]->SetLineColor(kRed);
   DoubleToFMassHist10[1]->SetLineColor(kBlue);
   DoubleToFMassHist10[2]->SetLineColor(kGreen);

   DoubleToFMassHist10[0]->Draw("HIST");
   DoubleToFMassHist10[1]->Draw("HIST SAME");
   //DoubleToFMassHist10[2]->Draw("HIST SAME");
   leg3->Draw();
   c1.cd(4);
   DoubleToFMassHist40[0]->Scale(1./DoubleToFMassHist40[0]->Integral());
   DoubleToFMassHist40[1]->Scale(1./DoubleToFMassHist40[1]->Integral());
   DoubleToFMassHist40[2]->Scale(1./DoubleToFMassHist40[2]->Integral());
   DoubleToFMassHist40[0]->SetTitle("Resonstructed Mass from ToF");
   DoubleToFMassHist40[0]->GetYaxis()->SetTitle("1/N dN/dm_{tof} [1/GeV]");

   DoubleToFMassHist40[0]->GetXaxis()->SetTitle("m_{tof} [GeV]");

   DoubleToFMassHist40[1]->GetXaxis()->SetTitle("m_{tof} [GeV]");
   DoubleToFMassHist40[2]->GetXaxis()->SetTitle("m_{tof} [GeV]");

   DoubleToFMassHist40[0]->SetLineColor(kRed);
   DoubleToFMassHist40[1]->SetLineColor(kBlue);
   DoubleToFMassHist40[2]->SetLineColor(kGreen);

   DoubleToFMassHist40[0]->Draw("HIST");
   DoubleToFMassHist40[1]->Draw("HIST SAME");
   //DoubleToFMassHist40[2]->Draw("HIST SAME");
   leg2->Draw();
  
   c1.SaveAs("Plots/ToF/DoubleMass.pdf");
   
 

   MLDataTree->Write();
   file->Close();
}

