
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
#include "Calorimeternew.cxx"
#include "GreatCluster.cxx"


void IDAnalysisNew()
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
   float HCalEoverP;
   float ECalEoverP;
   float Momentum;
   float IsMuon;
   float FileIndex;
   vector<float> EcalShape;
   vector<float> HcalShape;

   TFile *file = new TFile("MLData.root", "RECREATE");
   TTree *MLDataTree = new TTree("MLDataTree", "MLDataTree");
   /*
   MLDataTree->Branch("ECalEnergy", &ECalEnergy, "ECalEnergy/F");
   MLDataTree->Branch("HCalEnergy", &HCalEnergy, "HCalEnergy/F");
   MLDataTree->Branch("ECalNumber", &ECalNumber, "ECalNumber/F");
   MLDataTree->Branch("HCalNumber", &HCalNumber, "HCalNumber/F");
   MLDataTree->Branch("HCalEoverP", &HCalEoverP, "HCalEoverP/F");
   MLDataTree->Branch("ECalEoverP", &ECalEoverP, "ECalEoverP/F");
   MLDataTree->Branch("Momentum", &Momentum, "Momentum/F");
   */
   MLDataTree->Branch("EcalShape", &EcalShape);
   MLDataTree->Branch("HcalShape", &HcalShape);
   MLDataTree->Branch("IsMuon", &IsMuon, "IsMuon/F");
   MLDataTree->Branch("FileIndex", &FileIndex, "FileIndex/F");

   //////////////////////
   //Setting up histograms
   //////////////////////
   static constexpr int NumOfFiles=4;
   TH1D *EnergyEcal[NumOfFiles],*EnergyHcal[NumOfFiles],*NumberEcal[NumOfFiles],*NumberHcal[NumOfFiles];
   TH1D *NumberEcalBarrel[NumOfFiles],*NumberEcalEndcapP[NumOfFiles],*NumberEcalEndcapN[NumOfFiles],*NumberHcalBarrel[NumOfFiles],
      *NumberHcalEndcapP[NumOfFiles],*NumberHcalEndcapN[NumOfFiles],*NumberLFHcal[NumOfFiles],*NumberB0Barrel[NumOfFiles];
   TH1D *PDG[NumOfFiles],*NumberParticles[NumOfFiles];
   TH1D *AllParticEta[NumOfFiles], *AllParticPhi[NumOfFiles], *AllParticEnergy[NumOfFiles];
   TH1D *NotFoundParticEta[NumOfFiles], *NotFoundParticPhi[NumOfFiles], *NotFoundParticEnergy[NumOfFiles];
   TH1D *ECalEnergyHist[NumOfFiles], *ECalEnergyMomHist[NumOfFiles],*HCalEnergyHist[NumOfFiles], *HCalEnergyMomHist[NumOfFiles];
   TH2D *ECalEnergyvsMomHist[NumOfFiles],*HCalEnergyvsMomHist[NumOfFiles];
   TH2D *ECalEnergyMomvsEtaHist[NumOfFiles],  *HCalEnergyMomvsEtaHist[NumOfFiles];
   TH1D *ToFTimeHist[NumOfFiles], *ToFMassHist[NumOfFiles], *ToFMassHist2[NumOfFiles], *ToFMassHist10[NumOfFiles], *ToFMassHist40[NumOfFiles];
   TH1D *DoubleToFMassHist[NumOfFiles], *DoubleToFMassHist2[NumOfFiles], *DoubleToFMassHist10[NumOfFiles], *DoubleToFMassHist40[NumOfFiles];
   TH1D *EcalShapeHist[7][NumOfFiles], *HcalShapeHist[7][NumOfFiles];
   TH1D *simuasocHist[NumOfFiles];

   
   vector<TString> files(NumOfFiles);

   files.at(0)="/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
   files.at(1)="/run/media/epic/Data/Muons/Grape-10x275/Paper/RECO/*.root";
   files.at(2)="/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
   files.at(3)="/run/media/epic/Data/Tau/reco/Energy_10x275/double_pi/recoDoublePi.root";
   //files.at(1)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Pions.root";
   //files.at(2)="/run/media/epic/Data/Background/JPsi/March/*.root";



   //files.at(2)="/run/media/epic/Data/Background/Pions/*.root";

   
   /*
   files.at(0)="/Data/Muons/ToF/Electrons.root";
   files.at(1)="/Data/Muons/Epic-10x275/recoEL0S.root";
   files.at(2)="/Data/Tau/EpIC/tcs/tau_tcs_hist.root";
   */



   TF1 *upperbondE = new TF1("upperbondE", "2/(x**2)+0.05", 0.001, 24.0);

   //TF1 *upperbondE = new TF1("upperbondE", "2/x+0.05", 0.01, 24.0);
   upperbondE->SetLineColor(kRed);
   upperbondE->SetLineWidth(1);

   TF1 *upperbondH = new TF1("upperbondH", "3.5/x+0.1",  0.01, 24.0); 
   upperbondH->SetLineColor(kRed);
   upperbondH->SetLineWidth(1);
      
   TF1 *lowerbondH = new TF1("lowerbondH", "0.3/x-0.25/(x*x)",  0.01, 24.0); 
   lowerbondH->SetLineColor(kRed);
   lowerbondH->SetLineWidth(1);
   
   for(int File=0; File<NumOfFiles;File++)
   {
      string name;
      if(File==3 || File==2) name="Pions";
      else name="Muons";
      // Set up input file chain
      TChain *mychain = new TChain("events");
      
      mychain->Add(files.at(File));

      // Initialize reader
      TTreeReader tree_reader(mychain);
      Long64_t nEvents = mychain->GetEntries();
      cout<<nEvents<<endl;

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
      AllParticEta[File] = new TH1D(Form("AllParticEta%s",name.c_str()),Form("AllParticEta%s",name.c_str()),50,-4,4);
      AllParticPhi[File]= new TH1D(Form("AllParticPhi%s",name.c_str()),Form("AllParticPhi%s",name.c_str()),30,-180,180);
      AllParticEnergy[File]= new TH1D(Form("AllParticEnergy%s",name.c_str()),Form("AllParticEnergy%s",name.c_str()),50,0,20);
      
      NotFoundParticEta[File] = new TH1D(Form("NotFoundParticEta%s",name.c_str()),Form("NotFoundParticEta%s",name.c_str()),50,-4,4);
      NotFoundParticPhi[File]= new TH1D(Form("NotFoundParticPhi%s",name.c_str()),Form("NotFoundParticPhi%s",name.c_str()),30,-180,180);
      NotFoundParticEnergy[File]= new TH1D(Form("NotFoundParticEnergy%s",name.c_str()),Form("NotFoundParticEnergy%s",name.c_str()),50,0,20);
      //==================================//
      ECalEnergyHist[File]= new TH1D(Form("ECalEnergyHist%s",name.c_str()),Form("ECalEnergyHist%s",name.c_str()),50,0,15);
      ECalEnergyMomHist[File]= new TH1D(Form("ECalEnergyMomHist%s",name.c_str()),Form("ECalEnergyMomHist%s",name.c_str()),50,0,0.2);
      ECalEnergyvsMomHist[File]= new TH2D(Form("ECalEnergyvsMomHist%s",name.c_str()),Form("ECalEnergyvsMomHist%s",name.c_str()),50,0,22,50,0,2);
      ECalEnergyMomvsEtaHist[File]= new TH2D(Form("ECalEnergyMomvsEtaHist%s",name.c_str()),Form("ECalEnergyMomvsEtaist%s",name.c_str()),50,-3.5,3.5,50,0,2);


      HCalEnergyHist[File]= new TH1D(Form("HCalEnergyHist%s",name.c_str()),Form("HCalEnergyHist%s",name.c_str()),50,0,15);
      HCalEnergyMomHist[File]= new TH1D(Form("HCalEnergyMomHist%s",name.c_str()),Form("HCalEnergyMomHist%s",name.c_str()),50,0,4);
      HCalEnergyvsMomHist[File]= new TH2D(Form("HCalEnergyvsMomHist%s",name.c_str()),Form("HCalEnergyvsMomHist%s",name.c_str()),50,0,22,50,0,2);
      HCalEnergyMomvsEtaHist[File]= new TH2D(Form("HCalEnergyMomvsEtaHist%s",name.c_str()),Form("HCalEnergyMomvsEtaist%s",name.c_str()),50,-3.5,3.5,50,0,2);
      simuasocHist[File]= new TH1D(Form("simuasocHist%s",name.c_str()),Form("simuasocHist%s",name.c_str()),21,-10000.5,2000.5);

      vector<vector<float>> ParamsVector= {{100,0,800},{100,0,200},{100,0,1e-3},{100,0,1e-3},{100,0,2200},{100,0,2200},{100,0,2200}};
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
         //if(eventID>10) break;   //if(File==0) continue; //if(eventID!=13) continue;

         //if(eventID>40000) break; 
         if(eventID>0.9*nEvents) break; 

         
         if(eventID%100000==0) cout<<"File "<<name<<" and event number... "<<eventID<<endl;


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
         

         int id=0;
         for(int particle=0; particle<trackEng.GetSize();particle++)
         {
            //cout<<"Particle: "<<particle<<endl;
            ECalEnergy=0;
            HCalEnergy=0;
            ECalNumber=0;
            HCalNumber=0;
            ECalEoverP=0;
            HCalEoverP=0; 
            Momentum=0;
            EcalShape.clear();
            HcalShape.clear();
            particscount++;
            //Obligatory Cuts 
            double mass;
            if(File==0) mass=MuonMass;
            else if(File==1) mass=ElectronMass;
            else if(File==2) mass=PionMass;

            int Found=0;
            TLorentzVector Partic;
            Partic.SetPxPyPzE(trackMomX[particle],trackMomY[particle],trackMomZ[particle],trackEng[particle]);
            if(Partic.Theta()>170) continue;    
            if(Partic.Eta()<-1.25) continue;
            if(Partic.E()<1) continue;

         
            PDG[File]->Fill(trackPDG[particle]);
            AllParticEnergy[File]->Fill(Partic.E());
            AllParticEta[File]->Fill(Partic.Eta());
            AllParticPhi[File]->Fill(Partic.Phi()*DEG);
           
           //Ecal Energy Search
            NumberParticles[File]->Fill(simuAssocEcalBarrel.GetSize());
            int simuID = simuAssoc[particle];
            simuasocHist[File]->Fill(simuID);

            //////////////////////
            // Collect energies and shapes from all ECal detectors
            //////////////////////
            vector<vector<float>> EcalAllShapes;
            //cout<<"Tutaj EcalBarrel"<<endl;
            
            auto [EnergyEcalBarrel,NumberEcalBarrel,ShapeEcalBarrel] = Calorimeternew( simuID, EcalBarrelEng, simuAssocEcalBarrel, EcalBarrelx, EcalBarrely,
                EcalBarrelz, EcalBarrelShPB, EcalBarrelShPE,EcalBarrelShParameters);

            ECalEnergy+=EnergyEcalBarrel;
            
            if(!ShapeEcalBarrel.empty() && !ShapeEcalBarrel[0].empty() && ShapeEcalBarrel[0][0] != 0){
               ECalNumber+=NumberEcalBarrel;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeEcalBarrel.begin(), ShapeEcalBarrel.end());
            }  
         
            
            auto [EnergyEndcapP,NumberEndcapP,ShapeEndcapP] = Calorimeternew( simuID, EcalEndcapPEng, simuAssocEcalEndcapP, EcalEndcapPx, EcalEndcapPy,
                EcalEndcapPz, EcalEndcapPShPB, EcalEndcapPShPE,EcalEndcapPShParameters);
            ECalEnergy+=EnergyEndcapP;
            
            if(!ShapeEndcapP.empty() && !ShapeEndcapP[0].empty() && ShapeEndcapP[0][0] != 0){
               ECalNumber+=NumberEndcapP;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeEndcapP.begin(), ShapeEndcapP.end());
            }

            auto [EnergyEndcapN,NumberEndcapN,ShapeEndcapN] = Calorimeternew( simuID, EcalEndcapNEng, simuAssocEcalEndcapN, EcalEndcapNx, EcalEndcapNy,
                EcalEndcapNz, EcalEndcapNShPB, EcalEndcapNShPE,EcalEndcapNShParameters);

            ECalEnergy+=EnergyEndcapN;
            
            if(!ShapeEndcapN.empty() && !ShapeEndcapN[0].empty() && ShapeEndcapN[0][0] != 0){
               ECalNumber+=NumberEndcapN;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeEndcapN.begin(), ShapeEndcapN.end());
            }
            
            auto [EnergyB0,NumberB0,ShapeB0] = Calorimeternew( simuID, B0Eng, simuAssocB0, B0x, B0y, B0z, B0ShPB, B0ShPE,B0ShParameters);
               
            ECalEnergy+=EnergyB0;
            
            if(!ShapeB0.empty() && !ShapeB0[0].empty() && ShapeB0[0][0] != 0){
               ECalNumber+=NumberB0;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeB0.begin(), ShapeB0.end());
            }

            auto [EnergyImaging,NumberImaging,ShapeImaging] = Calorimeternew( simuID, EcalBarrelImagingEng, simuAssocEcalBarrelImaging, EcalBarrelImagingx, EcalBarrelImagingy,
                EcalBarrelImagingz, EcalBarrelImagingShPB, EcalBarrelImagingShPE,EcalBarrelImagingShParameters);

            ECalEnergy+=EnergyImaging;
            
            if(!ShapeImaging.empty() && !ShapeImaging[0].empty() && ShapeImaging[0][0] != 0){
               ECalNumber+=NumberImaging;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeImaging.begin(), ShapeImaging.end());
            }
            
            auto [EnergyScFi,NumberScFi,ShapeScFi] = Calorimeternew( simuID, EcalBarrelScFiEng, simuAssocEcalBarrelScFi, EcalBarrelScFix, EcalBarrelScFiy,
                EcalBarrelScFiz, EcalBarrelScFiShPB, EcalBarrelScFiShPE,EcalBarrelScFiShParameters);

            ECalEnergy+=EnergyScFi;
            
            if(!ShapeScFi.empty() && !ShapeScFi[0].empty() && ShapeScFi[0][0] != 0){
               ECalNumber+=NumberScFi;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeScFi.begin(), ShapeScFi.end());
            }
            //cout<<"ECAL"<<endl;
            
            // Assign shape from detector with highest energy


            if(ECalEnergy!=0 && ECalNumber!=0)
            {
               EnergyEcal[File]->Fill(ECalEnergy);
               EcalShape = GreatCluster(EcalAllShapes);
               Found=1;
            }
            //////////////////////           
            //Hcal Energy Search
            //////////////////////
            //cout<<"Tutaj ShapeHcalBarrel"<<endl;
            vector<vector<float>> HcalAllShapes;
            
            auto [EnergyHcalBarrel,NumberHcalBarrel,ShapeHcalBarrel] = Calorimeternew( simuID, HcalBarrelEng, simuAssocHcalBarrel, HcalBarrelx, HcalBarrely,
                HcalBarrelz, HcalBarrelShPB, HcalBarrelShPE,HcalBarrelShParameters);

            HCalEnergy+=EnergyHcalBarrel;
            
            if(!ShapeHcalBarrel.empty() && !ShapeHcalBarrel[0].empty() && ShapeHcalBarrel[0][0] != 0){
               HCalNumber+=NumberHcalBarrel;
               HcalAllShapes.insert(HcalAllShapes.end(), ShapeHcalBarrel.begin(), ShapeHcalBarrel.end());
            }
            
            auto [EnergyHcalEndcapP,NumberHcalEndcapP,ShapeHcalEndcapP] = Calorimeternew( simuID, HcalEndcapPEng, simuAssocHcalEndcapP, HcalEndcapPx, HcalEndcapPy,
                HcalEndcapPz, HcalEndcapPShPB, HcalEndcapPShPE,HcalEndcapPShParameters);

            HCalEnergy+=EnergyHcalEndcapP;
            
            if(!ShapeHcalEndcapP.empty() && !ShapeHcalEndcapP[0].empty() && ShapeHcalEndcapP[0][0] != 0){
               HCalNumber+=NumberHcalEndcapP;
               HcalAllShapes.insert(HcalAllShapes.end(), ShapeHcalEndcapP.begin(), ShapeHcalEndcapP.end());
            }
            
            auto [EnergyLFHcal,NumberLFHcal,ShapeLFHcal] = Calorimeternew( simuID, LFHcalEng, simuAssocLFHcal, LFHcalx, LFHcaly, LFHcalz, LFHcalShPB, LFHcalShPE,LFHcalShParameters);

            HCalEnergy+=EnergyLFHcal;
            
            if(!ShapeLFHcal.empty() && !ShapeLFHcal[0].empty() && ShapeLFHcal[0][0] != 0){
               HCalNumber+=NumberLFHcal;
               HcalAllShapes.insert(HcalAllShapes.end(), ShapeLFHcal.begin(), ShapeLFHcal.end());
            }
            
            auto [EnergyHcalEndcapN,NumberHcalEndcapN,ShapeHcalEndcapN] = Calorimeternew( simuID, HcalEndcapNEng, simuAssocHcalEndcapN, HcalEndcapNx, HcalEndcapNy,
                HcalEndcapNz, HcalEndcapNShPB, HcalEndcapNShPE,HcalEndcapNShParameters);

            HCalEnergy+=EnergyHcalEndcapN;
            
            if(!ShapeHcalEndcapN.empty() && !ShapeHcalEndcapN[0].empty() && ShapeHcalEndcapN[0][0] != 0){
               HCalNumber+=NumberHcalEndcapN;
               HcalAllShapes.insert(HcalAllShapes.end(), ShapeHcalEndcapN.begin(), ShapeHcalEndcapN.end());
            }
            
            // Assign shape from detector with highest energy
            //cout<<"HCAL"<<endl;
            //if(HCalNumber>=1) continue;
            
            if(HCalEnergy!=0 && HCalNumber!=0)
            {
               HcalShape = GreatCluster(HcalAllShapes);
               EnergyHcal[File]->Fill(HCalEnergy);
               Found=1;
            }
            
            

            

            if(abs(Partic.Eta())<1.3 && abs(Partic.Eta())>1) continue;
            FoundParticles+=Found;
            if(Found==0) continue;
            //if(!(trackPDG[particle]==0 || abs(trackPDG[particle])==13)) continue;
            
            //Track properties 
            double FullEnergy=HCalEnergy+ECalEnergy;
            if(FullEnergy==0) continue;

            Momentum=Partic.P();
            HCalEoverP=HCalEnergy/Momentum;
            ECalEoverP=ECalEnergy/Momentum;

            NumberEcal[File]->Fill(ECalNumber);
            NumberHcal[File]->Fill(HCalNumber);
            ECalEnergyMomvsEtaHist[File]->Fill(Partic.Eta(),ECalEoverP);
            HCalEnergyMomvsEtaHist[File]->Fill(Partic.Eta(),HCalEoverP);

            ECalEnergyHist[File]->Fill(ECalEnergy);
            ECalEnergyMomHist[File]->Fill(ECalEoverP);
            ECalEnergyvsMomHist[File]->Fill(Momentum,ECalEoverP);

            HCalEnergyHist[File]->Fill(HCalEnergy);
            HCalEnergyMomHist[File]->Fill(HCalEoverP);
            HCalEnergyvsMomHist[File]->Fill(Momentum,HCalEoverP);
            
            for(int i=0; i<7; i++){
               if(!EcalShape.empty()) EcalShapeHist[i][File]->Fill(EcalShape[i]);
               if(!HcalShape.empty()) HcalShapeHist[i][File]->Fill(HcalShape[i]);
            }
            
            if(File==3 || File==2) IsMuon=0;  
            else IsMuon=1; 
            FileIndex=File;
            
            if(HCalEoverP>upperbondH->Eval(Momentum)) continue;
            if(HCalEoverP<lowerbondH->Eval(Momentum)) continue;
            if(ECalEoverP>upperbondE->Eval(Momentum)) continue;
            
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
    leg->AddEntry(EnergyEcal[1],"Pions","l");




   TLegend* leg1 = new TLegend(0.78, 0.8, 0.95, 0.95);
    leg1->SetBorderSize(0);
    leg1->SetNColumns(1);
    leg1->SetColumnSeparation(0.1);
    leg1->SetEntrySeparation(0.1);
    leg1->SetMargin(0.15);
    leg1->SetTextFont(42);
    leg1->SetTextSize(0.05);
    leg1->AddEntry(EnergyEcal[0],"Muons ","l");
    leg1->AddEntry(EnergyEcal[3],"Pions","l");

    TLegend* leg2 = new TLegend(0.58, 0.2, 0.85, 0.45);
    leg2->SetBorderSize(0);
    leg2->SetNColumns(1);
    leg2->SetColumnSeparation(0.1);
    leg2->SetEntrySeparation(0.1);
    leg2->SetMargin(0.15);
    leg2->SetTextFont(42);
    leg2->SetTextSize(0.05);
    leg2->AddEntry(EnergyEcal[0],"Muons","l");
    leg2->AddEntry(EnergyEcal[3],"Pions","l");

    TLegend* leg3 = new TLegend(0.58, 0.6, 0.85, 0.85);
    leg3->SetBorderSize(0);
    leg3->SetNColumns(1);
    leg3->SetColumnSeparation(0.1);
    leg3->SetEntrySeparation(0.1);
    leg3->SetMargin(0.15);
    leg3->SetTextFont(42);
    leg3->SetTextSize(0.05);
    leg3->AddEntry(EnergyEcal[0],"Muons","l");
    leg3->AddEntry(EnergyEcal[3],"Pions","l");

   

   c1.SaveAs("Plots/CalID.pdf[");
 

   

   c1.Clear();
      EnergyEcal[0]->Scale(1./EnergyEcal[0]->Integral());
      EnergyEcal[2]->Scale(1./EnergyEcal[2]->Integral());

      EnergyEcal[0]->SetLineColor(kRed);
      EnergyEcal[2]->SetLineColor(kBlue);

      EnergyEcal[2]->Draw("HIST");
      EnergyEcal[0]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalID.pdf");


   

   c1.Clear();
      EnergyHcal[0]->Scale(1./EnergyHcal[0]->Integral());
      EnergyHcal[2]->Scale(1./EnergyHcal[2]->Integral());
      EnergyHcal[0]->SetLineColor(kRed);
      EnergyHcal[2]->SetLineColor(kBlue);
      EnergyHcal[0]->Draw("HIST");
      EnergyHcal[2]->Draw("HIST SAME");
      leg->Draw();

   c1.SaveAs("Plots/CalID.pdf");

   
   c1.Clear();
      NumberEcal[0]->Scale(1./NumberEcal[0]->Integral());
      NumberEcal[2]->Scale(1./NumberEcal[2]->Integral());
      NumberEcal[0]->SetLineColor(kRed);
      NumberEcal[2]->SetLineColor(kBlue);
      NumberEcal[2]->Draw("HIST");
      NumberEcal[0]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalID.pdf");

   c1.Clear();
      NumberHcal[0]->Scale(1./NumberHcal[0]->Integral());
      NumberHcal[2]->Scale(1./NumberHcal[2]->Integral());
      NumberHcal[0]->SetLineColor(kRed);
      NumberHcal[2]->SetLineColor(kBlue);
      NumberHcal[2]->Draw("HIST");
      NumberHcal[0]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalID.pdf");

   c1.Clear();

      simuasocHist[0]->SetLineColor(kRed);
      simuasocHist[2]->SetLineColor(kBlue);

      simuasocHist[2]->Draw("HIST");
      simuasocHist[0]->Draw("HIST SAME");
   c1.SaveAs("Plots/CalID.pdf");

   c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      ECalEnergyMomvsEtaHist[0]->Draw("colz");
   c1.cd(2);
      ECalEnergyMomvsEtaHist[2]->Draw("colz");
   c1.SaveAs("Plots/CalID.pdf");

   c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      HCalEnergyMomvsEtaHist[0]->Draw("colz");
   c1.cd(2);
      HCalEnergyMomvsEtaHist[2]->Draw("colz");
   c1.SaveAs("Plots/CalID.pdf");

   c1.Clear();
   c1.Divide(2,1);
   
   c1.cd(1);
      gPad->SetLogz(1);
      ECalEnergyvsMomHist[1]->Draw("HIST");
      upperbondE->Draw("same");
   c1.cd(2);
      gPad->SetLogz(1);
      ECalEnergyvsMomHist[3]->Draw("HIST");
      upperbondE->Draw("same");
   c1.SaveAs("Plots/CalID.pdf");
   gPad->SetLogz(0);

   c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      ECalEnergyHist[0]->Scale(1./ECalEnergyHist[0]->Integral());
      ECalEnergyHist[2]->Scale(1./ECalEnergyHist[2]->Integral());

      ECalEnergyHist[0]->SetLineColor(kRed);
      ECalEnergyHist[2]->SetLineColor(kBlue);

      ECalEnergyHist[2]->Draw("HIST");
      ECalEnergyHist[0]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(2);
      ECalEnergyMomHist[0]->Scale(1./ECalEnergyMomHist[0]->Integral());
      ECalEnergyMomHist[2]->Scale(1./ECalEnergyMomHist[2 ]->Integral());

      ECalEnergyMomHist[0]->SetLineColor(kRed);
      ECalEnergyMomHist[2]->SetLineColor(kBlue);

      ECalEnergyMomHist[2  ]->Draw("HIST");
      ECalEnergyMomHist[0]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalID.pdf");
   c1.Clear();
   c1.Divide(2,1);
   
   c1.cd(1);
      gPad->SetLogz(1);
      HCalEnergyvsMomHist[1]->Draw("HIST");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");

   c1.cd(2);
      gPad->SetLogz(1);
      HCalEnergyvsMomHist[3]->Draw("HIST");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
   c1.SaveAs("Plots/CalID.pdf");
   gPad->SetLogz(0);

   c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      HCalEnergyHist[0]->Scale(1./HCalEnergyHist[0]->Integral());
      HCalEnergyHist[2]->Scale(1./HCalEnergyHist[2]->Integral());

      HCalEnergyHist[0]->SetLineColor(kRed);
      HCalEnergyHist[2]->SetLineColor(kBlue);

      HCalEnergyHist[2]->Draw("HIST");
      HCalEnergyHist[0]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(2);
      HCalEnergyMomHist[0]->Scale(1./HCalEnergyMomHist[0]->Integral());
      HCalEnergyMomHist[2]->Scale(1./HCalEnergyMomHist[2 ]->Integral());

      HCalEnergyMomHist[0]->SetLineColor(kRed);
      HCalEnergyMomHist[2]->SetLineColor(kBlue);

      HCalEnergyMomHist[2  ]->Draw("HIST");
      HCalEnergyMomHist[0]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalID.pdf");


   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      AllParticEta[0]->SetLineColor(kBlue);
      NotFoundParticEta[0]->SetLineColor(kRed);
      
      AllParticEta[0]->Draw();
      NotFoundParticEta[0]->Draw("same");
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
   c1.SaveAs("Plots/CalID.pdf");
   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      AllParticEta[2]->SetLineColor(kBlue);
      NotFoundParticEta[2]->SetLineColor(kRed);
      AllParticEta[2]->Draw();
      NotFoundParticEta[2]->Draw("colz");
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
   c1.SaveAs("Plots/CalID.pdf");


   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      NumberEcalBarrel[0]->Scale(1./NumberEcalBarrel[0]->Integral());
      NumberEcalBarrel[2]->Scale(1./NumberEcalBarrel[2]->Integral());
      NumberEcalBarrel[0]->SetLineColor(kRed);
      NumberEcalBarrel[2]->SetLineColor(kBlue);
      NumberEcalBarrel[0]->Draw("HIST");
      NumberEcalBarrel[2   ]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(2);
      NumberB0Barrel[0]->Scale(1./NumberB0Barrel[0]->Integral());
      NumberB0Barrel[2]->Scale(1./NumberB0Barrel[2]->Integral());
      NumberB0Barrel[0]->SetLineColor(kRed);
      NumberB0Barrel[2]->SetLineColor(kBlue);
      NumberB0Barrel[0]->Draw("HIST");
      NumberB0Barrel[2]->Draw("HIST SAME");
   c1.cd(3);
      NumberEcalEndcapP[0]->Scale(1./NumberEcalEndcapP[0]->Integral());
      NumberEcalEndcapP[2]->Scale(1./NumberEcalEndcapP[2]->Integral());
      NumberEcalEndcapP[0]->SetLineColor(kRed);
      NumberEcalEndcapP[2]->SetLineColor(kBlue);
      NumberEcalEndcapP[0]->Draw("HIST");
      NumberEcalEndcapP[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(4);
      NumberEcalEndcapN[0]->Scale(1./NumberEcalEndcapN[0]->Integral());
      NumberEcalEndcapN[2]->Scale(1./NumberEcalEndcapN[2]->Integral());
      NumberEcalEndcapN[0]->SetLineColor(kRed);
      NumberEcalEndcapN[2]->SetLineColor(kBlue);
      NumberEcalEndcapN[0]->Draw("HIST");
      NumberEcalEndcapN[2]->Draw("HIST SAME");
      leg->Draw();      
   c1.SaveAs("Plots/CalID.pdf");

   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      NumberHcalBarrel[0]->Scale(1./NumberHcalBarrel[0]->Integral());
      NumberHcalBarrel[2]->Scale(1./NumberHcalBarrel[2]->Integral());
      NumberHcalBarrel[0]->SetLineColor(kRed);
      NumberHcalBarrel[2   ]->SetLineColor(kBlue);
      NumberHcalBarrel[0]->Draw("HIST");
      NumberHcalBarrel[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(2);
      NumberLFHcal[0]->Scale(1./NumberLFHcal[0]->Integral());
      NumberLFHcal[2]->Scale(1./NumberLFHcal[2]->Integral());
      NumberLFHcal[0]->SetLineColor(kRed);
      NumberLFHcal[2]->SetLineColor(kBlue);
      NumberLFHcal[0]->Draw("HIST");
      NumberLFHcal[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(3);
      NumberHcalEndcapP[0]->Scale(1./NumberHcalEndcapP[0]->Integral());
      NumberHcalEndcapP[2]->Scale(1./NumberHcalEndcapP[2]->Integral());
      NumberHcalEndcapP[0]->SetLineColor(kRed);
      NumberHcalEndcapP[2]->SetLineColor(kBlue);
      NumberHcalEndcapP[0]->Draw("HIST");
      NumberHcalEndcapP[2]->Draw("HIST SAME");
      leg->Draw();
   c1.cd(4);
      NumberHcalEndcapN[0]->Scale(1./NumberHcalEndcapN[0]->Integral());
      NumberHcalEndcapN[2]->Scale(1./NumberHcalEndcapN[2]->Integral());
      NumberHcalEndcapN[0]->SetLineColor(kRed);
      NumberHcalEndcapN[2]->SetLineColor(kBlue);
      NumberHcalEndcapN[0]->Draw("HIST");
      NumberHcalEndcapN[2]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalID.pdf");


   c1.Clear();
      PDG[0]->Scale(1./PDG[0]->Integral());
      PDG[2]->Scale(1./PDG[2]->Integral());
      PDG[0]->SetLineColor(kRed);
      PDG[2]->SetLineColor(kBlue);
      PDG[2 ]->Draw("HIST");
      PDG[0]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalID.pdf");

   c1.Clear();
      NumberParticles[0]->Scale(1./NumberParticles[0]->Integral());
      NumberParticles[2]->Scale(1./NumberParticles[2]->Integral());
      NumberParticles[0]->SetLineColor(kRed);
      NumberParticles[2]->SetLineColor(kBlue);
      NumberParticles[2]->Draw("HIST");
      NumberParticles[0]->Draw("HIST SAME");
      leg->Draw();
   c1.SaveAs("Plots/CalID.pdf");

     // Shape histograms
   c1.Clear();
   c1.Divide(2,4);
   for(int i=0; i<7; i++){
      c1.cd(i+1);
      EcalShapeHist[i][0]->SetLineColor(kRed);
      EcalShapeHist[i][2]->SetLineColor(kBlue);
      EcalShapeHist[i][0]->Draw("HIST");
      EcalShapeHist[i][2]->Draw("HIST SAME");
   }
   c1.SaveAs("Plots/CalID.pdf");
   
   c1.Clear();
   c1.Divide(2,4);
   for(int i=0; i<7; i++){
      c1.cd(i+1);
      HcalShapeHist[i][0]->SetLineColor(kRed);
      HcalShapeHist[i][2]->SetLineColor(kBlue);
      HcalShapeHist[i][0]->Draw("HIST");
      HcalShapeHist[i][2]->Draw("HIST SAME");
   }
   c1.SaveAs("Plots/CalID.pdf");

   c1.SaveAs("Plots/CalID.pdf]");
   
   
   MLDataTree->Write();
   file->Close();
}

