#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <TH1.h>
#include <TH3.h>
#include <TBranch.h>
#include <TH2.h>
#include <TTree.h>
#include <TChain.h>
#include <TCut.h>
#include <TProfile.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include <TRandom.h>
#include <TEventList.h>
#include <TMultiLayerPerceptron.h>
#include <TComplex.h>
#include <TVirtualGeoPainter.h>
#include <TFile.h>
#include <TSystem.h>
#include <TClassTree.h>
#include <TPaveLabel.h>
#include <TCanvas.h>
#include <TGClient.h>
#include <RQ_OBJECT.h>
#include <TApplication.h>
#include <TRint.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TF1.h>
#include <TF2.h>
#include <TGenPhaseSpace.h>
#include <TLorentzVector.h>
#include <Riostream.h>
#include <TObjString.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TLatex.h>
#include <Math/Boost.h>
#include "TUnfoldSys.h"
#include "TUnfold.h"
#include "MuonIDEvaluator.h"



void Check()
{
   Float_t pi=TMath::Pi();
   Float_t MuonMass=0.1056583;
   Float_t ElectronMass=0.00051099895000;
   Float_t ProtonMass=0.93827208816;
   int eventID = 0; // Event ID counter
   int eventPassed = 0;
   int count = 0;
   gROOT->SetBatch(kTRUE);
   gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");
   //gStyle->SetOptStat(0);

   TString infile="/run/media/epic/Data/Muons/Grape-10x275/Paper/Current/reco*.root";




   // Set up input file chain
   TChain *mychain = new TChain("events");
   mychain->Add(infile);

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
   //TTreeReaderArray<int> partParI(tree_reader, "_MCParticles_parents.index");

   TTreeReaderArray<int> mcGenStat(tree_reader, "MCParticlesHeadOnFrameNoBeamFX.generatorStatus");
   TTreeReaderArray<double> mcMomX(tree_reader, "MCParticlesHeadOnFrameNoBeamFX.momentum.x");
   TTreeReaderArray<double> mcMomY(tree_reader, "MCParticlesHeadOnFrameNoBeamFX.momentum.y");
   TTreeReaderArray<double> mcMomZ(tree_reader, "MCParticlesHeadOnFrameNoBeamFX.momentum.z");
   TTreeReaderArray<int> mcPdg(tree_reader, "MCParticlesHeadOnFrameNoBeamFX.PDG");
   TTreeReaderArray<double> mcMass(tree_reader, "MCParticlesHeadOnFrameNoBeamFX.mass");
   TTreeReaderArray<float> mcCharge(tree_reader, "MCParticlesHeadOnFrameNoBeamFX.charge");
   TTreeReaderArray<unsigned int> mcParb(tree_reader, "MCParticlesHeadOnFrameNoBeamFX.parents_begin");
   TTreeReaderArray<unsigned int> mcPare(tree_reader, "MCParticlesHeadOnFrameNoBeamFX.parents_end");
   //TTreeReaderArray<int> mcParI(tree_reader, "_MCParticlesHeadOnFrameNoBeamFX_parents.index");

   // Get Reconstructed Track Information
   TTreeReaderArray<float> trackMomX(tree_reader, "ReconstructedParticles.momentum.x");
   TTreeReaderArray<float> trackMomY(tree_reader, "ReconstructedParticles.momentum.y");
   TTreeReaderArray<float> trackMomZ(tree_reader, "ReconstructedParticles.momentum.z");
   TTreeReaderArray<int> trackPDG(tree_reader, "ReconstructedParticles.PDG");
   TTreeReaderArray<float> trackMass(tree_reader, "ReconstructedParticles.mass");
   TTreeReaderArray<float> trackCharge(tree_reader, "ReconstructedParticles.charge");
   TTreeReaderArray<float> trackEng(tree_reader, "ReconstructedParticles.energy");



    
   //Muon Efficiency
      vector<double> MUEta={50,-3,3};
      vector<double> MUPhi={50,-pi,pi};
      vector<double> MUEnergy={50,0,5};
      vector<double> MUPt={50,0,4};

      TH1D *MuRecoPtHist = new TH1D("MuRecoPtHist","MuRecoPtHist",MUPt[0],MUPt[1],MUPt[2]);
      TH1D *MuRecoEtaHist = new TH1D("MuRecoEtaHist","MuRecoEtaHist",MUEta[0],MUEta[1],MUEta[2]);
      TH1D *MuRecoPhiHist = new TH1D("MuRecoPhiHist","MuRecoPhiHist",MUPhi[0],MUPhi[1],MUPhi[2]);
      TH1D *MuRecoEnergyHist = new TH1D("MuRecoEnergyHist","MuRecoEnergyHist",MUEnergy[0],MUEnergy[1],MUEnergy[2]);


    MuonIDEvaluator muID("plik.root");
    

    int muoncount=0;
   
    while(tree_reader.Next()) // Loop over events
    {
      eventID++;
      //if(eventID>20000) break;
      double Delta_Eta=8,Delta_Phi=3.95;

      vector<double> V_Eta(2),V_Phi(2);
      double t_mc=0;
      bool trmu=0,tramu=0;
      TLorentzVector muon,amuon,dimuon,electron,proton,mcelectron,mcdimuon,mcmuon,mcamuon,mcproton,mcphoton,mcebeam,mcpbeam;
      vector<TLorentzVector> mcphotons;
    //Monte Carlo
      for(int particle=0;particle<mcGenStat.GetSize();particle++)
      {    
        if(mcGenStat[particle]==1)
        {
            if(abs(mcPdg[particle])==13)
            {
                          
               TVector3 direction;
               direction.SetXYZ(mcMomX[particle],mcMomY[particle],mcMomZ[particle]);
               double energy=sqrt(pow(mcMomZ[particle],2)+pow(mcMomY[particle],2)+pow(mcMomX[particle],2)+pow(MuonMass,2));
               
               double Mu_Eta=direction.Eta();
               double Mu_Phi=direction.Phi();
               
               if(mcPdg[particle]==13) 
               {              
                  V_Eta[0]=Mu_Eta;
                  V_Phi[0]=Mu_Phi;
                  mcmuon.SetPxPyPzE(mcMomX[particle],mcMomY[particle],mcMomZ[particle],energy);                
               }
               else if(mcPdg[particle]==-13) 
               {
               
               V_Eta[1]=Mu_Eta;
               V_Phi[1]=Mu_Phi;
               mcamuon.SetPxPyPzE(mcMomX[particle],mcMomY[particle],mcMomZ[particle],energy);
               }
            }  
            
      }

   //Reconstructed
      ///////////////////////////////////////////////
      
      //int numberb=0,numbera=0;
      for(int particle=0;particle<trackEng.GetSize();particle++)
      {
         
         if( trackPDG[particle]==0)
         {
            double Mu_Eta,Mu_Phi;
            if(trackCharge[particle]==-1)
            {
               Mu_Eta=V_Eta[0];
               Mu_Phi=V_Phi[0];
            }
            else if(trackCharge[particle]==1)
            {  
               Mu_Eta=V_Eta[1];
               Mu_Phi=V_Phi[1];
            }
            else continue;
            
            
            TLorentzVector Partic;
            Partic.SetPxPyPzE(trackMomX[particle],trackMomY[particle],trackMomZ[particle],trackEng[particle]);            
          
            double REC_Eta=Partic.Eta();
            double REC_Phi=Partic.Phi();
            
            Delta_Eta=REC_Eta-Mu_Eta;
            Delta_Phi=acos(cos(REC_Phi-Mu_Phi));
            float prob = muID.evaluate(eventID-1, particle);
               
            double RPhiEta=sqrt(pow(Delta_Phi,2)+pow(Delta_Eta,2));
             if (prob < 0) continue;  // nie przeszedł cięć

                if (prob > 0.5) muoncount++;

            if(RPhiEta<0.1 && trackCharge[particle]==-1)
            {                     
               muon=Partic;            
               trmu=1;
      
            }
            if(RPhiEta<0.1 && trackCharge[particle]==1)
            {
               amuon=Partic;               
               tramu=1;
            }  

         }   
      
    
      }
      if(trmu && tramu)
      {
         dimuon=muon+amuon;
         
      }
  

      
         
      if(mcmuon.P()!=0 && muon.P()!=0)
      {
         MuRecoPtHist->Fill(muon.Pt());
         MuRecoEtaHist->Fill(muon.Eta());
         MuRecoPhiHist->Fill(muon.Phi());
         MuRecoEnergyHist->Fill(muon.E());


      if(mcamuon.P()!=0 && amuon.P()!=0)
      {
         MuRecoPtHist->Fill(amuon.Pt());
         MuRecoEtaHist->Fill(amuon.Eta());
         MuRecoPhiHist->Fill(amuon.Phi());
         MuRecoEnergyHist->Fill(amuon.E());

      }
      
    
        }
    }
    }
    cout<<"==========================="<<endl;
    cout<<muoncount<<" muons reconstructed"<<endl;
    cout<<"==========================="<<endl;
}