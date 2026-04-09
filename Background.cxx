
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



void Background()
{

    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");
    gStyle->SetOptStat(0);

    double DEG=180/TMath::Pi();

    float ECalEnergy;
    float HCalEnergy;
    float ECalNumber;
    float HCalNumber;
    float EoverP;
    float IsMuon;

    TFile *file = new TFile("MLDataEpic.root", "RECREATE");
    //TFile *file = new TFile("MLDataBackground.root", "RECREATE");

    TTree *MLDataTree = new TTree("MLDataTree", "MLDataTree");

    MLDataTree->Branch("ECalEnergy", &ECalEnergy, "ECalEnergy/F");
    MLDataTree->Branch("HCalEnergy", &HCalEnergy, "HCalEnergy/F");
    MLDataTree->Branch("ECalNumber", &ECalNumber, "ECalNumber/F");
    MLDataTree->Branch("HCalNumber", &HCalNumber, "HCalNumber/F");
    MLDataTree->Branch("EoverP", &EoverP, "EoverP/F");
    MLDataTree->Branch("IsMuon", &IsMuon, "IsMuon/F");


    //==================================//
    TH1D *NumberB0Barrel= new TH1D("NumberB0Barrel","NumberB0Barrel",5,-0.5,4.5);
    TH1D *NumberEcalBarrel= new TH1D("NumberEcalBarrel","NumberEcalBarrel",5,-0.5,4.5);
    TH1D *NumberEcalEndcapP= new TH1D("NumberEcalEndcapP","NumberEcalEndcapP",5,-0.5,4.5);
    TH1D *NumberEcalEndcapN= new TH1D("NumberEcalEndcapN","NumberEcalEndcapN",5,-0.5,4.5);
    TH1D *NumberHcalBarrel= new TH1D("NumberHcalBarrel","NumberHcalBarrel",5,-0.5,4.5);
    TH1D *NumberHcalEndcapP= new TH1D("NumberHcalEndcapP","NumberHcalEndcapP",10,-0.5,9.5);
    TH1D *NumberHcalEndcapN= new TH1D("NumberHcalEndcapN","NumberHcalEndcapN",10,-0.5,9.5);
    TH1D *NumberLFHcal= new TH1D("NumberLFHcal","NumberLFHcal",10,-0.5,9.5);
    //==================================//
    TH1D *NumberEcal= new TH1D("NumberEcal","NumberEcal",10,-0.5,9.5);
    TH1D *NumberHcal= new TH1D("NumberHcal","NumberHcal",10,-0.5,9.5);
    TH1D *EnergyEcal= new TH1D("EnergyEcal","EnergyEcal",100,0,7);
    TH1D *EnergyHcal= new TH1D("EnergyHcal","EnergyHcal",100,0,7);
    //==================================//
    TH1D *PDG= new TH1D("PDG","PDG",41,-230.5,230);
    TH1D *NumberParticles= new TH1D("NumberParticles","NumberParticles",9,-0.5,8.5);
    //==================================//
    TH1D *AllParticTheta = new TH1D("AllParticTheta","AllParticTheta",50,0,180);
    TH1D *AllParticPhi= new TH1D("AllParticPhi","AllParticPhi",30,-180,180);
    TH1D *AllParticEnergy= new TH1D("AllParticEnergy","AllParticEnergy",50,0,10);
    //==================================//
    TH1D *NotFoundParticTheta = new TH1D("NotFoundParticTheta","NotFoundParticTheta",50,0,180);
    TH1D *NotFoundParticPhi= new TH1D("NotFoundParticPhi","NotFoundParticPhi",30,-180,180);
    TH1D *NotFoundParticEnergy= new TH1D("NotFoundParticEnergy","NotFoundParticEnergy",50,0,10);
    //==================================//
    TH1D *RadiusEtaPhi= new TH1D("RadiusEtaPhi","RadiusEtaPhi",100,0,10);
    //==================================//
    TH1D *EnergyHist= new TH1D("EnergyHist","EnergyHist",50,0,15);
    TH1D *EnergyMomHist= new TH1D("EnergyMomHist","EnergyMomHist",50,0,2);
    TH2D *EnergyvsMomHist= new TH2D("EEnergyvsMomHist","EnergyvsMomHist",50,0,10,50,0,2);
   
    //TString infiles="/Data/Background/bgmerged_*";
    TString infiles="/Data/Muons/Epic-10x275/recoEL0S.root";

    // Set up input file chain
    TChain *mychain = new TChain("events");
    mychain->Add(infiles);
  

    
    // Initialize reader
    TTreeReader tree_reader(mychain);

    // Get Particle Information
    TTreeReaderArray<int> partGenStat(tree_reader, "MCParticles.generatorStatus");
    /*
    TTreeReaderArray<float> partMomX(tree_reader, "MCParticles.momentum.x");
    TTreeReaderArray<float> partMomY(tree_reader, "MCParticles.momentum.y");
    TTreeReaderArray<float> partMomZ(tree_reader, "MCParticles.momentum.z");
    */
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
    TTreeReaderArray<float> trackMomX(tree_reader, "ReconstructedParticles.momentum.x");
    TTreeReaderArray<float> trackMomY(tree_reader, "ReconstructedParticles.momentum.y");
    TTreeReaderArray<float> trackMomZ(tree_reader, "ReconstructedParticles.momentum.z");
    TTreeReaderArray<int> trackPDG(tree_reader, "ReconstructedParticles.PDG");
    TTreeReaderArray<float> trackMass(tree_reader, "ReconstructedParticles.mass");
    TTreeReaderArray<float> trackCharge(tree_reader, "ReconstructedParticles.charge");
    TTreeReaderArray<float> trackEng(tree_reader, "ReconstructedParticles.energy");
    
    // Get Associations Between MCParticles and ReconstructedParticles
    TTreeReaderArray<unsigned int> recoAssoc(tree_reader, "ReconstructedParticleAssociations.recID");
    TTreeReaderArray<unsigned int> simuAssoc(tree_reader, "ReconstructedParticleAssociations.simID");

    // Get B0 Information
    TTreeReaderArray<unsigned int> recoAssocB0(tree_reader, "B0ECalClusterAssociations.recID");
    TTreeReaderArray<unsigned int> simuAssocB0(tree_reader, "B0ECalClusterAssociations.simID");
    TTreeReaderArray<float> B0Eng(tree_reader, "B0ECalClusters.energy");
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
    
    int eventID=0;
    double FoundParticles=0;
    double particscount=0;
    double muonscount=0;

    while(tree_reader.Next()){
        eventID++;
        if(eventID%1000==0) cout<<"Event number: "<<eventID<<endl;
        NumberB0Barrel->Fill(B0Eng.GetSize());
        NumberEcalBarrel->Fill(EcalBarrelEng.GetSize());
        NumberEcalEndcapP->Fill(EcalEndcapPEng.GetSize());
        NumberEcalEndcapN->Fill(EcalEndcapNEng.GetSize());
        NumberHcalBarrel->Fill(HcalBarrelEng.GetSize());
        NumberHcalEndcapN->Fill(HcalEndcapNEng.GetSize());
        NumberHcalEndcapP->Fill(HcalEndcapPEng.GetSize());
        NumberLFHcal->Fill(LFHcalEng.GetSize());
        
        IsMuon=0;
        EoverP=0;
        double Delta_Eta=8,Delta_Phi=3.95;
        vector<double> V_Eta(2),V_Phi(2);
        ECalNumber=0;
        HCalNumber=0;
        for(int mcpartic=0;mcpartic<partMass.GetSize();mcpartic++)
        {
            if(abs(partPdg[mcpartic])==13 && partGenStat[mcpartic]==1)
            {
                TVector3 direction;
                direction.SetXYZ(partMomX[mcpartic], partMomY[mcpartic], partMomZ[mcpartic]);
                
                double MC_Eta=direction.Eta();
                double MC_Phi=direction.Phi();
            
                if(partPdg[mcpartic]==13) 
                {
                    V_Eta[0]=MC_Eta;
                    V_Phi[0]=MC_Phi;
                }
                else if(partPdg[mcpartic]==-13) 
                {
                    V_Eta[1]=MC_Eta;
                    V_Phi[1]=MC_Phi;

                }
            }
        }
        
        for(int particle=0; particle<simuAssoc.GetSize();particle++)
        {
            //Obligatory Cuts 
            int Found=0;
            TLorentzVector Partic;
            
            Partic.SetPxPyPzE(trackMomX[particle],trackMomY[particle],trackMomZ[particle],trackEng[particle]);
            if(!(trackPDG[particle]==0 || abs(trackPDG[particle])==13)) continue;
            if(trackPDG[particle]==0)
            {
                double MC_Eta,MC_Phi;
                if(trackCharge[particle]==-1)
                {
                MC_Eta=V_Eta[0];
                MC_Phi=V_Phi[0];
                }
                else if(trackCharge[particle]==1)
                {  
                MC_Eta=V_Eta[1];
                MC_Phi=V_Phi[1];
                }
                else continue;
                PDG->Fill(trackPDG[particle]);
                //Tagging muons
                double REC_Eta=Partic.Eta();
                double REC_Phi=Partic.Phi();
                
                Delta_Eta=REC_Eta-MC_Eta;
                Delta_Phi=acos(cos(REC_Phi-MC_Phi));
                
                double RPhiEta=sqrt(pow(Delta_Phi,2)+pow(Delta_Eta,2));
                RadiusEtaPhi->Fill(RPhiEta);
                if(RPhiEta<0.1) 
                {
                    IsMuon=1;
                    muonscount++;
                }  

            }
            
            //Ecal Energy Search
            ECalEnergy=0;
            NumberParticles->Fill(simuAssocEcalBarrel.GetSize());
            
            if(B0Eng.GetSize()==simuAssocB0.GetSize())
            {
                for(int cluster=0;cluster<B0Eng.GetSize();cluster++)
                {
                    
                    if(simuAssoc[particle]==simuAssocB0[cluster])
                    {
                        
                        ECalEnergy+=B0Eng[cluster];
                        ECalNumber++;
                    } 
                } 
            }
            
            if(EcalBarrelImagingEng.GetSize()==simuAssocEcalBarrelImaging.GetSize())
            {
                for(int cluster=0;cluster<EcalBarrelImagingEng.GetSize();cluster++)
                {
                    
                    if(simuAssoc[particle]==simuAssocEcalBarrelImaging[cluster])
                    {
                        
                        ECalEnergy+=EcalBarrelImagingEng[cluster];
                        ECalNumber++;
                    } 
                } 
            }
            
            if(EcalBarrelScFiEng.GetSize()==simuAssocEcalBarrelScFi.GetSize())
            {
                for(int cluster=0;cluster<EcalBarrelScFiEng.GetSize();cluster++)
                {
                    
                    if(simuAssoc[particle]==simuAssocEcalBarrelScFi[cluster])
                    {
                        
                        ECalEnergy+=EcalBarrelScFiEng[cluster];
                        ECalNumber++;
                    } 
                } 
            }
            if(EcalBarrelEng.GetSize()==simuAssocEcalBarrel.GetSize())
            {
                for(int cluster=0;cluster<EcalBarrelEng.GetSize();cluster++)
                {
                    
                    if(simuAssoc[particle]==simuAssocEcalBarrel[cluster])
                    {
                        
                        ECalEnergy+=EcalBarrelEng[cluster];
                        ECalNumber++;
                    } 
                } 
            }
            if(EcalEndcapPEng.GetSize()==simuAssocEcalEndcapP.GetSize())
            {
                for(int cluster=0;cluster<EcalEndcapPEng.GetSize();cluster++)
                    if(simuAssoc[particle]==simuAssocEcalEndcapP[cluster])  
                    {
                        ECalEnergy+=EcalEndcapPEng[cluster];
                        ECalNumber++;
                    }  
            }      
            if(EcalEndcapNEng.GetSize()==simuAssocEcalEndcapN.GetSize())
            {
                for(int cluster=0;cluster<EcalEndcapNEng.GetSize();cluster++)
                    if(simuAssoc[particle]==simuAssocEcalEndcapN[cluster])  
                    {
                        ECalEnergy+=EcalEndcapNEng[cluster];
                        ECalNumber++;
                    }  
            }
            if(ECalEnergy!=0)
            {
                EnergyEcal->Fill(ECalEnergy);
                Found=1;
            }
            
            //Hcal Energy Search
            HCalEnergy=0;
            if(HcalBarrelEng.GetSize()==simuAssocHcalBarrel.GetSize())
            {
                for(int cluster=0;cluster<HcalBarrelEng.GetSize();cluster++)
                    if(simuAssoc[particle]==simuAssocHcalBarrel[cluster])
                    {
                        HCalEnergy+=HcalBarrelEng[cluster];
                        HCalNumber++;
                    }  
            }
            if(HcalEndcapPEng.GetSize()==simuAssocHcalEndcapP.GetSize())
            {
                for(int cluster=0;cluster<HcalEndcapPEng.GetSize();cluster++)           
                    if(simuAssoc[particle]==simuAssocHcalEndcapP[cluster])  
                    {
                        HCalEnergy+=HcalEndcapPEng[cluster];
                        HCalNumber++;
                    }               
            }
            if(HcalEndcapNEng.GetSize()==simuAssocHcalEndcapN.GetSize())
            {
                for(int cluster=0;cluster<HcalEndcapNEng.GetSize();cluster++)
                    if(simuAssoc[particle]==simuAssocHcalEndcapN[cluster])     
                    {
                        HCalEnergy+=HcalEndcapNEng[cluster]; 
                        HCalNumber++;
                    }           
            }
            
            if(LFHcalEng.GetSize()==simuAssocLFHcal.GetSize())
            {
                for(int cluster=0;cluster<LFHcalEng.GetSize();cluster++)           
                    if(simuAssoc[particle]==simuAssocLFHcal[cluster])        
                    {
                        HCalEnergy+=LFHcalEng[cluster];
                        HCalNumber++;
                    }       
            }
            
            if(HCalEnergy!=0)
            {
                EnergyHcal->Fill(HCalEnergy);
                Found=1;
            }
            NumberEcal->Fill(ECalNumber);
            NumberHcal->Fill(HCalNumber);
            particscount++;
            FoundParticles+=Found;   

            //Track properties 
            double FullEnergy=HCalEnergy+ECalEnergy;
            if(FullEnergy==0) continue;
            double Momentum=sqrt(pow(trackMomX[particle],2)+pow(trackMomY[particle],2)+pow(trackMomZ[particle],2));
            EoverP=FullEnergy/Momentum;
            EnergyHist->Fill(FullEnergy);
            EnergyMomHist->Fill(EoverP);
            EnergyvsMomHist->Fill(Momentum,EoverP);

            if(Found==1)  MLDataTree->Fill();  
            
            AllParticTheta->Fill(Partic.Theta()*DEG);
            AllParticPhi->Fill(Partic.Phi()*DEG);
            AllParticEnergy->Fill(Partic.Energy());
            if(Found==0)
            {
                NotFoundParticTheta->Fill(Partic.Theta()*DEG);
                NotFoundParticPhi->Fill(Partic.Phi()*DEG);
                NotFoundParticEnergy->Fill(Partic.Energy());
            } 
        
            
        }
        
    
        
    }
    cout<<"==========================="<<endl;
    cout<<"End of file"<<endl;
    cout<<"Number of events: "<<eventID<<endl;
    cout<<"Found particles: "<<FoundParticles<<"   All particles: "<<particscount<<endl;
    cout<<"Muons: "<<muonscount<<endl;

    cout<<"==========================="<<endl;
   
   
    gStyle->SetOptStat(111111);
    TCanvas c1;

    

    c1.SaveAs("Plots/BackgroundMuon.pdf[");
    c1.Clear();
        
        EnergyEcal->Draw("HIST");
    c1.SaveAs("Plots/BackgroundMuon.pdf");

    c1.Clear();

        EnergyHcal->Draw("HIST");
    c1.SaveAs("Plots/BackgroundMuon.pdf");

    c1.Clear();

        NumberEcal->Draw("HIST");
    c1.SaveAs("Plots/BackgroundMuon.pdf");

    c1.Clear();

        NumberHcal->Draw("HIST");
    c1.SaveAs("Plots/BackgroundMuon.pdf");

    gPad->SetLogy(1);
    c1.Clear();

        RadiusEtaPhi->Draw("HIST");
    c1.SaveAs("Plots/BackgroundMuon.pdf");
    gPad->SetLogy(0);

    c1.Clear();

      gPad->SetLogz(1);
      EnergyvsMomHist->Draw("HIST");

   c1.SaveAs("Plots/BackgroundMuon.pdf");

   gPad->SetLogz(0);

    c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      EnergyHist->Draw("HIST");
   c1.cd(2);
      EnergyMomHist->Draw("HIST");
   c1.SaveAs("Plots/BackgroundMuon.pdf");

    c1.Clear();
    c1.Divide(2,2);
    c1.cd(1);
        AllParticTheta->SetLineColor(kBlue);
        NotFoundParticTheta->SetLineColor(kRed);
        AllParticTheta->Draw();
        NotFoundParticTheta->Draw("same");
    c1.cd(2);
        AllParticPhi->SetLineColor(kBlue);
        NotFoundParticPhi->SetLineColor(kRed);
        AllParticPhi->SetMinimum(0);
        AllParticPhi->Draw();
        NotFoundParticPhi->Draw("same");
    c1.cd(3); 
        AllParticEnergy->SetLineColor(kBlue);
        NotFoundParticEnergy->SetLineColor(kRed);
        AllParticEnergy->Draw();
        NotFoundParticEnergy->Draw("same");   
    c1.SaveAs("Plots/BackgroundMuon.pdf");

    c1.Clear();
    c1.Divide(2,2);
    c1.cd(1);

        NumberEcalBarrel->Draw("HIST");
    c1.cd(2);

        NumberB0Barrel->Draw("HIST");
    c1.cd(3);

        NumberEcalEndcapP->Draw("HIST");
    c1.cd(4);

        NumberEcalEndcapN->Draw("HIST");    
    c1.SaveAs("Plots/BackgroundMuon.pdf");

    c1.Clear();
    c1.Divide(2,2);
    c1.cd(1);

        NumberHcalBarrel->Draw("HIST");
    c1.cd(2);

        NumberLFHcal->Draw("HIST");
    c1.cd(3);

        NumberHcalEndcapP->Draw("HIST");
    c1.cd(4);

        NumberHcalEndcapN->Draw("HIST");
    c1.SaveAs("Plots/BackgroundMuon.pdf");

    c1.Clear();

        PDG->Draw("HIST");
    c1.SaveAs("Plots/BackgroundMuon.pdf");

    c1.Clear();

        NumberParticles->Draw("HIST");
    c1.SaveAs("Plots/BackgroundMuon.pdf");
    c1.SaveAs("Plots/BackgroundMuon.pdf]");

    MLDataTree->Write();
    file->Close();
}






        
        


    