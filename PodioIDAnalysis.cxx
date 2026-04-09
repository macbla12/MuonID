#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <TFile.h>
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
#include "podio/ROOTReader.h"
#include "podio/Frame.h"

#include "edm4hep/MCParticleCollection.h"
#include "edm4eic/ReconstructedParticleCollection.h"
#include "edm4eic/ClusterCollection.h"

#include <iomanip>


double deltaR(double eta1, double phi1, double eta2, double phi2) {
    double deta = eta1 - eta2;
    double dphi = acos(cos(phi1 - phi2)); // ROOT-owa korekta okresowości phi
    return std::sqrt(deta*deta + dphi*dphi);
}

void PodioIDAnalysis()
{

   gROOT->SetBatch(kTRUE);
   gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");
   //gStyle->SetOptStat(0);

   double DEG=180/TMath::Pi();


   static constexpr int NumOfFiles=5;
   TH1D *ParticleMomentumHist[NumOfFiles], *ParticleEtaHist[NumOfFiles];
   TH1D *ECalEnergyHist[NumOfFiles], *ECalEnergyMomHist[NumOfFiles], *HCalEnergyHist[NumOfFiles], *HCalEnergyMomHist[NumOfFiles];
   TH2D *ECalEnergyvsMomHist[NumOfFiles],*HCalEnergyvsMomHist[NumOfFiles];
   TH2D *AfterCutECalEnergyvsMomHist[NumOfFiles],*AfterCutHCalEnergyvsMomHist[NumOfFiles];
   TH1D *REtaPhiHist[NumOfFiles];

   
   for(int i=0;i<NumOfFiles;i++)
   {
      string name;
      if(i==0) name="Muons";
      if(i==1) name="Electrons";
      if(i==2) name="Pions";
      if(i==3) name="Kaons";
      if(i==4) name="Protons";


      //==================================//
      ParticleMomentumHist[i]= new TH1D(Form("ParticleMomentumHist%s",name.c_str()),Form("ParticleMomentumHist%s",name.c_str()),30,-2,22);
      ParticleEtaHist[i]= new TH1D(Form("ParticleEtaHist%s",name.c_str()),Form("ParticleEtaHist%s",name.c_str()),30,-4,4);
      REtaPhiHist[i]= new TH1D(Form("REtaPhiHist%s",name.c_str()),Form("REtaPhiHist%s",name.c_str()),50,-0.1,4);
      
      //==================================//
      ECalEnergyHist[i]= new TH1D(Form("ECalEnergyHist%s",name.c_str()),Form("ECalEnergyHist%s",name.c_str()),50,0,15);
      ECalEnergyMomHist[i]= new TH1D(Form("ECalEnergyMomHist%s",name.c_str()),Form("ECalEnergyMomHist%s",name.c_str()),50,0,2);
      ECalEnergyvsMomHist[i]= new TH2D(Form("ECalEnergyvsMomHist%s",name.c_str()),Form("ECalEnergyvsMomHist%s",name.c_str()),100,0,22,100,0,2);
      AfterCutECalEnergyvsMomHist[i]= new TH2D(Form("AfterCutECalEnergyvsMomHist%s",name.c_str()),Form("AfterCutECalEnergyvsMomHist%s",name.c_str()),100,0,22,100,0,2);
      
      HCalEnergyHist[i]= new TH1D(Form("HCalEnergyHist%s",name.c_str()),Form("HCalEnergyHist%s",name.c_str()),50,0,15);
      HCalEnergyMomHist[i]= new TH1D(Form("HCalEnergyMomHist%s",name.c_str()),Form("HCalEnergyMomHist%s",name.c_str()),50,0,2);      
      HCalEnergyvsMomHist[i]= new TH2D(Form("HCalEnergyvsMomHist%s",name.c_str()),Form("HCalEnergyvsMomHist%s",name.c_str()),100,0,22,100,0,2);
      AfterCutHCalEnergyvsMomHist[i]= new TH2D(Form("AfterCutHCalEnergyvsMomHist%s",name.c_str()),Form("AfterCutHCalEnergyvsMomHist%s",name.c_str()),100,0,22,100,0,2);

   }
   TF1 *upperbondE = new TF1("upperbondE", "0.7/x", 0.5, 20.0);
   upperbondE->SetLineColor(kRed);
   upperbondE->SetLineWidth(1);

   TF1 *upperbondH = new TF1("upperbondH", "2.3/x", 0.5, 20.0); 
   upperbondH->SetLineColor(kRed);
   upperbondH->SetLineWidth(1);
      
   TF1 *lowerbondH = new TF1("lowerbondH", "0.35/x-0.25/(x*x)", 0.5, 20.0); 
   lowerbondH->SetLineColor(kRed);
   lowerbondH->SetLineWidth(1);

  
   

   vector<std::string> infiles = {
      "/run/media/epic/Data/Background/SingleParticles/muMinus/EICreconOut_00_500_0-500.root",
      "/run/media/epic/Data/Background/SingleParticles/electron/EICreconOut_00_500_0-500.root",
      "/run/media/epic/Data/Background/SingleParticles/piMinus/eicrecon_out.root",
      "/run/media/epic/Data/Background/SingleParticles/kMinus/EICreconOut_00_500_0-500.root",
      "/run/media/epic/Data/Background/SingleParticles/proton/EICreconOut_00_500_0-500.root",
   };

    vector<std::string> infilestrue = {
      "/run/media/epic/Data/Muons/Grape-10x100/Paper/recoGL.root",
      "/run/media/epic/Data/Muons/ToF/Electrons.root",
      "/run/media/epic/Data/Tau/reco/Energy_10x275/double_pi/recoDoublePi.root",
      "/run/media/epic/Data/Background/SingleParticles/kMinus/EICreconOut_00_500_0-500.root",
      "/run/media/epic/Data/Background/SingleParticles/proton/EICreconOut_00_500_0-500.root",
   };
  
   vector<std::string> infilesfull = {
      "/run/media/epic/Data/Background/SingleParticles/SingleFiles/Muons.root",
      "/run/media/epic/Data/Background/SingleParticles/SingleFiles/Electrons.root",
      "/run/media/epic/Data/Background/SingleParticles/SingleFiles/Pions.root",
      "/run/media/epic/Data/Background/SingleParticles/SingleFiles/Kaons.root",
      "/run/media/epic/Data/Background/SingleParticles/SingleFiles/Protons.root",
   };
   vector<double> AllParticles(NumOfFiles);
   vector<double> FirstCutParticles(NumOfFiles);
   vector<double> SecondCutParticles(NumOfFiles);
   
   for(int File=0; File<NumOfFiles;File++)  
   {
      string name;
      if(File==0) name="Muons";
      if(File==1) name="Electrons";
      if(File==2) name="Pions";
      if(File==3) name="Kaons";
      if(File==4) name="Protons";
      
      auto reader = podio::ROOTReader();
      //reader.openFile(infilesfull[File]);
      reader.openFile(infilesfull[File]);
      //reader.openFile(infiles[File]);


      
      size_t nevents = reader.getEntries("events");

      int eventID=0;
      double FoundParticles=0;
      double particscount=0;
      double BadPDG=0;
      
      for (size_t eventID = 0; eventID < nevents/5; eventID++){  // Loop over events
      //for (size_t eventID = 0; eventID < 5; eventID++){ 

         if(eventID%5000==0) cout<<"File "<<name<<" and event number... "<<eventID<<endl;

         const auto event = podio::Frame(reader.readNextEntry("events"));

         // HCAL
         const auto &hcalBarrel = event.get<edm4eic::ClusterCollection>("HcalBarrelClusters");
         const auto &hcalEndcapP = event.get<edm4eic::ClusterCollection>("HcalEndcapPInsertClusters");
         const auto &hcalEndcapN = event.get<edm4eic::ClusterCollection>("HcalEndcapNClusters");
         const auto &lfHcal = event.get<edm4eic::ClusterCollection>("LFHCALClusters");
        
         // ReconstructedParticles
         const auto &rcparts = event.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");
         
         for (const auto &rcp : rcparts) 
         {
            auto p = rcp.getMomentum();
            TLorentzVector particle;
            particle.SetPxPyPzE(p.x,p.y,p.z,rcp.getEnergy());

            if(particle.Theta()*DEG>178) continue;
            double pEta = particle.Theta();
            double pPhi = particle.Phi();
            double Momentum=particle.P();
            ParticleMomentumHist[File]->Fill(Momentum);
            ParticleEtaHist[File]->Fill(pEta);
            AllParticles[File]++;

            
            double ecalEnergy = 0.0;
            double hcalEnergy = 0.0;

            for (const auto& cluster : rcp.getClusters()) ecalEnergy += cluster.getEnergy();
            
            auto checkHcalColl = [&](const auto& coll) {
               for (const auto& cluster : coll) {
                     // Uwaga: cluster.getPosition() zwraca x,y,z. Zamieńmy to na eta/phi:
                     TVector3 pos(cluster.getPosition().x, cluster.getPosition().y, cluster.getPosition().z);
                     double REtaPhi=deltaR(pEta, pPhi, pos.Eta(), pos.Phi());
                     REtaPhiHist[File]->Fill(REtaPhi);
                     if (REtaPhi < 1)  hcalEnergy += cluster.getEnergy();
               }
            };

            checkHcalColl(hcalBarrel);
            checkHcalColl(hcalEndcapP);
            checkHcalColl(hcalEndcapN);
            checkHcalColl(lfHcal);
            if(ecalEnergy + hcalEnergy==0) continue;
            double ECalEoverP=ecalEnergy/Momentum;
            double HCalEoverP=hcalEnergy/Momentum;

            //cout<<"EoverP "<<EoverP<<" Momentum "<< Momentum<<endl;
            ECalEnergyHist[File]->Fill(ecalEnergy);
            ECalEnergyMomHist[File]->Fill(ECalEoverP);
            ECalEnergyvsMomHist[File]->Fill(Momentum,ECalEoverP);

            HCalEnergyHist[File]->Fill(hcalEnergy);
            HCalEnergyMomHist[File]->Fill(HCalEoverP);
            HCalEnergyvsMomHist[File]->Fill(Momentum,HCalEoverP);
            
            if(abs(rcp.getPDG())!=13 && rcp.getPDG()!=0) continue;
            if(abs(particle.Eta())<1.3 && abs(particle.Eta())>1) continue;
            if(Momentum<1) continue;


            FirstCutParticles[File]++;

            if(ECalEoverP>upperbondE->Eval(Momentum)) continue;
            if(HCalEoverP>upperbondH->Eval(Momentum)) continue;
            if(HCalEoverP<lowerbondH->Eval(Momentum)) continue;
            SecondCutParticles[File]++;
            AfterCutECalEnergyvsMomHist[File]->Fill(Momentum,ECalEoverP);
            AfterCutHCalEnergyvsMomHist[File]->Fill(Momentum,HCalEoverP);


         }

      } 
   }
   cout<<"==================================="<<endl;
   cout<<"All Muons: "<<AllParticles[0]<<" After first cut: "<<FirstCutParticles[0]/AllParticles[0]<<" After second cut: "<<SecondCutParticles[0]/AllParticles[0]<<endl;
   cout<<"All Electrons: "<<AllParticles[1]<<" After first cut: "<<FirstCutParticles[1]/AllParticles[1]<<" After second cut: "<<SecondCutParticles[1]/AllParticles[1]<<endl;
   cout<<"All Pions: "<<AllParticles[2]<<" After first cut: "<<FirstCutParticles[2]/AllParticles[2]<<" After second cut: "<<SecondCutParticles[2]/AllParticles[2]<<endl;
   cout<<"All Kaons: "<<AllParticles[3]<<" After first cut: "<<FirstCutParticles[3]/AllParticles[3]<<" After second cut: "<<SecondCutParticles[3]/AllParticles[3]<<endl;
   cout<<"All Protons: "<<AllParticles[4]<<" After first cut: "<<FirstCutParticles[4]/AllParticles[4]<<" After second cut: "<<SecondCutParticles[4]/AllParticles[4]<<endl;
   cout<<"==================================="<<endl;
   cout<<"Muon Efficiency: "<<SecondCutParticles[0]*100/AllParticles[0]<<"%"<<endl;
   double sumParticle;
   for (auto& n : SecondCutParticles)
     sumParticle += n;
   cout<<"Muon Purity: "<<SecondCutParticles[0]*100/sumParticle<<"%"<<endl;



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
    leg->AddEntry(ECalEnergyHist[0],"Muons","l");
    leg->AddEntry(ECalEnergyHist[1],"Electrons","l");
    leg->AddEntry(ECalEnergyHist[2],"Pions","l");
    leg->AddEntry(ECalEnergyHist[3],"Kaons","l");
    leg->AddEntry(ECalEnergyHist[4],"Protons","l");

   c1.SaveAs("Plots/Finaltest.pdf[");
   c1.Clear();
   c1.Divide(3,2);
   c1.cd(1);
      ParticleMomentumHist[0]->SetLineColor(kRed);
      ParticleMomentumHist[0]->Draw("HIST");
   c1.cd(2);
      ParticleMomentumHist[1]->SetLineColor(kBlue);
      ParticleMomentumHist[1]->Draw("HIST");
   c1.cd(3);
      ParticleMomentumHist[2]->SetLineColor(kGreen);
      ParticleMomentumHist[2]->Draw("HIST");
   c1.cd(4);
      ParticleMomentumHist[3]->SetLineColor(kCyan);
      ParticleMomentumHist[3]->Draw("HIST");
   c1.cd(5);
      ParticleMomentumHist[4]->SetLineColor(kMagenta);
      ParticleMomentumHist[4]->Draw("HIST");
   c1.SaveAs("Plots/Finaltest.pdf");

   c1.Clear();
   c1.Divide(3,2);
   c1.cd(1);
      ParticleEtaHist[0]->SetLineColor(kRed);
      ParticleEtaHist[0]->Draw("HIST");
   c1.cd(2);
      ParticleEtaHist[1]->SetLineColor(kBlue);
      ParticleEtaHist[1]->Draw("HIST");
   c1.cd(3);
      ParticleEtaHist[2]->SetLineColor(kGreen);
      ParticleEtaHist[2]->Draw("HIST");
   c1.cd(4);
      ParticleEtaHist[3]->SetLineColor(kCyan);
      ParticleEtaHist[3]->Draw("HIST");
   c1.cd(5);
      ParticleEtaHist[4]->SetLineColor(kMagenta);
      ParticleEtaHist[4]->Draw("HIST");
   c1.SaveAs("Plots/Finaltest.pdf");

   c1.Clear();
   c1.Divide(3,2);
   c1.cd(1);
      TProfile *prof = ECalEnergyvsMomHist[0]->ProfileX("prof_temp");
      TF1 *bestFit = new TF1("bestFit", "[0]/x", 0.5, 20.0);
      bestFit->SetParameter(0, 0.3);
      prof->Fit(bestFit, "RQ");
      bestFit->SetLineColor(kBlack);
      bestFit->SetLineWidth(1);
      ECalEnergyvsMomHist[0]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");

   c1.cd(2);
      ECalEnergyvsMomHist[1]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");
   c1.cd(3);
      ECalEnergyvsMomHist[2]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");
   c1.cd(4);
      ECalEnergyvsMomHist[3]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");
   c1.cd(5);
      ECalEnergyvsMomHist[4]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");
   c1.SaveAs("Plots/Finaltest.pdf");
   

   c1.Clear();
   c1.Divide(3,2);
   c1.cd(1);
      prof = HCalEnergyvsMomHist[0]->ProfileX("prof_temp");
      bestFit = new TF1("bestFit", "[0]/x", 0.5, 20.0);
      bestFit->SetParameter(0, 0.3);
      prof->Fit(bestFit, "RQ");
      bestFit->SetLineColor(kBlack);
      bestFit->SetLineWidth(1);
      
      HCalEnergyvsMomHist[0]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.cd(2);
      HCalEnergyvsMomHist[1]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.cd(3);
      HCalEnergyvsMomHist[2]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.cd(4);
      HCalEnergyvsMomHist[3]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.cd(5);
      HCalEnergyvsMomHist[4]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.SaveAs("Plots/Finaltest.pdf");

   c1.Clear();
   c1.Divide(3,2);
   c1.cd(1);
      prof = AfterCutECalEnergyvsMomHist[0]->ProfileX("prof_temp");
      bestFit = new TF1("bestFit", "[0]/x", 0.5, 20.0);
      bestFit->SetParameter(0, 0.3);
      prof->Fit(bestFit, "RQ");
      bestFit->SetLineColor(kBlack);
      bestFit->SetLineWidth(1);
      AfterCutECalEnergyvsMomHist[0]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");

   c1.cd(2);
      AfterCutECalEnergyvsMomHist[1]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");
   c1.cd(3);
      AfterCutECalEnergyvsMomHist[2]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");
   c1.cd(4);
      AfterCutECalEnergyvsMomHist[3]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");
   c1.cd(5);
      AfterCutECalEnergyvsMomHist[4]->Draw("colz");
      upperbondE->Draw("same");
      bestFit->Draw("same");
   c1.SaveAs("Plots/Finaltest.pdf");
   

   c1.Clear();
   c1.Divide(3,2);
   c1.cd(1);
      prof = AfterCutHCalEnergyvsMomHist[0]->ProfileX("prof_temp");
      bestFit = new TF1("bestFit", "[0]/x", 0.5, 20.0);
      bestFit->SetParameter(0, 0.3);
      prof->Fit(bestFit, "RQ");
      bestFit->SetLineColor(kBlack);
      bestFit->SetLineWidth(1);  
      AfterCutHCalEnergyvsMomHist[0]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.cd(2);
      AfterCutHCalEnergyvsMomHist[1]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.cd(3);
      AfterCutHCalEnergyvsMomHist[2]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.cd(4);
      AfterCutHCalEnergyvsMomHist[3]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.cd(5);
      AfterCutHCalEnergyvsMomHist[4]->Draw("colz");
      upperbondH->Draw("same");
      lowerbondH->Draw("same");
      bestFit->Draw("same");
   c1.SaveAs("Plots/Finaltest.pdf");
   

   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      ECalEnergyHist[0]->Scale(1./ECalEnergyHist[0]->Integral());
      ECalEnergyHist[1]->Scale(1./ECalEnergyHist[1]->Integral());
      ECalEnergyHist[2]->Scale(1./ECalEnergyHist[2]->Integral());
      ECalEnergyHist[3]->Scale(1./ECalEnergyHist[3]->Integral());
      ECalEnergyHist[4]->Scale(1./ECalEnergyHist[4]->Integral());

      ECalEnergyHist[0]->SetLineColor(kRed);
      ECalEnergyHist[1]->SetLineColor(kBlue);
      ECalEnergyHist[2]->SetLineColor(kGreen);
      ECalEnergyHist[3]->SetLineColor(kCyan);
      ECalEnergyHist[4]->SetLineColor(kMagenta);

      ECalEnergyHist[0]->Draw("HIST");
      ECalEnergyHist[1]->Draw("HIST SAME");
      ECalEnergyHist[2]->Draw("HIST SAME");
      ECalEnergyHist[3]->Draw("HIST SAME");
      ECalEnergyHist[4]->Draw("HIST SAME");
      leg->Draw();

   c1.cd(2);
      ECalEnergyMomHist[0]->Scale(1./ECalEnergyMomHist[0]->Integral());
      ECalEnergyMomHist[1]->Scale(1./ECalEnergyMomHist[1]->Integral());
      ECalEnergyMomHist[2]->Scale(1./ECalEnergyMomHist[2]->Integral());
      ECalEnergyMomHist[3]->Scale(1./ECalEnergyMomHist[3]->Integral());
      ECalEnergyMomHist[4]->Scale(1./ECalEnergyMomHist[4]->Integral());

      ECalEnergyMomHist[0]->SetLineColor(kRed);
      ECalEnergyMomHist[1]->SetLineColor(kBlue);
      ECalEnergyMomHist[2]->SetLineColor(kGreen);
      ECalEnergyMomHist[3]->SetLineColor(kCyan);
      ECalEnergyMomHist[4]->SetLineColor(kMagenta);

      ECalEnergyMomHist[0]->Draw("HIST");
      ECalEnergyMomHist[1]->Draw("HIST SAME");
      ECalEnergyMomHist[2]->Draw("HIST SAME");
      ECalEnergyMomHist[3]->Draw("HIST SAME");
      ECalEnergyMomHist[4]->Draw("HIST SAME");
      leg->Draw();

   c1.cd(3);
      HCalEnergyHist[0]->Scale(1./HCalEnergyHist[0]->Integral());
      HCalEnergyHist[1]->Scale(1./HCalEnergyHist[1]->Integral());
      HCalEnergyHist[2]->Scale(1./HCalEnergyHist[2]->Integral());
      HCalEnergyHist[3]->Scale(1./HCalEnergyHist[3]->Integral());
      HCalEnergyHist[4]->Scale(1./HCalEnergyHist[4]->Integral());

      HCalEnergyHist[0]->SetLineColor(kRed);
      HCalEnergyHist[1]->SetLineColor(kBlue);
      HCalEnergyHist[2]->SetLineColor(kGreen);
      HCalEnergyHist[3]->SetLineColor(kCyan);
      HCalEnergyHist[4]->SetLineColor(kMagenta);

      HCalEnergyHist[0]->Draw("HIST");
      HCalEnergyHist[1]->Draw("HIST SAME");
      HCalEnergyHist[2]->Draw("HIST SAME");
      HCalEnergyHist[3]->Draw("HIST SAME");
      HCalEnergyHist[4]->Draw("HIST SAME");
      leg->Draw();

   c1.cd(4);
      HCalEnergyMomHist[0]->Scale(1./HCalEnergyMomHist[0]->Integral());
      HCalEnergyMomHist[1]->Scale(1./HCalEnergyMomHist[1]->Integral());
      HCalEnergyMomHist[2]->Scale(1./HCalEnergyMomHist[2]->Integral());
      HCalEnergyMomHist[3]->Scale(1./HCalEnergyMomHist[3]->Integral());
      HCalEnergyMomHist[4]->Scale(1./HCalEnergyMomHist[4]->Integral());

      HCalEnergyMomHist[0]->SetLineColor(kRed);
      HCalEnergyMomHist[1]->SetLineColor(kBlue);
      HCalEnergyMomHist[2]->SetLineColor(kGreen);
      HCalEnergyMomHist[3]->SetLineColor(kCyan);
      HCalEnergyMomHist[4]->SetLineColor(kMagenta);

      HCalEnergyMomHist[0]->Draw("HIST");
      HCalEnergyMomHist[1]->Draw("HIST SAME");
      HCalEnergyMomHist[2]->Draw("HIST SAME");
      HCalEnergyMomHist[3]->Draw("HIST SAME");
      HCalEnergyMomHist[4]->Draw("HIST SAME");
      leg->Draw();

   c1.SaveAs("Plots/Finaltest.pdf");
   c1.Clear();
      REtaPhiHist[0]->Scale(1./REtaPhiHist[0]->Integral());   
      REtaPhiHist[1]->Scale(1./REtaPhiHist[1]->Integral());
      REtaPhiHist[2]->Scale(1./REtaPhiHist[2]->Integral());
      REtaPhiHist[3]->Scale(1./REtaPhiHist[3]->Integral());
      REtaPhiHist[4]->Scale(1./REtaPhiHist[4]->Integral());

      REtaPhiHist[0]->SetLineColor(kRed);
      REtaPhiHist[1]->SetLineColor(kBlue);
      REtaPhiHist[2]->SetLineColor(kGreen);
      REtaPhiHist[3]->SetLineColor(kCyan);
      REtaPhiHist[4]->SetLineColor(kMagenta);

      REtaPhiHist[0]->Draw("HIST");
      REtaPhiHist[1]->Draw("HIST SAME");
      REtaPhiHist[2]->Draw("HIST SAME");
      REtaPhiHist[3]->Draw("HIST SAME");
      REtaPhiHist[4]->Draw("HIST SAME");
      leg->Draw();

   c1.SaveAs("Plots/Finaltest.pdf");
   
   c1.SaveAs("Plots/Finaltest.pdf]");

}


