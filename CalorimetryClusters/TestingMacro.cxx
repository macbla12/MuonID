#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TEfficiency.h>
#include <iostream>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TMath.h>
#include <string>
#include <TLegend.h>
#include <vector>
#include <tuple>
#include <onnxruntime_cxx_api.h>
#include <numeric>

#include "CalorimeterShapes.cxx"
#include "GreatCluster.cxx"

float run_muon_id_pipeline(
    Ort::Session& session,
    Ort::MemoryInfo& memory_info,
    double ECalEnergy_d, double HCalEnergy_d,
    double ECalNumber_d, double HCalNumber_d,
    double ECalEoverP_d, double HCalEoverP_d,
    const std::vector<float>& EcalShapeIn,
    const std::vector<float>& HcalShapeIn)
{
    float ECalEnergy = static_cast<float>(ECalEnergy_d);
    float HCalEnergy = static_cast<float>(HCalEnergy_d);
    float ECalNumber = static_cast<float>(ECalNumber_d);
    float HCalNumber = static_cast<float>(HCalNumber_d);
    float ECalEoverP = static_cast<float>(ECalEoverP_d);
    float HCalEoverP = static_cast<float>(HCalEoverP_d);

   // The model expects exactly 7 elements; enforce the same safeguard here
   // as in the production ROOT pipeline.
    std::vector<float> e = (EcalShapeIn.size() == 7 ? EcalShapeIn : std::vector<float>(7, 0.0f));
    std::vector<float> h = (HcalShapeIn.size() == 7 ? HcalShapeIn : std::vector<float>(7, 0.0f));

    int64_t shape_scalar[] = {1, 1};
    int64_t shape_vec[]    = {1, 7};

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, &ECalEnergy, 1, shape_scalar, 2));
    ort_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, &HCalEnergy, 1, shape_scalar, 2));
    ort_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, &ECalNumber, 1, shape_scalar, 2));
    ort_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, &HCalNumber, 1, shape_scalar, 2));
    ort_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, &ECalEoverP, 1, shape_scalar, 2));
    ort_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, &HCalEoverP, 1, shape_scalar, 2));
    ort_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, e.data(), e.size(), shape_vec, 2));
    ort_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, h.data(), h.size(), shape_vec, 2));

    static const char* input_names[] = {
        "ECalEnergy", "HCalEnergy", "ECalNumber", "HCalNumber",
        "ECalEoverP", "HCalEoverP", "EcalShape", "HcalShape"
    };
    static const char* output_names[] = {"probabilities"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, ort_inputs.data(), ort_inputs.size(),
        output_names, 1
    );

    float* probs = output_tensors[0].GetTensorMutableData<float>();
   return probs[1];  // P(muon), column 1, matching probabilities[:, 1] in Python
}

void TestingMacro()
{
    //////////////////////
    //Setting up constants
    //////////////////////

    static double MuonMass=0.1056583;
    static double ElectronMass=0.00051099895;
    static double PionMass=0.13957039;

    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");
    gStyle->SetOptStat(0);

    double DEG=180/TMath::Pi();

   // --- ONNX initialization ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "ONNX/xgb_muonID.onnx", session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    //////////////////////
    //Setting up histograms
    //////////////////////
    static constexpr int NumOfFiles=2;

    TH1D *AllParticEta[NumOfFiles], *AllParticPhi[NumOfFiles], *AllParticEnergy[NumOfFiles], *AllParticPt[NumOfFiles];
    TH1D *CutParticEta[NumOfFiles], *CutParticPhi[NumOfFiles], *CutParticEnergy[NumOfFiles], *CutParticPt[NumOfFiles];
    TH1D *FoundParticEta[NumOfFiles], *FoundParticPhi[NumOfFiles], *FoundParticEnergy[NumOfFiles], *FoundParticPt[NumOfFiles];
    TH1D *ECalEnergyHist[NumOfFiles], *ECalEnergyMomHist[NumOfFiles],*HCalEnergyHist[NumOfFiles], *HCalEnergyMomHist[NumOfFiles];
    TH2D *ECalEnergyvsMomHist[NumOfFiles],*HCalEnergyvsMomHist[NumOfFiles];
    TH2D *ECalEnergyMomvsEtaHist[NumOfFiles],  *HCalEnergyMomvsEtaHist[NumOfFiles];
    TH1D *XGBResponse[NumOfFiles], *XGBResponse_Stage2[NumOfFiles];


    vector<TString> files(NumOfFiles);


   files.at(0)="/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
   //files.at(0)="/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275*";
   //files.at(0)="/run/media/epic/Data/Background/JPsi/March/*.root";


   files.at(1)="/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
   //files.at(1)="/run/media/epic/Data/Tau/reco/Energy_10x275/old/double_pi/recoDoublePi.root";

   //files.at(1)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Electrons.root";
   //files.at(1)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Kaons.root";
   //files.at(1)="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Protons.root";



   TF1 *upperbondE = new TF1("upperbondE", "2/(x**2)+0.05", 0.001, 24.0);
   upperbondE->SetLineColor(kRed);
   upperbondE->SetLineWidth(1);

   TF1 *upperbondH = new TF1("upperbondH", "3.5/x",  0.001, 24.0);
   upperbondH->SetLineColor(kRed);
   upperbondH->SetLineWidth(1);

   TF1 *lowerbondH = new TF1("lowerbondH", "0.3/x-0.25/(x*x)",  0.001, 24.0);
   lowerbondH->SetLineColor(kRed);
   lowerbondH->SetLineWidth(1);

   for(int File=0; File<NumOfFiles;File++)
   {
      string name;
      if(File==0) name="Muons";
      if(File==1) name="Pions";

      // Set up input file chain
      TChain *mychain = new TChain("events");
      mychain->Add(files.at(File));

      // Initialize reader
      TTreeReader tree_reader(mychain);
      Long64_t nEvents = mychain->GetEntries();

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

      AllParticEta[File] = new TH1D(Form("AllParticEta%s",name.c_str()),Form("AllParticEta%s",name.c_str()),50,-1.2,3.4);
      AllParticPhi[File]= new TH1D(Form("AllParticPhi%s",name.c_str()),Form("AllParticPhi%s",name.c_str()),50,-180,180);
      AllParticEnergy[File]= new TH1D(Form("AllParticEnergy%s",name.c_str()),Form("AllParticEnergy%s",name.c_str()),50,0,20);
      AllParticPt[File]= new TH1D(Form("AllParticPt%s",name.c_str()),Form("AllParticPt%s",name.c_str()),50,0,20);

      CutParticEta[File] = new TH1D(Form("CutParticEta%s",name.c_str()),Form("CutParticEta%s",name.c_str()),50,-1.2,3.4);
      CutParticPhi[File]= new TH1D(Form("CutParticPhi%s",name.c_str()),Form("CutParticPhi%s",name.c_str()),50,-180,180);
      CutParticEnergy[File]= new TH1D(Form("CutParticEnergy%s",name.c_str()),Form("CutParticEnergy%s",name.c_str()),50,0,20);
      CutParticPt[File]= new TH1D(Form("CutParticPt%s",name.c_str()),Form("CutParticPt%s",name.c_str()),50,0,20);

      FoundParticEta[File] = new TH1D(Form("FoundParticEta%s",name.c_str()),Form("FoundParticEta%s",name.c_str()),50,-1.2,3.4);
      FoundParticPhi[File]= new TH1D(Form("FoundParticPhi%s",name.c_str()),Form("FoundParticPhi%s",name.c_str()),50,-180,180);
      FoundParticEnergy[File]= new TH1D(Form("FoundParticEnergy%s",name.c_str()),Form("FoundParticEnergy%s",name.c_str()),50,0,20);
      FoundParticPt[File]= new TH1D(Form("FoundParticPt%s",name.c_str()),Form("FoundParticPt%s",name.c_str()),50,0,20);

      //==================================//
      ECalEnergyHist[File]= new TH1D(Form("ECalEnergyHist%s",name.c_str()),Form("ECalEnergyHist%s",name.c_str()),50,0,15);
      ECalEnergyMomHist[File]= new TH1D(Form("ECalEnergyMomHist%s",name.c_str()),Form("ECalEnergyMomHist%s",name.c_str()),50,0,0.2);
      ECalEnergyvsMomHist[File]= new TH2D(Form("ECalEnergyvsMomHist%s",name.c_str()),Form("ECalEnergyvsMomHist%s",name.c_str()),50,0,22,50,0,2);
      ECalEnergyMomvsEtaHist[File]= new TH2D(Form("ECalEnergyMomvsEtaHist%s",name.c_str()),Form("ECalEnergyMomvsEtaist%s",name.c_str()),50,-3.5,3.5,50,0,2);


      HCalEnergyHist[File]= new TH1D(Form("HCalEnergyHist%s",name.c_str()),Form("HCalEnergyHist%s",name.c_str()),50,0,15);
      HCalEnergyMomHist[File]= new TH1D(Form("HCalEnergyMomHist%s",name.c_str()),Form("HCalEnergyMomHist%s",name.c_str()),50,0,4);
      HCalEnergyvsMomHist[File]= new TH2D(Form("HCalEnergyvsMomHist%s",name.c_str()),Form("HCalEnergyvsMomHist%s",name.c_str()),50,0,22,50,0,2);
      HCalEnergyMomvsEtaHist[File]= new TH2D(Form("HCalEnergyMomvsEtaHist%s",name.c_str()),Form("HCalEnergyMomvsEtaist%s",name.c_str()),50,-3.5,3.5,50,0,2);

      //==============================//
      XGBResponse[File] = new TH1D(Form("XGBResponse%s",name.c_str()),Form("XGBResponse%s",name.c_str()),100,0,1);
      XGBResponse_Stage2[File] = new TH1D(Form("XGBResponse_Stage2%s",name.c_str()),Form("XGBResponse_Stage2%s",name.c_str()),100,0,1);


      //Long64_t startEvent = 0.9 * nEvents;
      //tree_reader.SetEntry(startEvent);

      int eventID=0;
      double FoundParticles=0;
      double particscount=0;
      double BadPDG=0;
      double aftercuts=0,secondcuts=0;
      double CaloHit=0;



      while(tree_reader.Next()){
         eventID++;
         //if(eventID>2000) break;


         int id=0;
         for(int particle=0; particle<trackEng.GetSize();particle++)
         {
            double ECalEnergy=0, HCalEnergy=0, ECalNumber=0, HCalNumber=0;
            std::vector<float> EcalShape, HcalShape;
            particscount++;
            //Obligatory Cuts
            CaloHit=0;
            double mass;
            if(File==0) mass=MuonMass;
            else if(File==1) mass=ElectronMass;
            else if(File==2) mass=PionMass;

            int Found=0;
            TLorentzVector Partic;
            Partic.SetPxPyPzE(trackMomX[particle],trackMomY[particle],trackMomZ[particle],trackEng[particle]);
            if(Partic.Theta()>177) continue;
            if(abs(Partic.Eta())<1.3 && abs(Partic.Eta())>1) continue;
            if(Partic.Eta()<-1.25) continue;
            if(Partic.P()<1) continue;

           //Ecal Energy Search
            int simuID = simuAssoc[particle];

            //////////////////////
            // Collect energies and shapes from all ECal detectors
            //////////////////////
            vector<vector<float>> EcalAllShapes;
            //cout<<"Tutaj EcalBarrel"<<endl;

            auto [EnergyEcalBarrel,NumberEcalBarrel,ShapeEcalBarrel] = Calorimeter( simuID, EcalBarrelEng, simuAssocEcalBarrel, EcalBarrelx, EcalBarrely,
                EcalBarrelz, EcalBarrelShPB, EcalBarrelShPE,EcalBarrelShParameters);

            ECalEnergy+=EnergyEcalBarrel;

            if(!ShapeEcalBarrel.empty() && !ShapeEcalBarrel[0].empty() && ShapeEcalBarrel[0][0] != 0){
               ECalNumber+=NumberEcalBarrel;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeEcalBarrel.begin(), ShapeEcalBarrel.end());
            }


            auto [EnergyEndcapP,NumberEndcapP,ShapeEndcapP] = Calorimeter( simuID, EcalEndcapPEng, simuAssocEcalEndcapP, EcalEndcapPx, EcalEndcapPy,
                EcalEndcapPz, EcalEndcapPShPB, EcalEndcapPShPE,EcalEndcapPShParameters);
            ECalEnergy+=EnergyEndcapP;

            if(!ShapeEndcapP.empty() && !ShapeEndcapP[0].empty() && ShapeEndcapP[0][0] != 0){
               ECalNumber+=NumberEndcapP;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeEndcapP.begin(), ShapeEndcapP.end());
            }

            auto [EnergyEndcapN,NumberEndcapN,ShapeEndcapN] = Calorimeter( simuID, EcalEndcapNEng, simuAssocEcalEndcapN, EcalEndcapNx, EcalEndcapNy,
                EcalEndcapNz, EcalEndcapNShPB, EcalEndcapNShPE,EcalEndcapNShParameters);

            ECalEnergy+=EnergyEndcapN;

            if(!ShapeEndcapN.empty() && !ShapeEndcapN[0].empty() && ShapeEndcapN[0][0] != 0){
               ECalNumber+=NumberEndcapN;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeEndcapN.begin(), ShapeEndcapN.end());
            }

            auto [EnergyB0,NumberB0,ShapeB0] = Calorimeter( simuID, B0Eng, simuAssocB0, B0x, B0y, B0z, B0ShPB, B0ShPE,B0ShParameters);

            ECalEnergy+=EnergyB0;

            if(!ShapeB0.empty() && !ShapeB0[0].empty() && ShapeB0[0][0] != 0){
               ECalNumber+=NumberB0;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeB0.begin(), ShapeB0.end());
            }

            auto [EnergyImaging,NumberImaging,ShapeImaging] = Calorimeter( simuID, EcalBarrelImagingEng, simuAssocEcalBarrelImaging, EcalBarrelImagingx, EcalBarrelImagingy,
                EcalBarrelImagingz, EcalBarrelImagingShPB, EcalBarrelImagingShPE,EcalBarrelImagingShParameters);

            ECalEnergy+=EnergyImaging;

            if(!ShapeImaging.empty() && !ShapeImaging[0].empty() && ShapeImaging[0][0] != 0){
               ECalNumber+=NumberImaging;
               EcalAllShapes.insert(EcalAllShapes.end(), ShapeImaging.begin(), ShapeImaging.end());
            }

            auto [EnergyScFi,NumberScFi,ShapeScFi] = Calorimeter( simuID, EcalBarrelScFiEng, simuAssocEcalBarrelScFi, EcalBarrelScFix, EcalBarrelScFiy,
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
               EcalShape = GreatCluster(EcalAllShapes);
               CaloHit=1;
            }
            else EcalShape = vector<float>(7, 0.0f);

            //////////////////////
            //Hcal Energy Search
            //////////////////////
            //cout<<"Tutaj ShapeHcalBarrel"<<endl;
            vector<vector<float>> HcalAllShapes;

            auto [EnergyHcalBarrel,NumberHcalBarrel,ShapeHcalBarrel] = Calorimeter( simuID, HcalBarrelEng, simuAssocHcalBarrel, HcalBarrelx, HcalBarrely,
                HcalBarrelz, HcalBarrelShPB, HcalBarrelShPE,HcalBarrelShParameters);

            HCalEnergy+=EnergyHcalBarrel;

            if(!ShapeHcalBarrel.empty() && !ShapeHcalBarrel[0].empty() && ShapeHcalBarrel[0][0] != 0){
               HCalNumber+=NumberHcalBarrel;
               HcalAllShapes.insert(HcalAllShapes.end(), ShapeHcalBarrel.begin(), ShapeHcalBarrel.end());
            }

            auto [EnergyHcalEndcapP,NumberHcalEndcapP,ShapeHcalEndcapP] = Calorimeter( simuID, HcalEndcapPEng, simuAssocHcalEndcapP, HcalEndcapPx, HcalEndcapPy,
                HcalEndcapPz, HcalEndcapPShPB, HcalEndcapPShPE,HcalEndcapPShParameters);

            HCalEnergy+=EnergyHcalEndcapP;

            if(!ShapeHcalEndcapP.empty() && !ShapeHcalEndcapP[0].empty() && ShapeHcalEndcapP[0][0] != 0){
               HCalNumber+=NumberHcalEndcapP;
               HcalAllShapes.insert(HcalAllShapes.end(), ShapeHcalEndcapP.begin(), ShapeHcalEndcapP.end());
            }

            auto [EnergyLFHcal,NumberLFHcal,ShapeLFHcal] = Calorimeter( simuID, LFHcalEng, simuAssocLFHcal, LFHcalx, LFHcaly, LFHcalz, LFHcalShPB, LFHcalShPE,LFHcalShParameters);

            HCalEnergy+=EnergyLFHcal;

            if(!ShapeLFHcal.empty() && !ShapeLFHcal[0].empty() && ShapeLFHcal[0][0] != 0){
               HCalNumber+=NumberLFHcal;
               HcalAllShapes.insert(HcalAllShapes.end(), ShapeLFHcal.begin(), ShapeLFHcal.end());
            }

            auto [EnergyHcalEndcapN,NumberHcalEndcapN,ShapeHcalEndcapN] = Calorimeter( simuID, HcalEndcapNEng, simuAssocHcalEndcapN, HcalEndcapNx, HcalEndcapNy,
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
               CaloHit=1;
            }
            else HcalShape = vector<float>(7, 0.0f);

            //Track properties
            double FullEnergy=HCalEnergy+ECalEnergy;
            if(FullEnergy==0) continue;
            FoundParticles+=1;


            double Momentum=Partic.P();
            double HCalEoverP=HCalEnergy/Momentum;
            double ECalEoverP=ECalEnergy/Momentum;

            AllParticEnergy[File]->Fill(Partic.P());
            AllParticEta[File]->Fill(Partic.Eta());
            AllParticPhi[File]->Fill(Partic.Phi()*DEG);
            AllParticPt[File]->Fill(Partic.Perp());


            //if(!(trackPDG[particle]==0 || abs(trackPDG[particle])==13)) continue;

            if(HCalEoverP<upperbondH->Eval(Momentum) && HCalEoverP>lowerbondH->Eval(Momentum) && ECalEoverP<upperbondE->Eval(Momentum)){

               aftercuts++;
               if(File==0){
                  CutParticEta[File] ->Fill(Partic.Eta());
                  CutParticPhi[File]->Fill(Partic.Phi()*DEG);
                  CutParticEnergy[File]->Fill(Partic.P());
                  CutParticPt[File]->Fill(Partic.Perp());
               }

            }
            else{
               if(File==1){
                  CutParticEta[File] ->Fill(Partic.Eta());
                  CutParticPhi[File]->Fill(Partic.Phi()*DEG);
                  CutParticEnergy[File]->Fill(Partic.P());
                  CutParticPt[File]->Fill(Partic.Perp());
               }

            }

            if(CaloHit==0){

                  if(HCalEoverP<upperbondH->Eval(Momentum) && HCalEoverP>lowerbondH->Eval(Momentum) && ECalEoverP<upperbondE->Eval(Momentum)){

                  secondcuts++;
                  if(File==0){
                     FoundParticEta[File] ->Fill(Partic.Eta());
                     FoundParticPhi[File]->Fill(Partic.Phi()*DEG);
                     FoundParticEnergy[File]->Fill(Partic.P());
                     FoundParticPt[File]->Fill(Partic.Perp());
                  }

               }
               else{
                  if(File==1){
                     FoundParticEta[File] ->Fill(Partic.Eta());
                     FoundParticPhi[File]->Fill(Partic.Phi()*DEG);
                     FoundParticEnergy[File]->Fill(Partic.P());
                     FoundParticPt[File]->Fill(Partic.Perp());
                  }

               }

            }
            else{

               float muon_prob = run_muon_id_pipeline(session, memory_info,
                    ECalEnergy, HCalEnergy, ECalNumber, HCalNumber,
                    ECalEoverP, HCalEoverP, EcalShape, HcalShape
                );
               XGBResponse[File]->Fill(muon_prob);



               if(muon_prob>0.4){


                  secondcuts++;
                  if(File==0){
                     FoundParticEta[File] ->Fill(Partic.Eta());
                     FoundParticPhi[File]->Fill(Partic.Phi()*DEG);
                     FoundParticEnergy[File]->Fill(Partic.P());
                     FoundParticPt[File]->Fill(Partic.Perp());
                  }

                  ECalEnergyMomvsEtaHist[File]->Fill(Partic.Eta(),ECalEoverP);
                  HCalEnergyMomvsEtaHist[File]->Fill(Partic.Eta(),HCalEoverP);

                  ECalEnergyvsMomHist[File]->Fill(Momentum,ECalEoverP);
                  HCalEnergyvsMomHist[File]->Fill(Momentum,HCalEoverP);

               }
               else{
                  if(File==1){
                     FoundParticEta[File] ->Fill(Partic.Eta());
                     FoundParticPhi[File]->Fill(Partic.Phi()*DEG);
                     FoundParticEnergy[File]->Fill(Partic.P());
                     FoundParticPt[File]->Fill(Partic.Perp());
                  }

               }
            }

         }

      }


      cout<<"==========================="<<endl;
      cout<<"End of "<< name << " file"<<endl;
      cout<<"Number of events: "<<eventID<<endl;
      cout<<"Found particles: "<<FoundParticles<<"   All particles: "<<particscount<<endl;
      cout<<"Found Ratio: "<<FoundParticles*100/particscount<<'%'<<endl;
      if(File==0) cout<<"After First Cuts Ratio: "<<aftercuts*100/FoundParticles<<'%'<<endl;
      if(File==1) cout<<"After First Cuts Ratio: "<<100-(aftercuts*100/FoundParticles)<<'%'<<endl;
      cout<<"   After first cut particles: "<<aftercuts<<endl;
      if(File==0)    cout<<"After Second Cuts Ratio: "<<secondcuts*100/FoundParticles<<'%'<<endl;
      if(File==1)    cout<<"After Second Cuts Ratio: "<<100-(secondcuts*100/FoundParticles)<<'%'<<endl;

      cout<<"   After second cuts particles: "<<secondcuts<<endl;
      cout<<"==========================="<<endl;

      delete mychain;
   }

   // =========================================================================
   // Save all histograms and TEfficiency objects to a .root file
   // instead of drawing and saving canvases to PDF.
   // =========================================================================

   TFile *outfile = new TFile("Plots/FinalCalID.root", "RECREATE");
   outfile->cd();

   for (int File = 0; File < NumOfFiles; File++)
   {
      AllParticEta[File]->Write();
      AllParticPhi[File]->Write();
      AllParticEnergy[File]->Write();
      AllParticPt[File]->Write();

      CutParticEta[File]->Write();
      CutParticPhi[File]->Write();
      CutParticEnergy[File]->Write();
      CutParticPt[File]->Write();

      FoundParticEta[File]->Write();
      FoundParticPhi[File]->Write();
      FoundParticEnergy[File]->Write();
      FoundParticPt[File]->Write();

      ECalEnergyvsMomHist[File]->Write();
      ECalEnergyMomvsEtaHist[File]->Write();
      HCalEnergyvsMomHist[File]->Write();
      HCalEnergyMomvsEtaHist[File]->Write();

      XGBResponse[File]->Write();
      XGBResponse_Stage2[File]->Write();
   }

   // --- Cut curves (E/p bounds), useful for comparisons when reading the file ---
   upperbondE->Write("upperbondE");
   upperbondH->Write("upperbondH");
   lowerbondH->Write("lowerbondH");

   // --- Efficiency / Rejection vs p ---
   TEfficiency *pEff1 = new TEfficiency(*CutParticEnergy[0], *AllParticEnergy[0]);
   pEff1->SetName("Eff_EoverPcut_vs_P_Muon");
   pEff1->SetStatisticOption(TEfficiency::kBUniform);
   pEff1->Write();

   TEfficiency *pEff2 = new TEfficiency(*FoundParticEnergy[0], *AllParticEnergy[0]);
   pEff2->SetName("Eff_XGBoost_vs_P_Muon");
   pEff2->SetStatisticOption(TEfficiency::kBUniform);
   pEff2->Write();

   TEfficiency *pEff0 = new TEfficiency(*CutParticEnergy[1], *AllParticEnergy[1]);
   pEff0->SetName("Rej_EoverPcut_vs_P_Pion");
   pEff0->SetStatisticOption(TEfficiency::kBUniform);
   pEff0->Write();

   TEfficiency *pEff3 = new TEfficiency(*FoundParticEnergy[1], *AllParticEnergy[1]);
   pEff3->SetName("Rej_XGBoost_vs_P_Pion");
   pEff3->SetStatisticOption(TEfficiency::kBUniform);
   pEff3->Write();

   // --- Efficiency / Rejection vs eta ---
   TEfficiency *pEffEta1 = new TEfficiency(*CutParticEta[0], *AllParticEta[0]);
   pEffEta1->SetName("Eff_EoverPcut_vs_Eta_Muon");
   pEffEta1->SetStatisticOption(TEfficiency::kBUniform);
   pEffEta1->Write();

   TEfficiency *pEffEta3 = new TEfficiency(*FoundParticEta[0], *AllParticEta[0]);
   pEffEta3->SetName("Eff_XGBoost_vs_Eta_Muon");
   pEffEta3->SetStatisticOption(TEfficiency::kBUniform);
   pEffEta3->Write();

   TEfficiency *pEffEta0 = new TEfficiency(*CutParticEta[1], *AllParticEta[1]);
   pEffEta0->SetName("Rej_EoverPcut_vs_Eta_Pion");
   pEffEta0->SetStatisticOption(TEfficiency::kBUniform);
   pEffEta0->Write();

   TEfficiency *pEffEta2 = new TEfficiency(*FoundParticEta[1], *AllParticEta[1]);
   pEffEta2->SetName("Rej_XGBoost_vs_Eta_Pion");
   pEffEta2->SetStatisticOption(TEfficiency::kBUniform);
   pEffEta2->Write();

   // --- Efficiency / Rejection vs pT ---
   TEfficiency *pEffPt1 = new TEfficiency(*CutParticPt[0], *AllParticPt[0]);
   pEffPt1->SetName("Eff_EoverPcut_vs_Pt_Muon");
   pEffPt1->SetStatisticOption(TEfficiency::kBUniform);
   pEffPt1->Write();

   TEfficiency *pEffPt2 = new TEfficiency(*FoundParticPt[0], *AllParticPt[0]);
   pEffPt2->SetName("Eff_XGBoost_vs_Pt_Muon");
   pEffPt2->SetStatisticOption(TEfficiency::kBUniform);
   pEffPt2->Write();

   TEfficiency *pEffPt0 = new TEfficiency(*CutParticPt[1], *AllParticPt[1]);
   pEffPt0->SetName("Rej_EoverPcut_vs_Pt_Pion");
   pEffPt0->SetStatisticOption(TEfficiency::kBUniform);
   pEffPt0->Write();

   TEfficiency *pEffPt3 = new TEfficiency(*FoundParticPt[1], *AllParticPt[1]);
   pEffPt3->SetName("Rej_XGBoost_vs_Pt_Pion");
   pEffPt3->SetStatisticOption(TEfficiency::kBUniform);
   pEffPt3->Write();

   outfile->Write();
   outfile->Close();

   cout << "All histograms and TEfficiency objects saved to Plots/FinalCalID.root" << endl;
}