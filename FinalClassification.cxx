
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
#include <onnxruntime_cxx_api.h>
#include <numeric>

#include "Calorimeternew.cxx"
#include "GreatCluster.cxx"

std::vector<float> load_line(const std::string& path, int line_no)
{
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Nie mogę otworzyć scalars.txt");

    std::string line;
    for (int i = 0; i <= line_no; i++)
        std::getline(in, line);

    std::vector<float> vals;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ',')) {
        vals.push_back(std::stof(item));
    }

    return vals;
}

auto scaler_mean  = load_line("ONNX/scalars.txt", 0);
auto scaler_scale = load_line("ONNX/scalars.txt", 1);


inline float safe_div(float a, float b) {
    return (std::abs(b) > 1e-12f ? a / b : 0.0f);
}

std::vector<float> prepare_45_features(
    float ECalEnergy,
    float HCalEnergy,
    float ECalNumber,
    float HCalNumber,
    float ECalEoverP,
    float HCalEoverP,
    const std::vector<float>& eS,
    const std::vector<float>& hS)
{
    std::vector<float> X;
    X.reserve(45);

    // --- 0. Bezpieczne shape’y ---
    std::vector<float> e = (eS.size() == 7 ? eS : std::vector<float>(7, 0.0f));
    std::vector<float> h = (hS.size() == 7 ? hS : std::vector<float>(7, 0.0f));

    // ============================================================
    // 1. SCALAR FEATURES (16)
    // ============================================================

    // --- Surowe ---
    X.push_back(ECalEnergy);   // 0
    X.push_back(HCalEnergy);   // 1
    X.push_back(ECalNumber);   // 2
    X.push_back(HCalNumber);   // 3
    X.push_back(ECalEoverP);   // 4
    X.push_back(HCalEoverP);   // 5    // 6

    // --- Pochodne ---
    float totalE = ECalEnergy + HCalEnergy;

    X.push_back(safe_div(ECalEnergy, totalE));  // 7  ECalFrac
    X.push_back(safe_div(HCalEnergy, totalE));  // 8  HCalFrac

    X.push_back(safe_div(ECalNumber, HCalNumber)); // 9 HitRatio

    X.push_back(safe_div(ECalEoverP, HCalEoverP)); // 10 EoverP_ratio

    X.push_back(safe_div(ECalEnergy, ECalNumber)); // 11 ECalDensity
    X.push_back(safe_div(HCalEnergy, HCalNumber)); // 12 HCalDensity

    X.push_back(std::log1p(ECalEnergy));   // 14 logECal
    X.push_back(std::log1p(HCalEnergy));   // 15 logHCal

    // ============================================================
    // 2. SHAPE FEATURES (17 derived + 14 raw = 31)
    // ============================================================

    // --- Derived shape ---
    float e_trans = std::sqrt(std::max(0.0f, e[4] * e[5]));
    float h_trans = std::sqrt(std::max(0.0f, h[4] * h[5]));

    float e_long = e[6];
    float h_long = h[6];

    float e_LoverT = safe_div(e_long, e_trans);
    float h_LoverT = safe_div(h_long, h_trans);

    float e_sph = safe_div(e[4], e[6]);
    float h_sph = safe_div(h[4], h[6]);

    float e_asym = safe_div(e[2] - e[3], e[2] + e[3]);
    float h_asym = safe_div(h[2] - h[3], h[2] + h[3]);

    X.push_back(e_trans);     // 16
    X.push_back(h_trans);     // 17
    X.push_back(e_long);      // 18
    X.push_back(h_long);      // 19
    X.push_back(e_LoverT);    // 20
    X.push_back(h_LoverT);    // 21
    X.push_back(e_sph);       // 22
    X.push_back(h_sph);       // 23
    X.push_back(e_asym);      // 24
    X.push_back(h_asym);      // 25

    X.push_back(safe_div(e[0], h[0]));      // 26 radius_ratio
    X.push_back(safe_div(e[1], h[1]));      // 27 disp_ratio
    X.push_back(safe_div(e_trans, h_trans)); // 28 trans_ratio
    X.push_back(safe_div(e_long, h_long));   // 29 long_ratio

    X.push_back(std::abs(e_LoverT - h_LoverT)); // 30 LoverT_mismatch
    X.push_back(std::abs(e_sph - h_sph));       // 31 sphericity_mismatch

    X.push_back(safe_div(h[0], e[0] + h[0]));   // 32 Radial_HCal_Fraction

    // --- Raw shapes (14) ---
    for (float v : e) X.push_back(v); // 33–39
    for (float v : h) X.push_back(v); // 40–46

    // ============================================================
    // 3. SKALOWANIE (47 cech)
    // ============================================================
    for (size_t i = 0; i < X.size(); i++)
        X[i] = (X[i] - scaler_mean[i]) / scaler_scale[i];

    return X;
}


void FinalClassification()
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

    // --- INICJALIZACJA ONNX (Dodaj tutaj) ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "ONNX/xgb_muonID.onnx", session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Nazwy wejść/wyjść (zależą od konwertera, zazwyczaj "input" i "probabilities")
    const char* input_names[] = {"input"};
    const char* output_names[] = {"probabilities"};

    // --- INICJALIZACJA ONNX (Dodaj tutaj) ---
    Ort::Session session2(env, "ONNX/xgb_muonID_stage2.onnx", session_options);

    // Nazwy wejść/wyjść (zależą od konwertera, zazwyczaj "input" i "probabilities")
    const char* input_names2[] = {"input"};
    const char* output_names2[] = {"probabilities"};


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
   //files.at(0)="/run/media/epic/Data/Muons/Grape-10x275/Paper/RECO/*.root";
   //files.at(0)="/run/media/epic/Data/Background/JPsi/March/*.root";


   files.at(1)="/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
   //files.at(1)="/run/media/epic/Data/Tau/reco/Energy_10x275/double_pi/recoDoublePi.root";

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

      AllParticEta[File] = new TH1D(Form("AllParticEta%s",name.c_str()),Form("AllParticEta%s",name.c_str()),30,-1.2,3.4);
      AllParticPhi[File]= new TH1D(Form("AllParticPhi%s",name.c_str()),Form("AllParticPhi%s",name.c_str()),30,-180,180);
      AllParticEnergy[File]= new TH1D(Form("AllParticEnergy%s",name.c_str()),Form("AllParticEnergy%s",name.c_str()),40,0,20);
      AllParticPt[File]= new TH1D(Form("AllParticPt%s",name.c_str()),Form("AllParticPt%s",name.c_str()),30,0,20);
      
      CutParticEta[File] = new TH1D(Form("CutParticEta%s",name.c_str()),Form("CutParticEta%s",name.c_str()),30,-1.2,3.4);
      CutParticPhi[File]= new TH1D(Form("CutParticPhi%s",name.c_str()),Form("CutParticPhi%s",name.c_str()),30,-180,180);
      CutParticEnergy[File]= new TH1D(Form("CutParticEnergy%s",name.c_str()),Form("CutParticEnergy%s",name.c_str()),40,0,20);
      CutParticPt[File]= new TH1D(Form("CutParticPt%s",name.c_str()),Form("CutParticPt%s",name.c_str()),30,0,20);

      FoundParticEta[File] = new TH1D(Form("FoundParticEta%s",name.c_str()),Form("FoundParticEta%s",name.c_str()),30,-1.2,3.4);
      FoundParticPhi[File]= new TH1D(Form("FoundParticPhi%s",name.c_str()),Form("FoundParticPhi%s",name.c_str()),30,-180,180);
      FoundParticEnergy[File]= new TH1D(Form("FoundParticEnergy%s",name.c_str()),Form("FoundParticEnergy%s",name.c_str()),40,0,20);
      FoundParticPt[File]= new TH1D(Form("FoundParticPt%s",name.c_str()),Form("FoundParticPt%s",name.c_str()),30,0,20);
      
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
         //if(eventID>200000) break;
         

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
               EcalShape = GreatCluster(EcalAllShapes);
               CaloHit=1;
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
               CaloHit=1;
            }
            
            
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

               std::vector<float> feats = prepare_45_features(ECalEnergy, HCalEnergy, ECalNumber, HCalNumber, ECalEoverP, HCalEoverP, EcalShape, HcalShape);

               // 2. Stwórz tensor wejściowy
               int64_t input_shape[] = {1, 45};
               Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                  memory_info, feats.data(), feats.size(), input_shape, 2
               );

               // 3. Uruchom model
               auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                                input_names, &input_tensor, 1, 
                                                output_names, 1);

               // 4. Pobierz wynik (prawdopodobieństwo miona)
               float* probs = output_tensors[0].GetTensorMutableData<float>();
               float muon_prob = probs[1]; 
               XGBResponse[File]->Fill(muon_prob);


               

               if(muon_prob>0.86){ 

                  // 3. Uruchom model
                  auto output_tensors2 = session2.Run(Ort::RunOptions{nullptr}, 
                                                   input_names, &input_tensor, 1, 
                                                   output_names, 1);

                  // 4. Pobierz wynik (prawdopodobieństwo miona)
                  float* probs2 = output_tensors2[0].GetTensorMutableData<float>();
                  float muon_prob2 = probs2[1]; 
                  XGBResponse_Stage2[File]->Fill(muon_prob2);
         

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
   }
   
   gStyle->SetOptStat(111111);
   //gStyle->SetOptStat(000000);

   TCanvas c1;

   

   TLegend* leg1 = new TLegend(0.78, 0.8, 0.95, 0.95);
    leg1->SetBorderSize(0);
    leg1->SetNColumns(1);
    leg1->SetColumnSeparation(0.1);
    leg1->SetEntrySeparation(0.1);
    leg1->SetMargin(0.15);
    leg1->SetTextFont(42);
    leg1->SetTextSize(0.05);
    leg1->AddEntry(AllParticEta[0],"All Partics ","l");
    leg1->AddEntry(FoundParticEta[0],"Found Partics","l");

   TLegend* leg2 = new TLegend(0.78, 0.8, 0.95, 0.95);
    leg2->SetBorderSize(0);
    leg2->SetNColumns(1);
    leg2->SetColumnSeparation(0.1);
    leg2->SetEntrySeparation(0.1);
    leg2->SetMargin(0.15);
    leg2->SetTextFont(42);
    leg2->SetTextSize(0.05);
    leg2->AddEntry(XGBResponse[0],"Muons ","l");
    leg2->AddEntry(XGBResponse[1],"Pions","l");
   c1.SaveAs("Plots/FinalCalID.pdf[");
   c1.Clear();
   //gPad->SetLogy(1);
   XGBResponse[0]->Scale(1/XGBResponse[0]->Integral());
   XGBResponse[1]->Scale(1/XGBResponse[1]->Integral());
   XGBResponse[0]->SetLineColor(kBlue);
   XGBResponse[1]->SetLineColor(kRed);
   XGBResponse[0]->Draw("HIST");
   XGBResponse[1]->Draw("HIST SAME");
   c1.SaveAs("Plots/FinalCalID.pdf");

   c1.Clear();
   //gPad->SetLogy(1);
   XGBResponse_Stage2[0]->Scale(1/XGBResponse_Stage2[0]->Integral());
   XGBResponse_Stage2[1]->Scale(1/XGBResponse_Stage2[1]->Integral());
   XGBResponse_Stage2[0]->SetLineColor(kBlue);
   XGBResponse_Stage2[1]->SetLineColor(kRed);
   XGBResponse_Stage2[0]->Draw("HIST");
   XGBResponse_Stage2[1]->Draw("HIST SAME");
   c1.SaveAs("Plots/FinalCalID.pdf");
   //gPad->SetLogy(0);


   c1.Clear();
   c1.Divide(2,2);
   c1.cd(1);
      AllParticEta[0]->SetLineColor(kBlue);
      FoundParticEta[0]->SetLineColor(kRed);
      CutParticEta[0]->SetLineColor(kGreen+1);
      CutParticEta[0]->SetLineWidth(2);
      AllParticEta[0]->Draw();
      FoundParticEta[0]->Draw("same");
   c1.cd(2);
      AllParticPhi[0]->SetLineColor(kBlue);
      FoundParticPhi[0]->SetLineColor(kRed);
      CutParticPhi[0]->SetLineColor(kGreen+1);
      CutParticPhi[0]->SetLineWidth(2);
      AllParticPhi[0]->SetMinimum(0);
      AllParticPhi[0]->Draw();
      FoundParticPhi[0]->Draw("same");
      CutParticPhi[0]->Draw("same");
   c1.cd(3); 
      AllParticEnergy[0]->SetLineColor(kBlue);
      FoundParticEnergy[0]->SetLineColor(kRed);
      CutParticEnergy[0]->SetLineColor(kGreen+1);
      CutParticEnergy[0]->SetLineWidth(2);
      AllParticEnergy[0]->Draw();
      FoundParticEnergy[0]->Draw("same");
      CutParticEnergy[0]->Draw("same");
   c1.cd(4); 
      gPad->SetLogy(1);
      AllParticPt[0]->SetLineColor(kBlue);
      FoundParticPt[0]->SetLineColor(kRed);
      CutParticPt[0]->SetLineColor(kGreen+1);
      CutParticPt[0]->SetLineWidth(2);
      AllParticPt[0]->SetMinimum(0.1);
      AllParticPt[0]->Draw();
      FoundParticPt[0]->Draw("same");
      CutParticPt[0]->Draw("same");    
   c1.SaveAs("Plots/FinalCalID.pdf");
   c1.Clear();

   c1.Divide(2,2);
   c1.cd(1);
      gPad->SetLogy(1);
      AllParticEta[1]->SetLineColor(kBlue);
      FoundParticEta[1]->SetLineColor(kRed);
      CutParticEta[1]->SetLineColor(kGreen+1);
      CutParticEta[1]->SetLineWidth(2);
      AllParticEta[1]->SetMinimum(0.1);

      AllParticEta[1]->Draw();
      FoundParticEta[1]->Draw("same");
      CutParticEta[1]->Draw("same");
   c1.cd(2);
      gPad->SetLogy(1);
      AllParticPhi[1]->SetLineColor(kBlue);
      FoundParticPhi[1]->SetLineColor(kRed);
      CutParticPhi[1]->SetLineColor(kGreen+1);
      CutParticPhi[1]->SetLineWidth(2);
      AllParticPhi[1]->SetMinimum(0.1);
      AllParticPhi[1]->Draw();
      FoundParticPhi[1]->Draw("same");
      CutParticPhi[1]->Draw("same");
   c1.cd(3); 
      gPad->SetLogy(1);
      AllParticEnergy[1]->SetLineColor(kBlue);
      FoundParticEnergy[1]->SetLineColor(kRed);
      CutParticEnergy[1]->SetLineColor(kGreen+1);
      CutParticEnergy[1]->SetLineWidth(2);
      AllParticEnergy[1]->SetMinimum(0.1);
      AllParticEnergy[1]->Draw();
      FoundParticEnergy[1]->Draw("same");
      CutParticEnergy[1]->Draw("same"); 
   c1.cd(4); 
      gPad->SetLogy(1);
      AllParticPt[1]->SetLineColor(kBlue);
      FoundParticPt[1]->SetLineColor(kRed);
      CutParticPt[1]->SetLineColor(kGreen+1);
      CutParticPt[1]->SetLineWidth(2);
      AllParticPt[1]->SetMinimum(0.1);
      AllParticPt[1]->Draw();
      FoundParticPt[1]->Draw("same");
      CutParticPt[1]->Draw("same"); 
   


   c1.SaveAs("Plots/FinalCalID.pdf");
   gPad->SetLogy(0);

   c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      ECalEnergyMomvsEtaHist[0]->Draw("colz");
   c1.cd(2);
      ECalEnergyMomvsEtaHist[1]->Draw("colz");
   c1.SaveAs("Plots/FinalCalID.pdf");

   c1.Clear();
   c1.Divide(2,1);
   c1.cd(1);
      HCalEnergyMomvsEtaHist[0]->Draw("colz");
   c1.cd(2);
      HCalEnergyMomvsEtaHist[1]->Draw("colz");
   c1.SaveAs("Plots/FinalCalID.pdf");

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
   c1.SaveAs("Plots/FinalCalID.pdf");
   gPad->SetLogz(0);

  
   c1.Clear();
   c1.Divide(2,1);
   
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
   c1.SaveAs("Plots/FinalCalID.pdf");
   c1.SaveAs("Plots/FinalCalID.pdf]");

   gPad->SetLogz(0);
   c1.Clear();
   gPad->SetLogz(1);

   gStyle->SetOptStat(0);          // Usuwamy to brzydkie pudełko ze statystykami
   gPad->SetRightMargin(0.15);     // Miejsce na skalę kolorów (Z axis)
   gPad->SetLeftMargin(0.12);
   gPad->SetBottomMargin(0.12);

   c1.Clear();

   // --- Wykres główny (Histogram 2D) ---
   HCalEnergyvsMomHist[1]->SetTitle(""); // Tytuł lepiej dodać przez TLatex dla kontroli
   HCalEnergyvsMomHist[1]->GetXaxis()->SetTitle("Momentum [GeV]");
   HCalEnergyvsMomHist[1]->GetYaxis()->SetTitle("HCal Energy/Momentum");
   HCalEnergyvsMomHist[1]->GetXaxis()->SetTitleSize(0.05);
   HCalEnergyvsMomHist[1]->GetYaxis()->SetTitleSize(0.05);
   HCalEnergyvsMomHist[1]->GetXaxis()->SetLabelSize(0.04);
   HCalEnergyvsMomHist[1]->GetYaxis()->SetLabelSize(0.04);

   // Rysowanie z opcją COLZ (mapa ciepła z legendą kolorów)
   HCalEnergyvsMomHist[1]->Draw("colz"); 

   // --- Upiększanie linii cięć (Bonds) ---
   // Górna granica
   upperbondH->SetLineColor(kRed+1);
   upperbondH->SetLineWidth(3);
   upperbondH->SetLineStyle(1); // Ciągła
   upperbondH->Draw("same");

   // Dolna granica
   lowerbondH->SetLineColor(kRed+1);
   lowerbondH->SetLineWidth(3);
   lowerbondH->SetLineStyle(1); // Przerywana dla odróżnienia
   lowerbondH->Draw("same");

   // --- Dodanie eleganckiego napisu/tytułu ---
   TLatex title;
   title.SetNDC();
   title.SetTextFont(42);
   title.SetTextSize(0.05);
   title.DrawLatex(0.32, 0.93, "HCal Response to Pions");
   c1.SaveAs("Plots/Presentation/HCalPion.png");

   c1.Clear();

   // --- Wykres główny (Histogram 2D) ---
   HCalEnergyvsMomHist[0]->SetTitle(""); // Tytuł lepiej dodać przez TLatex dla kontroli
   HCalEnergyvsMomHist[0]->GetXaxis()->SetTitle("Momentum [GeV]");
   HCalEnergyvsMomHist[0]->GetYaxis()->SetTitle("HCal Energy/Momentum");
   HCalEnergyvsMomHist[0]->GetXaxis()->SetTitleSize(0.05);
   HCalEnergyvsMomHist[0]->GetYaxis()->SetTitleSize(0.05);
   HCalEnergyvsMomHist[0]->GetXaxis()->SetLabelSize(0.04);
   HCalEnergyvsMomHist[0]->GetYaxis()->SetLabelSize(0.04);

   // Rysowanie z opcją COLZ (mapa ciepła z legendą kolorów)
   HCalEnergyvsMomHist[0]->Draw("colz"); 

   // --- Upiększanie linii cięć (Bonds) ---
   // Górna granica
   upperbondH->SetLineColor(kRed+1);
   upperbondH->SetLineWidth(2);
   upperbondH->SetLineStyle(1); // Ciągła
   upperbondH->Draw("same");

   // Dolna granica
   lowerbondH->SetLineColor(kRed+1);
   lowerbondH->SetLineWidth(2);
   lowerbondH->SetLineStyle(1); // Przerywana dla odróżnienia
   lowerbondH->Draw("same");

   // --- Dodanie eleganckiego napisu/tytułu ---
   title.SetNDC();
   title.SetTextFont(42);
   title.SetTextSize(0.05);
   title.DrawLatex(0.32, 0.93, "HCal Response to Muons");
   c1.SaveAs("Plots/Presentation/JPsiHCalMuon.png");
   c1.Clear();

   // --- Wykres ECal ---
   ECalEnergyvsMomHist[1]->SetTitle(""); 
   ECalEnergyvsMomHist[1]->GetXaxis()->SetTitle("Momentum [GeV]");
   ECalEnergyvsMomHist[1]->GetYaxis()->SetTitle("ECal Energy/Momentum");
   ECalEnergyvsMomHist[1]->GetXaxis()->SetTitleSize(0.05);
   ECalEnergyvsMomHist[1]->GetYaxis()->SetTitleSize(0.05);
   ECalEnergyvsMomHist[1]->GetXaxis()->SetLabelSize(0.04);
   ECalEnergyvsMomHist[1]->GetYaxis()->SetLabelSize(0.04);

   // Rysujemy jako mapę kolorów
   ECalEnergyvsMomHist[1]->Draw("COLZ"); 

   // --- Cięcia (Bonds) ---
   // Górna granica
   upperbondE->SetLineColor(kRed+1); // Ciemniejszy niebieski dla ECal (rozróżnienie od HCal)
   upperbondE->SetLineWidth(2);
   upperbondE->SetLineStyle(1);
   upperbondE->Draw("same");



   // --- Profesjonalny opis ---
   TLatex tex;
   tex.SetNDC();
   tex.SetTextFont(42);
   tex.SetTextSize(0.045);
   tex.DrawLatex(0.32, 0.93, "ECal Response to Pions");

   // --- Zapis ---
   c1.SaveAs("Plots/Presentation/EcalPions.png");
   c1.Clear();

      // --- Wykres ECal ---
   ECalEnergyvsMomHist[0]->SetTitle(""); 
   ECalEnergyvsMomHist[0]->GetXaxis()->SetTitle("Momentum [GeV]");
   ECalEnergyvsMomHist[0]->GetYaxis()->SetTitle("ECal Energy/Momentum");
   ECalEnergyvsMomHist[0]->GetXaxis()->SetTitleSize(0.05);
   ECalEnergyvsMomHist[0]->GetYaxis()->SetTitleSize(0.05);
   ECalEnergyvsMomHist[0]->GetXaxis()->SetLabelSize(0.04);
   ECalEnergyvsMomHist[0]->GetYaxis()->SetLabelSize(0.04);

   // Rysujemy jako mapę kolorów
   ECalEnergyvsMomHist[0]->Draw("COLZ"); 

   // --- Cięcia (Bonds) ---
   // Górna granica
   upperbondE->SetLineColor(kRed+1); // Ciemniejszy niebieski dla ECal (rozróżnienie od HCal)
   upperbondE->SetLineWidth(2);
   upperbondE->SetLineStyle(1);
   upperbondE->Draw("same");



   // --- Profesjonalny opis ---
   tex.SetNDC();
   tex.SetTextFont(42);
   tex.SetTextSize(0.045);
   tex.DrawLatex(0.32, 0.93, "ECal Response to Muons");

   // --- Zapis ---
   c1.SaveAs("Plots/Presentation/EcalMuons.png");


   c1.Clear();

   // --- Stylistyka ---
   XGBResponse[0]->Scale(1.0/XGBResponse[0]->Integral());
   XGBResponse[1]->Scale(1.0/XGBResponse[1]->Integral());

   XGBResponse[0]->SetLineColor(kBlue+1);
   XGBResponse[0]->SetLineWidth(3);
   XGBResponse[0]->SetFillColorAlpha(kBlue+1, 0.1); // Delikatne wypełnienie tła
   
   XGBResponse[1]->SetLineColor(kRed+1);
   XGBResponse[1]->SetLineWidth(3);
   XGBResponse[1]->SetFillColorAlpha(kRed+1, 0.1); // Delikatne wypełnienie sygnału

   // Ustawienie osi
   XGBResponse[0]->SetTitle(";XGBoost Response;Normalized Counts");
   XGBResponse[0]->GetXaxis()->SetTitleSize(0.05);
   XGBResponse[0]->GetYaxis()->SetTitleSize(0.05);

   XGBResponse[0]->Draw("HIST");
   XGBResponse[1]->Draw("HIST SAME");

   gPad->Update(); // <-- konieczne, żeby ROOT obliczył zakres osi

   // Linia od 0 do aktualnego maksimum osi Y na padzie
   TLine *line = new TLine(0.86, gPad->GetUymin(), 0.86, gPad->GetUymax());
   line->SetLineColor(kBlack);
   line->SetLineStyle(2); // Linia przerywana
   line->SetLineWidth(3);
   line->Draw("same");

   // --- DODANIE LEGENDY ---
   TLegend *leg = new TLegend(0.4, 0.75, 0.6, 0.88);
   leg->SetBorderSize(0);
   leg->SetFillStyle(0);
   leg->SetTextSize(0.04);
   leg->AddEntry(XGBResponse[0], "Muon (Signal)", "l");
   leg->AddEntry(XGBResponse[1], "Pion (Background)", "l");
   leg->AddEntry(line, "Cut at 0.86", "l");
   leg->Draw();

   // --- NAPIS TYTUŁOWY ---
   tex.SetNDC();
   tex.SetTextFont(42);
   tex.SetTextSize(0.05);
   tex.DrawLatex(0.12, 0.92, "#bf{XGBoost Output Distribution}");

   c1.SaveAs("Plots/Presentation/ResponseContinuous.png");

   // === Efficiency for Muon Candidates ===
   // =============================================
   // === PLOT 1: Efficiency (E/p + XGBoost) ===
   // =============================================
   c1.Clear();

   TEfficiency *pEff1 = new TEfficiency(*CutParticEnergy[0], *AllParticEnergy[0]);
   pEff1->SetTitle("; Momentum [GeV/c]; Efficiency");
   pEff1->SetLineColor(kRed+1);
   pEff1->SetMarkerStyle(20);
   pEff1->SetMarkerSize(0.8);
   pEff1->SetMarkerColor(kRed+1);
   pEff1->SetStatisticOption(TEfficiency::kBUniform);

   TEfficiency *pEff2 = new TEfficiency(*FoundParticEnergy[0], *AllParticEnergy[0]);
   pEff2->SetLineColor(kGreen+2);
   pEff2->SetMarkerStyle(22);
   pEff2->SetMarkerSize(0.8);
   pEff2->SetMarkerColor(kGreen+2);
   pEff2->SetStatisticOption(TEfficiency::kBUniform);

   pEff1->Draw("AP");
   gPad->Update();
   pEff1->GetPaintedGraph()->SetMinimum(0.9);
   pEff1->GetPaintedGraph()->SetMaximum(1.01);
   gPad->Update();

   pEff2->Draw("P SAME");
   gPad->Update();

   TLegend *legEff = new TLegend(0.5, 0.75, 0.83, 0.92);
   legEff->SetBorderSize(0);
   legEff->SetFillStyle(0);
   legEff->AddEntry(pEff1, "Previous efficiency", "lp");
   legEff->AddEntry(pEff2, "XGBoost efficiency", "lp");
   legEff->Draw();

   tex.SetNDC();
   tex.DrawLatex(0.25, 0.94, "#bf{Muon Candidate Efficiency vs p}");
   c1.SaveAs("Plots/Efficiency/EfficiencyBoth.png");

   // =============================================
   // === PLOT 2: Rejection (E/p + XGBoost) ===
   // =============================================
   c1.Clear();

   TEfficiency *pEff0 = new TEfficiency(*CutParticEnergy[1], *AllParticEnergy[1]);
   pEff0->SetTitle("; Momentum [GeV/c]; Efficiency");
   pEff0->SetLineColor(kRed+1);
   pEff0->SetMarkerStyle(20);
   pEff0->SetMarkerSize(0.8);
   pEff0->SetMarkerColor(kRed+1);
   pEff0->SetStatisticOption(TEfficiency::kBUniform);

   TEfficiency *pEff3 = new TEfficiency(*FoundParticEnergy[1], *AllParticEnergy[1]);
   pEff3->SetLineColor(kGreen+2);
   pEff3->SetMarkerStyle(22);
   pEff3->SetMarkerSize(0.8);
   pEff3->SetMarkerColor(kGreen+2);
   pEff3->SetStatisticOption(TEfficiency::kBUniform);

   pEff0->Draw("AP");
   gPad->Update();
   pEff0->GetPaintedGraph()->SetMinimum(0.4);
   pEff0->GetPaintedGraph()->SetMaximum(1.05);
   gPad->Update();

   pEff3->Draw("P SAME");
   gPad->Update();

   TLegend *legRej = new TLegend(0.5, 0.15, 0.83, 0.42);
   legRej->SetBorderSize(0);
   legRej->SetFillStyle(0);
   legRej->AddEntry(pEff0, "Previous rejection", "lp");
   legRej->AddEntry(pEff3, "XGBoost rejection", "lp");
   legRej->Draw();

   tex.SetNDC();
   tex.DrawLatex(0.4, 0.92, "#bf{Pion Rejection vs p}");
   c1.SaveAs("Plots/Rejection//RejectionBoth.png");

   c1.Clear();

   // =============================================
   // === PLOT: Eta Efficiency (E/p + XGBoost) ===
   // =============================================
   c1.Clear();

   TEfficiency *pEffEta1 = new TEfficiency(*CutParticEta[0], *AllParticEta[0]);
   pEffEta1->SetTitle("; #eta; Efficiency");
   pEffEta1->SetLineColor(kRed+1);
   pEffEta1->SetMarkerStyle(20);
   pEffEta1->SetMarkerSize(0.8);
   pEffEta1->SetMarkerColor(kRed+1);
   pEffEta1->SetStatisticOption(TEfficiency::kBUniform);

   TEfficiency *pEffEta3 = new TEfficiency(*FoundParticEta[0], *AllParticEta[0]);
   pEffEta3->SetLineColor(kGreen+2);
   pEffEta3->SetMarkerStyle(22);
   pEffEta3->SetMarkerSize(0.8);
   pEffEta3->SetMarkerColor(kGreen+2);
   pEffEta3->SetStatisticOption(TEfficiency::kBUniform);

   pEffEta1->Draw("AP");
   gPad->Update();
   pEffEta1->GetPaintedGraph()->SetMinimum(0.88);
   pEffEta1->GetPaintedGraph()->SetMaximum(1.02);
   gPad->Update();

   pEffEta3->Draw("P SAME");
   gPad->Update();

   TLegend *legEffEta = new TLegend(0.5, 0.75, 0.83, 0.92);
   legEffEta->SetBorderSize(0);
   legEffEta->SetFillStyle(0);
   legEffEta->AddEntry(pEffEta1, "E/p cut efficiency", "lp");
   legEffEta->AddEntry(pEffEta3, "XGBoost efficiency", "lp");
   legEffEta->Draw();

   tex.SetNDC();
   tex.DrawLatex(0.25, 0.94, "#bf{Muon Candidate Efficiency vs #eta}");
   c1.SaveAs("Plots/Efficiency/EfficiencyBoth_Eta.png");

   // =============================================
   // === PLOT: Eta Rejection (E/p + XGBoost) ===
   // =============================================
   c1.Clear();

   TEfficiency *pEffEta0 = new TEfficiency(*CutParticEta[1], *AllParticEta[1]);
   pEffEta0->SetTitle("; #eta; Efficiency");
   pEffEta0->SetLineColor(kRed+1);
   pEffEta0->SetMarkerStyle(20);
   pEffEta0->SetMarkerSize(0.8);
   pEffEta0->SetMarkerColor(kRed+1);
   pEffEta0->SetStatisticOption(TEfficiency::kBUniform);

   TEfficiency *pEffEta2 = new TEfficiency(*FoundParticEta[1], *AllParticEta[1]);
   pEffEta2->SetLineColor(kGreen+2);
   pEffEta2->SetMarkerStyle(22);
   pEffEta2->SetMarkerSize(0.8);
   pEffEta2->SetMarkerColor(kGreen+2);
   pEffEta2->SetStatisticOption(TEfficiency::kBUniform);

   pEffEta0->Draw("AP");
   gPad->Update();
   pEffEta0->GetPaintedGraph()->SetMinimum(0.7);
   pEffEta0->GetPaintedGraph()->SetMaximum(1.02);
   gPad->Update();

   pEffEta2->Draw("P SAME");
   gPad->Update();

   TLegend *legRejEta = new TLegend(0.15, 0.75, 0.48, 0.92);
   legRejEta->SetBorderSize(0);
   legRejEta->SetFillStyle(0);
   legRejEta->AddEntry(pEffEta0, "E/p cut rejection", "lp");
   legRejEta->AddEntry(pEffEta2, "XGBoost rejection", "lp");
   legRejEta->Draw();

   tex.SetNDC();
   tex.DrawLatex(0.4, 0.94, "#bf{Pion Rejection vs #eta}");
   c1.SaveAs("Plots/Rejection/RejectionBoth_Eta.png");


   // =============================================
   // === PLOT: Pt Efficiency (E/p + XGBoost) ===
   // =============================================
   c1.Clear();

   TEfficiency *pEffPt1 = new TEfficiency(*CutParticPt[0], *AllParticPt[0]);
   pEffPt1->SetTitle("; p_{T} [GeV/c]; Efficiency");
   pEffPt1->SetLineColor(kRed+1);
   pEffPt1->SetMarkerStyle(20);
   pEffPt1->SetMarkerSize(0.8);
   pEffPt1->SetMarkerColor(kRed+1);
   pEffPt1->SetStatisticOption(TEfficiency::kBUniform);

   TEfficiency *pEffPt2 = new TEfficiency(*FoundParticPt[0], *AllParticPt[0]);
   pEffPt2->SetLineColor(kGreen+2);
   pEffPt2->SetMarkerStyle(22);
   pEffPt2->SetMarkerSize(0.8);
   pEffPt2->SetMarkerColor(kGreen+2);
   pEffPt2->SetStatisticOption(TEfficiency::kBUniform);

   pEffPt1->Draw("AP");
   gPad->Update();
   pEffPt1->GetPaintedGraph()->SetMinimum(0.94);
   pEffPt1->GetPaintedGraph()->SetMaximum(1.02);
   gPad->Update();

   pEffPt2->Draw("P SAME");
   gPad->Update();

   TLegend *legEffPt = new TLegend(0.5, 0.75, 0.83, 0.92);
   legEffPt->SetBorderSize(0);
   legEffPt->SetFillStyle(0);
   legEffPt->AddEntry(pEffPt1, "E/p cut efficiency", "lp");
   legEffPt->AddEntry(pEffPt2, "XGBoost efficiency", "lp");
   legEffPt->Draw();

   tex.SetNDC();
   tex.DrawLatex(0.25, 0.94, "#bf{Muon Candidate Efficiency vs p_{T}}");
   c1.SaveAs("Plots/Efficiency/EfficiencyBoth_Pt.png");

   // =============================================
   // === PLOT: Pt Rejection (E/p + XGBoost) ===
   // =============================================
   c1.Clear();

   TEfficiency *pEffPt0 = new TEfficiency(*CutParticPt[1], *AllParticPt[1]);
   pEffPt0->SetTitle("; p_{T} [GeV/c]; Efficiency");
   pEffPt0->SetLineColor(kRed+1);
   pEffPt0->SetMarkerStyle(20);
   pEffPt0->SetMarkerSize(0.8);
   pEffPt0->SetMarkerColor(kRed+1);
   pEffPt0->SetStatisticOption(TEfficiency::kBUniform);

   TEfficiency *pEffPt3 = new TEfficiency(*FoundParticPt[1], *AllParticPt[1]);
   pEffPt3->SetLineColor(kGreen+2);
   pEffPt3->SetMarkerStyle(22);
   pEffPt3->SetMarkerSize(0.8);
   pEffPt3->SetMarkerColor(kGreen+2);
   pEffPt3->SetStatisticOption(TEfficiency::kBUniform);

   pEffPt0->Draw("AP");
   gPad->Update();
   pEffPt0->GetPaintedGraph()->SetMinimum(0.47);
   pEffPt0->GetPaintedGraph()->SetMaximum(1.02);
   gPad->Update();

   pEffPt3->Draw("P SAME");
   gPad->Update();

   TLegend *legRejPt = new TLegend(0.5, 0.15, 0.83, 0.42);
   legRejPt->SetBorderSize(0);
   legRejPt->SetFillStyle(0);
   legRejPt->AddEntry(pEffPt0, "E/p cut rejection", "lp");
   legRejPt->AddEntry(pEffPt3, "XGBoost rejection", "lp");
   legRejPt->Draw();

   tex.SetNDC();
   tex.DrawLatex(0.4, 0.94, "#bf{Pion Rejection vs p_{T}}");
   c1.SaveAs("Plots/Rejection//RejectionBoth_Pt.png");

   // =========================================================================
   // 1. ZINTEGROWANY WYKRES DLA PT (XGBoost Efficiency + Rejection)
   // =========================================================================
   c1.Clear();

   // Tworzymy główny pad dla Efficiency (lewa oś Y)
   TPad *padPt1 = new TPad("padPt1", "", 0, 0, 1, 1);
   padPt1->SetGrid();
   padPt1->Draw();
   padPt1->cd();

   // Pomiary XGBoost dla mionów (Efficiency) -> indeks [0]
   TEfficiency *pXGB_EffPt = new TEfficiency(*FoundParticPt[0], *AllParticPt[0]);
   pXGB_EffPt->SetLineColor(kGreen+2);
   pXGB_EffPt->SetMarkerStyle(22);
   pXGB_EffPt->SetMarkerSize(1.0);
   pXGB_EffPt->SetMarkerColor(kGreen+2);
   pXGB_EffPt->SetStatisticOption(TEfficiency::kBUniform);
   
   pXGB_EffPt->Draw("AP");
   gPad->Update();
   
   // Formatowanie lewej osi Y (Efficiency)
   TGraphAsymmErrors *gPt1 = pXGB_EffPt->GetPaintedGraph();
   gPt1->SetTitle(";p_{T} [GeV/c];");
   gPt1->GetYaxis()->SetTitleColor(kGreen+2);
   gPt1->GetYaxis()->SetLabelColor(kGreen+2);
   gPt1->SetMinimum(0.80); // Dopasowane pod wydajność mionów
   gPt1->SetMaximum(1.02);
   gPad->Update();

   // Tworzymy drugi, nakładający się transparentny pad dla Rejection (prawa oś Y)
   c1.cd();
   TPad *padPt2 = new TPad("padPt2", "", 0, 0, 1, 1);
   padPt2->SetFillStyle(4000); // Transparentny
   padPt2->SetFrameFillStyle(4000);
   padPt2->Draw();
   padPt2->cd();

   // Pomiary XGBoost dla pionów (Rejection) -> indeks [1]
   TEfficiency *pXGB_RejPt = new TEfficiency(*FoundParticPt[1], *AllParticPt[1]);
   pXGB_RejPt->SetLineColor(kRed+1);
   pXGB_RejPt->SetMarkerStyle(23);
   pXGB_RejPt->SetMarkerSize(1.0);
   pXGB_RejPt->SetMarkerColor(kRed+1);
   pXGB_RejPt->SetStatisticOption(TEfficiency::kBUniform);
   
   // Rysujemy z opcją "Y+", aby oś znalazła się po prawej stronie
   pXGB_RejPt->Draw("APY+"); 
   gPad->Update();

   // Formatowanie prawej osi Y (Rejection)
   TGraphAsymmErrors *gPt2 = pXGB_RejPt->GetPaintedGraph();
   gPt2->SetTitle(";p_{T} [GeV/c];"); // Czyścimy tytuł X, żeby się nie nałożył
   gPt2->GetYaxis()->SetTitleColor(kRed+1);
   gPt2->GetYaxis()->SetLabelColor(kRed+1);
   gPt2->SetMinimum(0.80); // Dopasowane pod szerokie odrzucenie pionów
   gPt2->SetMaximum(1.02);
   gPad->Update();

   // Wspólna legenda
   TLegend *legPt = new TLegend(0.37, 0.20, 0.65, 0.35);
   legPt->SetBorderSize(1);
   legPt->AddEntry(pXGB_EffPt, "Muon Efficiency", "lp");
   legPt->AddEntry(pXGB_RejPt, "Pion Rejection", "lp");
   legPt->Draw();

   c1.cd();
   tex.SetNDC();
   tex.DrawLatex(0.22, 0.94, "#bf{XGBoost Performance vs p_{T}}");
   c1.SaveAs("Plots/XGBoost_Performance_Pt.png");


   // =========================================================================
   // 2. ZINTEGROWANY WYKRES DLA ETA (XGBoost Efficiency + Rejection)
   // =========================================================================
   c1.Clear();

   double etaMin = FoundParticEta[0]->GetXaxis()->GetXmin();
   double etaMax = FoundParticEta[0]->GetXaxis()->GetXmax();

   // Główny pad dla Eta Efficiency
   TPad *padEta1 = new TPad("padEta1", "", 0, 0, 1, 1);
   padEta1->SetGrid();
   padEta1->Draw();
   padEta1->cd();

   TEfficiency *pXGB_EffEta = new TEfficiency(*FoundParticEta[0], *AllParticEta[0]);
   pXGB_EffEta->SetLineColor(kGreen+2);
   pXGB_EffEta->SetMarkerStyle(22);
   pXGB_EffEta->SetMarkerSize(1.0);
   pXGB_EffEta->SetMarkerColor(kGreen+2);
   pXGB_EffEta->SetStatisticOption(TEfficiency::kBUniform);
   
   pXGB_EffEta->Draw("AP");
   gPad->Update();
   
   TGraphAsymmErrors *gEta1 = pXGB_EffEta->GetPaintedGraph();
   gEta1->SetTitle(";Pseudorapidity #eta;");
   gEta1->GetYaxis()->SetTitleColor(kGreen+2);
   gEta1->GetYaxis()->SetLabelColor(kGreen+2);
   gEta1->GetXaxis()->SetLimits(etaMin, etaMax);
   gEta1->SetMinimum(0.86);
   gEta1->SetMaximum(1.02);
   gPad->Update();

   // Transparentny pad dla Eta Rejection
   c1.cd();
   TPad *padEta2 = new TPad("padEta2", "", 0, 0, 1, 1);
   padEta2->SetFillStyle(4000);
   padEta2->SetFrameFillStyle(4000);
   padEta2->Draw();
   padEta2->cd();

   TEfficiency *pXGB_RejEta = new TEfficiency(*FoundParticEta[1], *AllParticEta[1]);
   pXGB_RejEta->SetLineColor(kRed+1);
   pXGB_RejEta->SetMarkerStyle(23);
   pXGB_RejEta->SetMarkerSize(1.0);
   pXGB_RejEta->SetMarkerColor(kRed+1);
   pXGB_RejEta->SetStatisticOption(TEfficiency::kBUniform);
   
   pXGB_RejEta->Draw("APY+");
   gPad->Update();

   TGraphAsymmErrors *gEta2 = pXGB_RejEta->GetPaintedGraph();
   gEta2->SetTitle(";Pseudorapidity #eta;");
   gEta2->GetYaxis()->SetTitleColor(kRed+1);
   gEta2->GetYaxis()->SetLabelColor(kRed+1);
   gEta2->GetXaxis()->SetLimits(etaMin, etaMax);
   gEta2->SetMinimum(0.86);
   gEta2->SetMaximum(1.02);
   gPad->Update();

   // Wspólna legenda (przesunięta na środek-dół)
   TLegend *legEta = new TLegend(0.37, 0.20, 0.65, 0.35);
   legEta->SetBorderSize(1);
   legEta->AddEntry(pXGB_EffEta, "Muon Efficiency", "lp");
   legEta->AddEntry(pXGB_RejEta, "Pion Rejection", "lp");
   legEta->Draw();

   c1.cd();
   tex.SetNDC();
   tex.DrawLatex(0.25, 0.94, "#bf{XGBoost Performance vs #eta}");
   c1.SaveAs("Plots/XGBoost_Performance_Eta.png");
}