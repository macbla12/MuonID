
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

#include "ToFFastSim.cxx"
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


float MuonID(TString file)
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

   
   

    // Set up input file chain
    TChain *mychain = new TChain("events");
    mychain->Add(file);

    // Initialize reader
    TTreeReader tree_reader(mychain);
    Long64_t nEvents = mychain->GetEntries();

    TF1 *upperbondE = new TF1("upperbondE", "2/(x**2)+0.05", 0.001, 24.0);
    upperbondE->SetLineColor(kRed);
    upperbondE->SetLineWidth(1);

    TF1 *upperbondH = new TF1("upperbondH", "3.5/x",  0.001, 24.0); 
    upperbondH->SetLineColor(kRed);
    upperbondH->SetLineWidth(1);
        
    TF1 *lowerbondH = new TF1("lowerbondH", "0.3/x-0.25/(x*x)",  0.001, 24.0); 
    lowerbondH->SetLineColor(kRed);
    lowerbondH->SetLineWidth(1);

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

    
    int eventID=startEvent;
    double FoundParticles=0;
    double particscount=0;
    double BadPDG=0;
    double aftercuts=0,secondcuts=0;
    double CaloHit=0;

    

    while(tree_reader.Next()){
        eventID++;
        

        int id=0;
        for(int particle=0; particle<trackEng.GetSize();particle++)
        {
        double ECalEnergy=0, HCalEnergy=0, ECalNumber=0, HCalNumber=0;
        std::vector<float> EcalShape, HcalShape;
        particscount++;
        //Obligatory Cuts 
        CaloHit=0;

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



        if(CaloHit==0){

            if(HCalEoverP<upperbondH->Eval(Momentum) && HCalEoverP>lowerbondH->Eval(Momentum) && ECalEoverP<upperbondE->Eval(Momentum)) return 1.; 
            
            else return 0.;

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
        return muon_prob;
        }

    

    }
}
}
   
