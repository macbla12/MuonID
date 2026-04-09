
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
//#include "Calorimeters.cxx"
#include "Calorimeternew.cxx"

std::vector<float> scaler_mean = {0.23376697f, 1.14607939f, 0.64005278f, 1.40825878f, 0.08338190f, 0.48935787f, 3.18928276f, 0.29481987f, 1.37984636f, 0.21024462f, 0.95256580f, 0.86218541f, 0.22057588f, 12.63038272f, 35813642315367410592382976.00000000f, 1156.13450482f, 56668.72235209f, 1197637.96025355f, 23609367.20731101f, 653057816.93298042f, -0.42320575f, -0.74713854f, 0.86925998f, 35.18557764f, 26.94725791f, 0.00020703f, 0.03042352f, 818.68957700f, 443.68193555f, 254.18972951f, 210.56265916f, 196.27291519f, 0.00009702f, 0.02374836f, 47627.06920600f, 9263.48542525f, 1769.96602489f};


std::vector<float> scaler_scale = {0.35722993f, 0.62739702f, 0.63137067f, 0.81716083f, 0.12703714f, 0.29750231f, 5.79220399f, 0.77528269f, 0.76175045f, 0.29915046f, 0.57032156f, 0.17956640f, 0.51880440f, 2432.37821684f, 14830913161250860793749045248.00000000f, 6050.75083496f, 62306.57658586f, 255097553.26807174f, 8599975792.08747101f, 289874773997.07965088f, 0.73809161f, 1.53611361f, 0.13868696f, 32.20092665f, 30.33968031f, 0.00373327f, 0.45552891f, 4183.28349218f, 4425.07238366f, 4976.43148373f, 142.47355585f, 141.90652549f, 0.00035247f, 0.43343317f, 62399.03848260f, 29080.29731516f, 9686.19217973f};

float safe_divide(float num, float denom) {
    return (denom != 0) ? (num / denom) : 0.0f;
}
std::vector<float> prepare_37_features(double eE, double hE, double eN, double hN, 
                                      double eP, double hP, double mom,
                                      const std::vector<float>& eS, 
                                      const std::vector<float>& hS) {
    std::vector<float> X;
    // Jeśli wektory shape są puste (brak klastra), wypełnij zerami (7 elementów)
    std::vector<float> eS_safe = eS.size() == 7 ? eS : std::vector<float>(7, 0.0f);
    std::vector<float> hS_safe = hS.size() == 7 ? hS : std::vector<float>(7, 0.0f);

    // --- 1. Scalar Features (12) ---
    float totalE = eE + hE;
    X.push_back(eE); X.push_back(hE); X.push_back(eN); X.push_back(hN);
    X.push_back(eP); X.push_back(hP); X.push_back(mom);
    X.push_back(safe_divide(eE, hE));      // ECal_HCal_ratio
    X.push_back(totalE);                   // TotalCalEnergy
    X.push_back(safe_divide(eE, eN));      // ECalDensity
    X.push_back(safe_divide(hE, hN));      // HCalDensity
    X.push_back(safe_divide(hE, totalE));  // HCalFraction

    // --- 2. Shape Derived (11) ---
    float eTrans = std::sqrt(std::pow(eS_safe[4], 2) + std::pow(eS_safe[5], 2)); // x_width, y_width
    float hTrans = std::sqrt(std::pow(hS_safe[4], 2) + std::pow(hS_safe[5], 2));
    
    X.push_back(safe_divide(eS_safe[0], hS_safe[0])); // radius_ratio
    X.push_back(safe_divide(eS_safe[1], hS_safe[1])); // disp_ratio
    X.push_back(safe_divide(eS_safe[6], hS_safe[6])); // z_width_ratio
    X.push_back(eTrans);                             // Ecal_transverse
    X.push_back(hTrans);                             // Hcal_transverse
    X.push_back(safe_divide(eTrans, hTrans));        // transverse_ratio
    X.push_back(safe_divide(eS_safe[6], eTrans));    // Ecal_long_trans_ratio
    X.push_back(safe_divide(hS_safe[6], hTrans));    // Hcal_long_trans_ratio
    X.push_back(safe_divide(eS_safe[2]-eS_safe[3], eS_safe[2]+eS_safe[3])); // Ecal_angular_asym
    X.push_back(safe_divide(hS_safe[2]-hS_safe[3], hS_safe[2]+hS_safe[3])); // Hcal_angular_asym
    X.push_back(safe_divide(hS_safe[0], eS_safe[0] + hS_safe[0]));          // Radial_HCal_Fraction

    // --- 3. Raw Shapes (14) ---
    for(float val : eS_safe) X.push_back(val);
    for(float val : hS_safe) X.push_back(val);

    // --- 4. Standaryzacja ---
    for(size_t i=0; i<X.size(); ++i) {
        X[i] = (X[i] - scaler_mean[i]) / scaler_scale[i];
    }
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
    //gStyle->SetOptStat(0);

        double DEG=180/TMath::Pi();

    // --- INICJALIZACJA ONNX (Dodaj tutaj) ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MuonID");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "Plots/xgb_muonID.onnx", session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Nazwy wejść/wyjść (zależą od konwertera, zazwyczaj "input" i "probabilities")
    const char* input_names[] = {"input"};
    const char* output_names[] = {"probabilities"};


    //////////////////////
    //Setting up histograms
    //////////////////////
    static constexpr int NumOfFiles=2;
   
    TH1D *AllParticEta[NumOfFiles], *AllParticPhi[NumOfFiles], *AllParticEnergy[NumOfFiles];
    TH1D *CutParticEta[NumOfFiles], *CutParticPhi[NumOfFiles], *CutParticEnergy[NumOfFiles];
    TH1D *FoundParticEta[NumOfFiles], *FoundParticPhi[NumOfFiles], *FoundParticEnergy[NumOfFiles];
    TH1D *ECalEnergyHist[NumOfFiles], *ECalEnergyMomHist[NumOfFiles],*HCalEnergyHist[NumOfFiles], *HCalEnergyMomHist[NumOfFiles];
    TH2D *ECalEnergyvsMomHist[NumOfFiles],*HCalEnergyvsMomHist[NumOfFiles];
    TH2D *ECalEnergyMomvsEtaHist[NumOfFiles],  *HCalEnergyMomvsEtaHist[NumOfFiles];
    TH1D *XGBResponse[NumOfFiles];
 
    
    vector<TString> files(NumOfFiles);


   files.at(0)="/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
   //files.at(0)="/run/media/epic/Data/Muons/Grape-10x275/Paper/RECO/*.root";
   //files.at(0)="/run/media/epic/Data/Background/JPsi/OLD/*.root";
   //files.at(0)="/run/media/epic/Data/Background/JPsi/March/*.root";


   files.at(1)="/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
   //files.at(1)="/run/media/epic/Data/Tau/reco/Energy_10x275/double_pi/recoDoublePi.root";



   TF1 *upperbondE = new TF1("upperbondE", "2/x+0.05", 0.001, 24.0);
   upperbondE->SetLineColor(kRed);
   upperbondE->SetLineWidth(1);

   TF1 *upperbondH = new TF1("upperbondH", "3.5/x+0.1",  0.001, 24.0); 
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
      TTreeReaderArray< int> simuAssoc(tree_reader, "_ReconstructedChargedParticleAssociations_sim.index");

      // Get B0 Information
      TTreeReaderArray< int> simuAssocB0(tree_reader, "_B0ECalClusterAssociations_sim.index");
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
      TTreeReaderArray< int> simuAssocEcalBarrel(tree_reader, "_EcalEndcapPClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalBarrelEng(tree_reader, "EcalBarrelClusters.energy");
      TTreeReaderArray<unsigned int> EcalBarrelShPB(tree_reader, "EcalBarrelClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalBarrelShPE(tree_reader, "EcalBarrelClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalBarrelShParameters(tree_reader, "_EcalBarrelClusters_shapeParameters");


      TTreeReaderArray< int> simuAssocEcalBarrelImaging(tree_reader, "_EcalBarrelImagingClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalBarrelImagingEng(tree_reader, "EcalBarrelImagingClusters.energy");
      TTreeReaderArray<unsigned int> EcalBarrelImagingShPB(tree_reader, "EcalBarrelImagingClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalBarrelImagingShPE(tree_reader, "EcalBarrelImagingClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalBarrelImagingShParameters(tree_reader, "_EcalBarrelImagingClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocEcalBarrelScFi(tree_reader, "_EcalBarrelScFiClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalBarrelScFiEng(tree_reader, "EcalBarrelScFiClusters.energy");
      TTreeReaderArray<unsigned int> EcalBarrelScFiShPB(tree_reader, "EcalBarrelScFiClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalBarrelScFiShPE(tree_reader, "EcalBarrelScFiClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalBarrelScFiShParameters(tree_reader, "_EcalBarrelScFiClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocEcalEndcapP(tree_reader, "_EcalEndcapPClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalEndcapPEng(tree_reader, "EcalEndcapPClusters.energy");
      TTreeReaderArray<unsigned int> EcalEndcapPShPB(tree_reader, "EcalEndcapPClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalEndcapPShPE(tree_reader, "EcalEndcapPClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalEndcapPShParameters(tree_reader, "_EcalEndcapPClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocEcalEndcapN(tree_reader, "_EcalEndcapNClusterAssociations_sim.index");
      TTreeReaderArray<float> EcalEndcapNEng(tree_reader, "EcalEndcapNClusters.energy");
      TTreeReaderArray<unsigned int> EcalEndcapNShPB(tree_reader, "EcalEndcapNClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> EcalEndcapNShPE(tree_reader, "EcalEndcapNClusters.shapeParameters_end");
      TTreeReaderArray<float> EcalEndcapNShParameters(tree_reader, "_EcalEndcapNClusters_shapeParameters");

      // Hcal Information
      TTreeReaderArray<int> simuAssocHcalBarrel(tree_reader, "_HcalBarrelClusterAssociations_sim.index");
      TTreeReaderArray<float> HcalBarrelEng(tree_reader, "HcalBarrelClusters.energy");
      TTreeReaderArray<unsigned int> HcalBarrelShPB(tree_reader, "HcalBarrelClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> HcalBarrelShPE(tree_reader, "HcalBarrelClusters.shapeParameters_end");
      TTreeReaderArray<float> HcalBarrelShParameters(tree_reader, "_HcalBarrelClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocHcalEndcapP(tree_reader, "_HcalEndcapPInsertClusterAssociations_sim.index");
      TTreeReaderArray<float> HcalEndcapPEng(tree_reader, "HcalEndcapPInsertClusters.energy");
      TTreeReaderArray<unsigned int> HcalEndcapPShPB(tree_reader, "HcalEndcapPInsertClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> HcalEndcapPShPE(tree_reader, "HcalEndcapPInsertClusters.shapeParameters_end");
      TTreeReaderArray<float> HcalEndcapPShParameters(tree_reader, "_HcalEndcapPInsertClusters_shapeParameters");

      TTreeReaderArray<int> simuAssocLFHcal(tree_reader, "_LFHCALClusterAssociations_sim.index");
      TTreeReaderArray<float> LFHcalEng(tree_reader, "LFHCALClusters.energy");
      TTreeReaderArray<unsigned int> LFHcalShPB(tree_reader, "LFHCALClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> LFHcalShPE(tree_reader, "LFHCALClusters.shapeParameters_end");
      TTreeReaderArray<float> LFHcalShParameters(tree_reader, "_LFHCALClusters_shapeParameters");

      TTreeReaderArray< int> simuAssocHcalEndcapN(tree_reader, "_HcalEndcapNClusterAssociations_sim.index");
      TTreeReaderArray<float> HcalEndcapNEng(tree_reader, "HcalEndcapNClusters.energy");
      TTreeReaderArray<unsigned int> HcalEndcapNShPB(tree_reader, "HcalEndcapNClusters.shapeParameters_begin");
      TTreeReaderArray<unsigned int> HcalEndcapNShPE(tree_reader, "HcalEndcapNClusters.shapeParameters_end");
      TTreeReaderArray<float> HcalEndcapNShParameters(tree_reader, "_HcalEndcapNClusters_shapeParameters");

     

      //==================================//

      AllParticEta[File] = new TH1D(Form("AllParticEta%s",name.c_str()),Form("AllParticEta%s",name.c_str()),50,-4,4);
      AllParticPhi[File]= new TH1D(Form("AllParticPhi%s",name.c_str()),Form("AllParticPhi%s",name.c_str()),30,-180,180);
      AllParticEnergy[File]= new TH1D(Form("AllParticEnergy%s",name.c_str()),Form("AllParticEnergy%s",name.c_str()),40,0,20);
      
      CutParticEta[File] = new TH1D(Form("CutParticEta%s",name.c_str()),Form("CutParticEta%s",name.c_str()),50,-4,4);
      CutParticPhi[File]= new TH1D(Form("CutParticPhi%s",name.c_str()),Form("CutParticPhi%s",name.c_str()),30,-180,180);
      CutParticEnergy[File]= new TH1D(Form("CutParticEnergy%s",name.c_str()),Form("CutParticEnergy%s",name.c_str()),40,0,20);

      FoundParticEta[File] = new TH1D(Form("FoundParticEta%s",name.c_str()),Form("FoundParticEta%s",name.c_str()),50,-4,4);
      FoundParticPhi[File]= new TH1D(Form("FoundParticPhi%s",name.c_str()),Form("FoundParticPhi%s",name.c_str()),30,-180,180);
      FoundParticEnergy[File]= new TH1D(Form("FoundParticEnergy%s",name.c_str()),Form("FoundParticEnergy%s",name.c_str()),40,0,20);
      
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

      int eventID=0;
      double FoundParticles=0;
      double particscount=0;
      double BadPDG=0;
      double aftercuts=0,secondcuts=0;

      while(tree_reader.Next()){
         eventID++;
         //if(eventID>30000) break;
         
         //if(File==0) if(eventID>3800) break;

         

         //if(eventID%20000==0) cout<<"File "<<name<<" and event number... "<<eventID<<endl;

         

         int id=0;
         for(int particle=0; particle<trackEng.GetSize();particle++)
         {
            double ECalEnergy=0, HCalEnergy=0, ECalNumber=0, HCalNumber=0;
            std::vector<float> EcalShape, HcalShape;
            particscount++;
            //Obligatory Cuts 
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
            if(Partic.E()<1) continue;

         
            AllParticEnergy[File]->Fill(Partic.E());
            AllParticEta[File]->Fill(Partic.Eta());
            AllParticPhi[File]->Fill(Partic.Phi()*DEG);
           
           //Ecal Energy Search
            int simuID = simuAssoc[particle];

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
               Found=1;
            }
            

            

            FoundParticles+=Found;
            
            //Track properties 
            double FullEnergy=HCalEnergy+ECalEnergy;
            if(FullEnergy==0) continue;

            double Momentum=Partic.P();
            double HCalEoverP=HCalEnergy/Momentum;
            double ECalEoverP=ECalEnergy/Momentum;
            
            ECalEnergyMomvsEtaHist[File]->Fill(Partic.Eta(),ECalEoverP);
            HCalEnergyMomvsEtaHist[File]->Fill(Partic.Eta(),HCalEoverP);

            ECalEnergyvsMomHist[File]->Fill(Momentum,ECalEoverP);
            HCalEnergyvsMomHist[File]->Fill(Momentum,HCalEoverP);
         
                

            if(!(trackPDG[particle]==0 || abs(trackPDG[particle])==13)) continue;
            if(HCalEoverP>upperbondH->Eval(Momentum)) continue;
            if(HCalEoverP<lowerbondH->Eval(Momentum)) continue;
            if(ECalEoverP>upperbondE->Eval(Momentum)) continue;

            CutParticEta[File] ->Fill(Partic.Eta());
            CutParticPhi[File]->Fill(Partic.Phi()*DEG);
            CutParticEnergy[File]->Fill(Partic.E());


             std::vector<float> feats = prepare_37_features(
                    ECalEnergy, HCalEnergy, ECalNumber, HCalNumber,
                    ECalEoverP, HCalEoverP, Momentum, EcalShape, HcalShape
                     );

               // 2. Stwórz tensor wejściowy
               int64_t input_shape[] = {1, 37};
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
            aftercuts++;

            if(muon_prob<0.26) continue;

            secondcuts++;
            FoundParticEta[File] ->Fill(Partic.Eta());
            FoundParticPhi[File]->Fill(Partic.Phi()*DEG);
            FoundParticEnergy[File]->Fill(Partic.E());
            



            

         } 

      }
      

      cout<<"==========================="<<endl;
      cout<<"End of "<< name << " file"<<endl;
      cout<<"Number of events: "<<eventID<<endl;
      cout<<"Found particles: "<<FoundParticles<<"   All particles: "<<particscount<<endl;
      cout<<"Found Ratio: "<<FoundParticles*100/particscount<<'%'<<endl;
      cout<<"After First Cuts Ratio: "<<aftercuts*100/FoundParticles<<'%'<<endl;
      cout<<"   After first cut particles: "<<aftercuts<<endl;

      cout<<"After Second Cuts Ratio: "<<secondcuts*100/aftercuts<<'%'<<endl;
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
   XGBResponse[0]->Scale(1/XGBResponse[0]->Integral());
   XGBResponse[1]->Scale(1/XGBResponse[1]->Integral());
   XGBResponse[0]->SetLineColor(kBlue);
   XGBResponse[1]->SetLineColor(kRed);
   XGBResponse[0]->Draw("HIST");
   XGBResponse[1]->Draw("HIST SAME");
   c1.SaveAs("Plots/FinalCalID.pdf");


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
   c1.SaveAs("Plots/FinalCalID.pdf");
   c1.Clear();

   c1.Divide(2,2);
   c1.cd(1);
      gPad->SetLogy(1);
      AllParticEta[1]->SetLineColor(kBlue);
      FoundParticEta[1]->SetLineColor(kRed);
      CutParticEta[1]->SetLineColor(kGreen+1);
      CutParticEta[1]->SetLineWidth(2);
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
   c1.SaveAs("Plots/HCalPion.png");

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
   c1.SaveAs("Plots/HCalMuon.png");
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
   c1.SaveAs("Plots/EcalPions.png");
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
   c1.SaveAs("Plots/EcalMuons.png");


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
   TLine *line = new TLine(0.26, gPad->GetUymin(), 0.26, gPad->GetUymax());
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
   leg->AddEntry(line, "Cut at 0.26", "l");
   leg->Draw();

   // --- NAPIS TYTUŁOWY ---
   tex.SetNDC();
   tex.SetTextFont(42);
   tex.SetTextSize(0.05);
   tex.DrawLatex(0.12, 0.92, "#bf{XGBoost Output Distribution}");

   c1.SaveAs("Plots/ResponseContinuous.png");

   // === Efficiency for Muon Candidates ===
c1.Clear();

TEfficiency *pEff1 = new TEfficiency(*CutParticEnergy[0], *AllParticEnergy[0]);
pEff1->SetTitle("; Energy [GeV];Efficiency");
pEff1->SetLineColor(kRed+1);
pEff1->SetMarkerStyle(20);
pEff1->SetMarkerSize(0.8);
pEff1->SetMarkerColor(kRed+1);

pEff1->Draw("AP");
gPad->Update();

pEff1->GetPaintedGraph()->SetMinimum(0.95);
pEff1->GetPaintedGraph()->SetMaximum(1.02);
gPad->Update();

tex.SetNDC();
tex.DrawLatex(0.2, 0.92, "#bf{E/p cut efficiency for Muon Candidates}");
c1.SaveAs("Plots/EfficiencyFirst.png");

// === Background Rejection for Pions ===
c1.Clear();

// Rejected = All - Passed
TH1F *hRejected0 = (TH1F*)AllParticEnergy[1]->Clone("hRejected0");
hRejected0->Add(CutParticEnergy[1], -1.0);

// Sprawdzenie spójności histogramów
if (!TEfficiency::CheckConsistency(*hRejected0, *AllParticEnergy[1])) {
    std::cerr << "ERROR: Inconsistent histograms for pion rejection!" << std::endl;
    return;
}

TEfficiency *pEff0 = new TEfficiency(*hRejected0, *AllParticEnergy[1]);
pEff0->SetTitle("; Energy [GeV];Rejection (1 - #varepsilon)");
pEff0->SetLineColor(kBlue+1);
pEff0->SetMarkerStyle(21);
pEff0->SetMarkerSize(0.8);
pEff0->SetMarkerColor(kBlue+1);
pEff0->SetStatisticOption(TEfficiency::kBUniform); // Bayesian, dobra dla wartości bliskich 0/1

pEff0->Draw("AP");
gPad->Update();

pEff0->GetPaintedGraph()->SetMinimum(0.85);
pEff0->GetPaintedGraph()->SetMaximum(1.02);
gPad->Update();

tex.SetNDC();
tex.DrawLatex(0.2, 0.92, "#bf{E/p cut efficiency of Pions Rejection}");
c1.SaveAs("Plots/RejectionFirst.png");

// Sprzątanie
delete hRejected0;


TEfficiency *pEff2 = new TEfficiency(*FoundParticEnergy[0], *CutParticEnergy[0]);
pEff2->SetTitle("; Energy [GeV];Efficiency");
pEff2->SetLineColor(kRed+1);
pEff2->SetMarkerStyle(20);
pEff2->SetMarkerSize(0.8);
pEff2->SetMarkerColor(kRed+1);

pEff2->Draw("AP");
gPad->Update();

pEff2->GetPaintedGraph()->SetMinimum(0.8);
pEff2->GetPaintedGraph()->SetMaximum(1.05);
gPad->Update();

tex.SetNDC();
tex.DrawLatex(0.2, 0.92, "#bf{XGBoost efficiency for Muon Candidates}");
c1.SaveAs("Plots/EfficiencySecond.png");

// === Background Rejection for Pions ===
c1.Clear();

// Rejected = All - Passed
TH1F *hRejected1 = (TH1F*)CutParticEnergy[1]->Clone("hRejected1");
hRejected1->Add(FoundParticEnergy[1], -1.0);

// Sprawdzenie spójności histogramów
if (!TEfficiency::CheckConsistency(*hRejected1, *CutParticEnergy[1])) {
    std::cerr << "ERROR: Inconsistent histograms for pion rejection!" << std::endl;
    return;
}

TEfficiency *pEff3 = new TEfficiency(*hRejected1, *CutParticEnergy[1]);
pEff3->SetTitle("; Energy [GeV];Rejection (1 - #varepsilon)");
pEff3->SetLineColor(kBlue+1);
pEff3->SetMarkerStyle(21);
pEff3->SetMarkerSize(0.8);
pEff3->SetMarkerColor(kBlue+1);
pEff3->SetStatisticOption(TEfficiency::kBUniform); // Bayesian, dobra dla wartości bliskich 0/1

pEff3->Draw("AP");
gPad->Update();

pEff3->GetPaintedGraph()->SetMinimum(0.5);
pEff3->GetPaintedGraph()->SetMaximum(1.05);
gPad->Update();

tex.SetNDC();
tex.DrawLatex(0.2, 0.92, "#bf{XGBoost efficiency of Pions Rejection}");
c1.SaveAs("Plots/RejectionSecond.png");

// Sprzątanie
delete hRejected1;




}

