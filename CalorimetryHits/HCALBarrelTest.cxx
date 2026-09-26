void HCALBarrelTest()
{
    static double MuonMass = 0.1056583;
    static double ElectronMass = 0.00051099895;
    static double PionMass = 0.13957039;

    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    // Approximate boundary of the "first layer" of the HCal Barrel in R [mm].
    // Note: the `layer` field in cellID is unfilled (-1) in this dataset,
    // so until cellID is decoded, an R cut is used as an approximation.
    const double R_FIRST_LAYER_CUT = 2260.0;

    static constexpr int NumOfFiles = 2;
    vector<TString> files(NumOfFiles);

    files.at(0) = "/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
    files.at(1) = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";

    for (int File = 0; File < NumOfFiles; File++)
    {
        string name = (File == 1) ? "Pions" : "Muons";

        TChain *mychain = new TChain("events");
        mychain->Add(files.at(File));

        TTreeReader tree_reader(mychain);
        Long64_t nEvents = mychain->GetEntries();
        cout << "Total events in chain (" << name << "): " << nEvents << endl;

        // MCParticles (kept, useful e.g. for a PDG cross-check later)
        TTreeReaderArray<int> partGenStat(tree_reader, "MCParticles.generatorStatus");
        TTreeReaderArray<double> partMomX(tree_reader, "MCParticles.momentum.x");
        TTreeReaderArray<double> partMomY(tree_reader, "MCParticles.momentum.y");
        TTreeReaderArray<double> partMomZ(tree_reader, "MCParticles.momentum.z");
        TTreeReaderArray<int> partPdg(tree_reader, "MCParticles.PDG");

        // Reconstructed Charged Particles (tracks)
        TTreeReaderArray<float> trackMomX(tree_reader, "ReconstructedChargedParticles.momentum.x");
        TTreeReaderArray<float> trackMomY(tree_reader, "ReconstructedChargedParticles.momentum.y");
        TTreeReaderArray<float> trackMomZ(tree_reader, "ReconstructedChargedParticles.momentum.z");
        TTreeReaderArray<float> trackEng(tree_reader, "ReconstructedChargedParticles.energy");
        TTreeReaderArray<int> simuAssoc(tree_reader, "_ReconstructedChargedParticleAssociations_sim.index");

        // Hcal Clusters & Associations
        TTreeReaderArray<float> hcalClusters(tree_reader, "HcalBarrelClusters.energy");
        TTreeReaderArray<int> simuAssocHcalBarrel(tree_reader, "_HcalBarrelClusterAssociations_sim.index");
        TTreeReaderArray<int> clusterHitAssoc(tree_reader, "_HcalBarrelClusters_hits.index");
        TTreeReaderArray<unsigned int> clusterHitBegin(tree_reader, "HcalBarrelClusters.hits_begin");
        TTreeReaderArray<unsigned int> clusterHitEnd(tree_reader, "HcalBarrelClusters.hits_end");

        // Hcal Rec Hits
        TTreeReaderArray<unsigned long> hitCellID(tree_reader, "HcalBarrelRecHits.cellID");
        TTreeReaderArray<float> hitEnergy(tree_reader, "HcalBarrelRecHits.energy");
        TTreeReaderArray<float> hitTime(tree_reader, "HcalBarrelRecHits.time");
        TTreeReaderArray<float> hitPosX(tree_reader, "HcalBarrelRecHits.position.x");
        TTreeReaderArray<float> hitPosY(tree_reader, "HcalBarrelRecHits.position.y");
        TTreeReaderArray<float> hitPosZ(tree_reader, "HcalBarrelRecHits.position.z");

        // Output ROOT file
        TFile *outfile = new TFile(Form("Plots/Hits/HitsBarrel_%s.root", name.c_str()), "RECREATE");

        // ---------------------------------------------------------------
        // Diagnostic histograms (raw hit-level view, same as before)
        // ---------------------------------------------------------------
        TH1F *HitR_hist   = new TH1F("HitR", "Hit R position;R [mm];Counts", 2000, 2240, 2310);
        TH1F *HitPhi_hist = new TH1F("HitPhi", "Hit Phi position;#phi;Counts", 200, -3.15, 3.15);
        TH1F *HitZ_hist   = new TH1F("HitZ", "Hit Z position;Z [mm];Counts", 200, -3000, 3000);
        TH1F *HitEnergy_hist = new TH1F("HitEnergy", "Hit Energy;Energy [GeV];Counts", 200, 0, 1);
        TH1F *HitTime_hist   = new TH1F("HitTime", "Hit Time;Time [ns];Counts", 200, 0, 200);

        TH1F *DeltaEta_hist = new TH1F("DeltaEta", "Delta Eta (hit - track);#Delta#eta;Counts", 200, -1, 1);
        TH1F *DeltaPhi_hist = new TH1F("DeltaPhi", "Delta Phi (hit - track);#Delta#phi;Counts", 200, -1, 1);

        TH2F *hHitEtaR   = new TH2F("hHitEtaR", "Hit map: #eta vs R;#eta;R [mm]", 120, -1.5, 1.5, 200, 2240, 2310);
        TH2F *hHitPhiR   = new TH2F("hHitPhiR", "Hit map: #phi vs R;#phi [rad];R [mm]", 180, -TMath::Pi(), TMath::Pi(), 200, 2240, 2310);
        TH2F *hHitEtaPhi = new TH2F("hHitEtaPhi", "Hit map: #eta vs #phi;#eta;#phi [rad]", 120, -1.5, 1.5, 180, -TMath::Pi(), TMath::Pi());

        // ---------------------------------------------------------------
        // Per-track feature histograms (quick look at muon/pion separation)
        // ---------------------------------------------------------------
        TH1F *h_EoverP       = new TH1F("h_EoverP", "E_{HCal}/p_{track};E/p;Counts", 150, 0, 3);
        TH1F *h_AvgHitEnergy = new TH1F("h_AvgHitEnergy", "Average Hit Energy per Track;E_{avg} [GeV];Counts", 200, 0, 0.5);
        TH1F *h_NHits        = new TH1F("h_NHits", "Number of Hits per Track;N_{hits};Counts", 60, 0, 60);
        TH1F *h_SpreadPhi    = new TH1F("h_SpreadPhi", "Spread in #phi (max-min);#Delta#phi [rad];Counts", 100, 0, 1.0);
        TH1F *h_SpreadEta    = new TH1F("h_SpreadEta", "Spread in #eta (max-min);#Delta#eta;Counts", 100, 0, 1.0);
        TH1F *h_SpreadR      = new TH1F("h_SpreadR", "Radial spread of hits (max-min);#Delta R [mm];Counts", 100, 0, 100);
        TH1F *h_MaxHitFrac   = new TH1F("h_MaxHitFrac", "Max hit energy / total;E_{max}/E_{tot};Counts", 100, 0, 1.05);
        TH1F *h_EnergyStdDev = new TH1F("h_EnergyStdDev", "Std Dev of hit energies;#sigma_{E} [GeV];Counts", 100, 0, 0.3);

        // ---------------------------------------------------------------
        // NEW: R dispersion (unweighted / energy-weighted), energy
        // concentration, and eta/phi dispersion (energy-weighted).
        // All computed around the ENERGY-WEIGHTED MEAN of the track's hits
        // (analogous to how the official CalorimeterClusterShape algorithm
        // computes radius/dispersion around the cluster centroid).
        // ---------------------------------------------------------------
        TH1F *h_R_Disp           = new TH1F("h_R_Disp", "R dispersion (unweighted, around energy-weighted mean R);#sigma_{R} [mm];Counts", 100, 0, 50);
        TH1F *h_R_DispWeighted   = new TH1F("h_R_DispWeighted", "R dispersion (energy-weighted);#sigma_{R}^{w} [mm];Counts", 100, 0, 50);
        TH1F *h_EnergyConcentration = new TH1F("h_EnergyConcentration", "Energy concentration #Sigma E_{i}^{2} / (#Sigma E_{i})^{2};Concentration;Counts", 100, 0, 1.05);
        TH1F *h_Eta_DispWeighted = new TH1F("h_Eta_DispWeighted", "#eta dispersion (energy-weighted, around track);#sigma_{#eta}^{w};Counts", 100, 0, 0.3);
        TH1F *h_Phi_DispWeighted = new TH1F("h_Phi_DispWeighted", "#phi dispersion (energy-weighted, around track);#sigma_{#phi}^{w} [rad];Counts", 100, 0, 0.3);

        int eventID = 0;

        while (tree_reader.Next())
        {
            eventID++;
            if (eventID == 200000) break;
            // NOTE: the old "if(eventID==4) break;" left-over test line was removed
            // (it was truncating processing to 3 events per file).

            if (eventID % 50000 == 0) cout << "File " << name << " and event number... " << eventID << endl;

            for (size_t particle = 0; particle < trackMomX.GetSize(); particle++)
            {
                TLorentzVector Partic;
                Partic.SetPxPyPzE(trackMomX[particle], trackMomY[particle], trackMomZ[particle], trackEng[particle]);
                double trackP = Partic.P();
                //if (abs(Partic.Eta()) > 0.2) continue;
                //if (trackP > 4) continue;

                double sumEnergy = 0.0;
                double phiMin = 1e9, phiMax = -1e9;
                double etaMin = 1e9, etaMax = -1e9;
                double Rmin = 1e9, Rmax = -1e9;
                double maxHitE = -1.0;

                vector<double> trackHitEnergies;
                vector<double> trackHitR;      // R position of each hit
                vector<double> trackHitDEta;   // hitEta - trackEta
                vector<double> trackHitDPhi;   // wrapped(hitPhi - trackPhi)

                for (size_t iCluster = 0; iCluster < hcalClusters.GetSize(); ++iCluster)
                {
                    if (simuAssoc[particle] != simuAssocHcalBarrel[iCluster]) continue;

                    unsigned int begin = clusterHitBegin[iCluster];
                    unsigned int end   = clusterHitEnd[iCluster];
                    int nHitsInCluster = static_cast<int>(end - begin);
                    if (nHitsInCluster <= 0) continue;

                    for (unsigned int i = begin; i < end; ++i)
                    {
                        int hitIndex = clusterHitAssoc[i];

                        float HitE = hitEnergy[hitIndex];
                        float Hitx = hitPosX[hitIndex];
                        float Hity = hitPosY[hitIndex];
                        float Hitz = hitPosZ[hitIndex];
                        float HitT = hitTime[hitIndex];

                        HitTime_hist->Fill(HitT);

                        if(HitT>20) continue;

                        TVector3 hitVec(Hitx, Hity, Hitz);
                        double hitEta = hitVec.Eta();
                        double hitPhi = hitVec.Phi();
                        double R = sqrt(Hitx * Hitx + Hity * Hity);

                        // Diagnostic histograms (per hit, unchanged)
                        HitR_hist->Fill(R);
                        hHitEtaR->Fill(hitEta, R);
                        hHitPhiR->Fill(hitPhi, R);
                        hHitEtaPhi->Fill(hitEta, hitPhi);
                        HitPhi_hist->Fill(hitPhi);
                        HitZ_hist->Fill(Hitz);
                        HitEnergy_hist->Fill(HitE);

                        double dEta = hitEta - Partic.Eta();
                        double dPhi = TVector2::Phi_mpi_pi(hitPhi - Partic.Phi());
                        DeltaEta_hist->Fill(dEta);
                        DeltaPhi_hist->Fill(dPhi);

                        // --- akumulacja PER TRACK (nie per klaster) ---
                        sumEnergy += HitE;
                        trackHitEnergies.push_back(HitE);
                        trackHitR.push_back(R);
                        trackHitDEta.push_back(dEta);
                        trackHitDPhi.push_back(dPhi);

                        phiMin = min(phiMin, hitPhi); phiMax = max(phiMax, hitPhi);
                        etaMin = min(etaMin, hitEta); etaMax = max(etaMax, hitEta);
                        Rmin   = min(Rmin, R);        Rmax   = max(Rmax, R);

                        if (HitE > maxHitE) maxHitE = HitE;
                    }
                }

                if (sumEnergy <= 0) continue;

                int nHitsTrack = static_cast<int>(trackHitEnergies.size());

                // --- Hit-energy statistics, computed once for the entire track ---
                double meanE = sumEnergy / nHitsTrack;
                double sumSqDevE = 0.0;
                double sumE2 = 0.0; // for energy concentration
                for (double e : trackHitEnergies)
                {
                    double devE = e - meanE;
                    sumSqDevE += devE * devE;
                    sumE2 += e * e;
                }
                double energyStdDev = sqrt(sumSqDevE / nHitsTrack);
                double energyConcentration = sumE2 / (sumEnergy * sumEnergy);

                h_NHits->Fill(nHitsTrack);
                h_AvgHitEnergy->Fill(meanE);
                h_EnergyStdDev->Fill(energyStdDev);
                h_EnergyConcentration->Fill(energyConcentration);

                // -------------------- Remaining per-track features --------------------
                double spreadPhi = phiMax - phiMin;
                double spreadEta = etaMax - etaMin;
                double spreadR = Rmax - Rmin;

                h_EoverP->Fill(sumEnergy / trackP);
                h_SpreadPhi->Fill(spreadPhi);
                h_SpreadEta->Fill(spreadEta);
                h_SpreadR->Fill(spreadR);
                h_MaxHitFrac->Fill(maxHitE / sumEnergy);

                // ---------------------------------------------------------------
                // NEW: R dispersion (unweighted + energy-weighted), computed
                // around the energy-weighted mean R of the track's hits
                // (analogous to the official radius/dispersion definition,
                // but restricted to the R coordinate).
                // ---------------------------------------------------------------
                double sumRw = 0.0;
                for (size_t k = 0; k < trackHitR.size(); ++k)
                    sumRw += trackHitR[k] * trackHitEnergies[k];
                double meanR_w = sumRw / sumEnergy;

                double sumR2diff  = 0.0; // unweighted sum of squared deviations
                double sumR2diffW = 0.0; // energy-weighted sum of squared deviations
                double sumDEta2diffW = 0.0;
                double sumDPhi2diffW = 0.0;
                for (size_t k = 0; k < trackHitR.size(); ++k)
                {
                    double dR = trackHitR[k] - meanR_w;
                    sumR2diff  += dR * dR;
                    sumR2diffW += dR * dR * trackHitEnergies[k];

                    double dEtaK = trackHitDEta[k];
                    double dPhiK = trackHitDPhi[k];
                    sumDEta2diffW += dEtaK * dEtaK * trackHitEnergies[k];
                    sumDPhi2diffW += dPhiK * dPhiK * trackHitEnergies[k];
                }

                double R_disp_unweighted = (nHitsTrack > 1) ? sqrt(sumR2diff / (nHitsTrack - 1)) : 0.0;
                double R_disp_weighted   = sqrt(sumR2diffW / sumEnergy);
                double Eta_disp_weighted = sqrt(sumDEta2diffW / sumEnergy);
                double Phi_disp_weighted = sqrt(sumDPhi2diffW / sumEnergy);

                h_R_Disp->Fill(R_disp_unweighted);
                h_R_DispWeighted->Fill(R_disp_weighted);
                h_Eta_DispWeighted->Fill(Eta_disp_weighted);
                h_Phi_DispWeighted->Fill(Phi_disp_weighted);
            }
        }
        cout << "===========================" << endl;
        cout << "End of " << name << " file" << endl;
        cout << "Number of events processed: " << eventID << endl;
        cout << "===========================" << endl;

        outfile->Write();
        outfile->Close();

        delete mychain;
    }
}