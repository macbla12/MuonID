#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TROOT.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TMath.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

void ClusterShape_HitCheck()
{
    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

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

        // --- HCal barrel clusters only ---
        TTreeReaderArray<float> clusterEnergy(tree_reader, "HcalBarrelClusters.energy");
        TTreeReaderArray<float> clusterPosX(tree_reader, "HcalBarrelClusters.position.x");
        TTreeReaderArray<float> clusterPosY(tree_reader, "HcalBarrelClusters.position.y");
        TTreeReaderArray<float> clusterPosZ(tree_reader, "HcalBarrelClusters.position.z");

        TTreeReaderArray<unsigned int> clusterHitBegin(tree_reader, "HcalBarrelClusters.hits_begin");
        TTreeReaderArray<unsigned int> clusterHitEnd(tree_reader, "HcalBarrelClusters.hits_end");
        TTreeReaderArray<int> clusterHitAssoc(tree_reader, "_HcalBarrelClusters_hits.index");

        TTreeReaderArray<unsigned int> clusterShapeBegin(tree_reader, "HcalBarrelClusters.shapeParameters_begin");
        TTreeReaderArray<unsigned int> clusterShapeEnd(tree_reader, "HcalBarrelClusters.shapeParameters_end");
        TTreeReaderArray<float> clusterShapeParams(tree_reader, "_HcalBarrelClusters_shapeParameters");

        // --- HCal Barrel Rec Hits ---
        TTreeReaderArray<float> hitEnergy(tree_reader, "HcalBarrelRecHits.energy");
        TTreeReaderArray<float> hitPosX(tree_reader, "HcalBarrelRecHits.position.x");
        TTreeReaderArray<float> hitPosY(tree_reader, "HcalBarrelRecHits.position.y");
        TTreeReaderArray<float> hitPosZ(tree_reader, "HcalBarrelRecHits.position.z");

        TFile *outfile = new TFile(Form("Plots/ShapeCheck/ShapeCheck_%s.root", name.c_str()), "RECREATE");

        // shape[1] = radius, shape[2] = dispersion
        TH1F *h_Radius_Stored   = new TH1F("h_Radius_Stored",   "Stored radius (shape[1]);Radius [mm];Counts", 200, 0, 200);
        TH1F *h_Radius_Computed = new TH1F("h_Radius_Computed", "Radius computed from hits;Radius [mm];Counts", 200, 0, 200);
        TH1F *h_Radius_Diff     = new TH1F("h_Radius_Diff",     "Radius: stored - computed;#Delta R [mm];Counts", 200, -0.1, 0.1);
        TH2F *h_Radius_Corr     = new TH2F("h_Radius_Corr",     "Radius: stored vs computed;Computed R [mm];Stored R [mm]", 10000, 0, 200, 10000, 0, 200);

        int eventID = 0;
        int singleClusterEvents = 0;

        while (tree_reader.Next())
        {
            eventID++;
            if (eventID % 50000 == 0) cout << "File " << name << " event: " << eventID << endl;
            if (eventID == 200000) break;

            if (clusterEnergy.GetSize() != 1) continue;
            singleClusterEvents++;

            const int iCluster = 0;

            float cx = clusterPosX[iCluster];
            float cy = clusterPosY[iCluster];
            float cz = clusterPosZ[iCluster];

            unsigned int hbegin = clusterHitBegin[iCluster];
            unsigned int hend   = clusterHitEnd[iCluster];
            int n = static_cast<int>(hend - hbegin);
            if (n <= 0) continue;

            float sum_r2 = 0.0f;

            for (unsigned int i = hbegin; i < hend; ++i)
            {
                int hitIndex = clusterHitAssoc[i];
                float dx = hitPosX[hitIndex] - cx;
                float dy = hitPosY[hitIndex] - cy;
                float dz = hitPosZ[hitIndex] - cz;
                float r2 = dx*dx + dy*dy + dz*dz;

                sum_r2 += r2;
            }

            double radiusComputed     = sqrt(sum_r2 / max(1, n - 1));
            unsigned int sbegin = clusterShapeBegin[iCluster];
            unsigned int send   = clusterShapeEnd[iCluster];
            int nShapeParams = static_cast<int>(send - sbegin);
            double radiusStored     = clusterShapeParams[sbegin + 0];
            h_Radius_Stored->Fill(radiusStored);
            h_Radius_Computed->Fill(radiusComputed);
            h_Radius_Diff->Fill(radiusStored - radiusComputed);
            h_Radius_Corr->Fill(radiusComputed, radiusStored);
        }

        cout << "===========================" << endl;
        cout << "End of " << name << " file" << endl;
        cout << "Events processed: " << eventID << endl;
        cout << "===========================" << endl;

        outfile->Write();
        outfile->Close();

        delete mychain;
    }
}