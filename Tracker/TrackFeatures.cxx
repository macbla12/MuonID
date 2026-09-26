// CompareEdep_MuonPion.cxx
//
// Simple code: reads two sets of files (muons and pions), calculates five
// edep-related features for every track in "CentralCKFTracks" (hits are taken ONLY
// from track.getMeasurements() -> meas.getHits(), not from the TrackerHit collection),
// and writes the histograms to one ROOT file.
//
// Histograms are named: <feature>_Muon and <feature>_Pion,
// so they can easily be overlaid later in ROOT, for example:
//   TFile f("edep_compare.root");
//   ((TH1D*)f.Get("EdepSum_Muon"))->Draw("hist norm");
//   ((TH1D*)f.Get("EdepSum_Pion"))->Draw("hist norm same");
//
// Usage: root -l -b -q CompareEdep_MuonPion.cxx+

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include <edm4eic/TrackCollection.h>
#include <edm4eic/Measurement2DCollection.h>
#include <edm4eic/TrackerHitCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <cmath>

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <glob.h>

using namespace std;

// =============================================================================
// SETTINGS (change paths and limits here)
// =============================================================================
const string MUON_FILES = "/run/media/epic/Data/Background/Muons/Continuous/reco_aMuonsLowE*.root";
const string PION_FILES = "/run/media/epic/Data/Background/Pions/Continuous/reco_piPlusLowE*.root"; 
const string OUTPUT     = "Plots/edep_compare.root";
const long   MAX_EVENTS = 100000;   // maximum number of events per sample
const double P_MAX = 1.0;   // [GeV] only tracks with p < P_MAX are stored
const int N_PBINS = 20;   // number of momentum bins on the X axis (0 - P_MAX)

// =============================================================================
// 5 FEATURES (name, X-axis title, number of bins, min, max)
// To change the histogram range, edit only this table.
// =============================================================================
struct FeatureDef
{
    string name;
    string title;
    int    nBins;
    double xMin;
    double xMax;
};

const vector<FeatureDef> FEATURES = {
    {"EdepSum",    "Track edep sum;Track edep [keV];Tracks",           100, 0,  1000},
    {"NHits",      "Number of hits in track;N hits;Tracks",              20, -0.5, 19.5},
    {"EdepMean",   "Mean edep per hit;Mean edep [keV];Tracks",   100, 0,  200},
    {"EdepMax",    "Maximum edep in track;Max edep [keV];Tracks",   100, 0,  500},
    {"EdepMedian", "Median edep of hits;Median edep [keV];Tracks",    100, 0,  40},
};
const int NFEAT = 5;

// =============================================================================
// Calculate features for one track.
// Returns false if the track has no hits.
// =============================================================================
bool ComputeFeatures(const edm4eic::Track &track, double out[NFEAT])
{
    vector<double> edeps; // [keV]

    for (const auto &meas : track.getMeasurements())
        for (const auto &hit : meas.getHits())
            edeps.push_back(hit.getEdep() * 1e6); // GeV -> keV

    if (edeps.empty()) return false;

    double sum = 0;
    for (double e : edeps) sum += e;

    sort(edeps.begin(), edeps.end());
    const size_t n = edeps.size();
    double median = (n % 2) ? edeps[n / 2] : 0.5 * (edeps[n / 2 - 1] + edeps[n / 2]);

    out[0] = sum;                 // EdepSum
    out[1] = (double)n;           // NHits
    out[2] = sum / n;             // EdepMean
    out[3] = edeps.back();        // EdepMax
    out[4] = median;              // EdepMedian
    return true;
}

// =============================================================================
// Helper: expand a file pattern
// =============================================================================
vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t g;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) result.push_back(g.gl_pathv[i]);
    globfree(&g);
    return result;
}

// =============================================================================
// Process one sample (muons or pions).
// label = "Muon" / "Pion" -> included in histogram names.
// =============================================================================
void ProcessSample(const string &pattern, const string &label, TFile *outFile)
{
    vector<string> files = ExpandGlob(pattern);
    if (files.empty())
    {
        cerr << "No files found for pattern: " << pattern << endl;
        return;
    }

    // 1D histograms (feature) and 2D histograms (momentum vs feature)
    TH1D *hist[NFEAT];
    TH2D *hist2D[NFEAT];
    for (int i = 0; i < NFEAT; ++i)
    {
        const auto &f = FEATURES[i];

        string hname = f.name + "_" + label;
        hist[i] = new TH1D(hname.c_str(), f.title.c_str(), f.nBins, f.xMin, f.xMax);

        string h2name  = f.name + "_vsP_" + label;
        string h2title = f.name + " vs p (" + label + ");p [GeV];" + f.name + ";Tracks";
        hist2D[i] = new TH2D(h2name.c_str(), h2title.c_str(),
                             N_PBINS, 0.0, P_MAX,
                             f.nBins, f.xMin, f.xMax);
    }

    podio::ROOTReader reader;
    reader.openFiles(files);

    long nEvents = min<long>(reader.getEntries("events"), MAX_EVENTS);
    long nTracks = 0;

    cout << "[" << label << "] files: " << files.size()
         << ", events: " << nEvents << endl;

    for (long iev = 0; iev < nEvents; ++iev)
    {
        podio::Frame frame(reader.readEntry("events", iev));
        const auto &reco_parts =
            frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedChargedParticles");

        for (const auto &rcp : reco_parts)
        {
            // Particle momentum [GeV]
            const auto mom = rcp.getMomentum();
            const double p = std::sqrt(double(mom.x) * mom.x +
                                       double(mom.y) * mom.y +
                                       double(mom.z) * mom.z);
            if (p >= P_MAX) continue;

            for (const auto &track : rcp.getTracks())
            {
                double val[NFEAT];
                if (!ComputeFeatures(track, val)) continue;

                for (int i = 0; i < NFEAT; ++i)
                {
                    hist[i]->Fill(val[i]);
                    hist2D[i]->Fill(p, val[i]);
                }
                ++nTracks;
            }
        }
    }

    cout << "[" << label << "] tracks with hits: " << nTracks << endl;

    outFile->cd();
    for (int i = 0; i < NFEAT; ++i)
    {
        hist[i]->Write();
        hist2D[i]->Write();
    }
}

// =============================================================================
// main
// =============================================================================
void TrackFeatures()
{
    TFile *outFile = new TFile(OUTPUT.c_str(), "RECREATE");

    ProcessSample(MUON_FILES, "Muon", outFile);
    ProcessSample(PION_FILES, "Pion", outFile);

    outFile->Close();
    delete outFile;
    cout << "Histograms saved to: " << OUTPUT << endl;
}

#ifndef __CLING__
int main()
{
    TrackFeatures();
    return 0;
}
#endif