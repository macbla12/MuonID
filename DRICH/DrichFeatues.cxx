// DrichFeatues.cxx
//
// Reads two file sets (muons and pions). For each CherenkovParticleID
// in dRICH (aerogel only), computes features that may distinguish muons from pions
// and saves the histograms to one ROOT file.
//
// Histograms:  <feature>_Aerogel_<Sample>        (1D)
//              <feature>_vsP_Aerogel_<Sample>    (2D: p vs feature)
// e.g. ThetaMean_vsP_Aerogel_Muon and ThetaMean_vsP_Aerogel_Pion
//
// Usage:  root -l -b -q DrichFeatues.cxx+

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include <edm4eic/CherenkovParticleIDCollection.h>
#include <edm4eic/TrackSegmentCollection.h>

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <glob.h>

using namespace std;

// =============================================================================
// SETTINGS
// =============================================================================
const string MUON_FILES = "/run/media/epic/Data/Background/Muons/Continuous/reco_aMuonsLowE*.root";
const string PION_FILES = "/run/media/epic/Data/Background/Pions/Continuous/reco_piPlusLowE*.root";
const string OUTPUT     = "Plots/drich_compare.root";
const long   MAX_EVENTS = 300000; // Maximum events per sample (few events contain dRICH hits)
const double P_MAX      = 2.0;    // [GeV] momentum axis
const int    N_PBINS    = 50;

// Radiator: PID collection, segment collection (momentum source), refractive
// index (the IRT n field is zero here, so provide it manually), and theta-axis scale.
struct Radiator
{
    string label;
    string collName;      // CherenkovParticleID
    string trackCollName; // TrackSegment
    double n;
    double thetaMax;
};

const vector<Radiator> RADIATORS = {
    {"Aerogel", "DRICHAerogelIrtCherenkovParticleID", "DRICHAerogelTracks", 1.02, 300.0},
};

// =============================================================================
// FEATURES (name, title, bins, min, max, whether to scale range by radiator)
// =============================================================================
struct FeatureDef
{
    string name;
    string title;
    int    nBins;
    double xMin;
    double xMax;
    bool   isTheta; // true -> xMax *= thetaMax/60
};

const vector<FeatureDef> FEATURES = {
    {"NPE",         "Number of photons (npe);npe;PID",                    21,    -.5,   20.5, false},
    {"ThetaMean",   "Mean photon #theta;<#theta> [mrad];PID",            120,    0,   60, true },
    {"ThetaMedian", "Median photon #theta;#theta_{med} [mrad];PID",      120,    0,   60, true },
    {"ThetaRMS",    "Photon #theta RMS;#sigma_{#theta} [mrad];PID",     100,    0,   10, true },
    {"Mass2Est",    "m^{2} estimate from angle;m^{2} [GeV^{2}];PID",     120, -0.04, 0.04, false},
    {"WeightPi",    "#pi hypothesis weight;weight;PID",                  100,    0, 1000, false},
    {"dW_PiK",      "Weight(#pi) - Weight(K);#Deltaweight;PID",          100, -100,  500, false},
    {"dW_PiE",      "Weight(#pi) - Weight(e);#Deltaweight;PID",          100, -200,  200, false},
    {"NpePiHyp",    "npe under the #pi hypothesis;npe;PID",               31,    -.5,   30.5, false},
};
const int NFEAT = 9;

// =============================================================================
// Helpers
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

// Mean momentum from TrackSegment points [GeV]; -1 if there are no points
double MomentumFromSegment(const edm4eic::TrackSegment &seg)
{
    double sum = 0;
    int n = 0;
    for (const auto &pt : seg.getPoints())
    {
        const auto &m = pt.momentum;
        sum += sqrt(double(m.x) * m.x + double(m.y) * m.y + double(m.z) * m.z);
        ++n;
    }
    return n ? sum / n : -1.0;
}

// Features for one CherenkovParticleID. Returns false if there are no photons.
bool ComputeFeatures(const edm4eic::CherenkovParticleID &pid, double p, double nRef,
                     double out[NFEAT])
{
    vector<double> th; // [mrad]
    for (const auto &tp : pid.getThetaPhiPhotons())
        th.push_back(tp.a * 1e3); // a = theta
    if (th.empty()) return false;

    const size_t n = th.size();
    double sum = 0;
    for (double t : th) sum += t;
    const double mean = sum / n;

    double var = 0;
    for (double t : th) var += (t - mean) * (t - mean);
    const double rms = sqrt(var / n);

    sort(th.begin(), th.end());
    const double median = (n % 2) ? th[n / 2] : 0.5 * (th[n / 2 - 1] + th[n / 2]);

    // m^2 = p^2 (n^2 cos^2(theta) - 1)
    const double c = cos(mean * 1e-3);
    const double m2 = p * p * (nRef * nRef * c * c - 1.0);

    // IRT hypotheses (pion, kaon, electron; no muon hypothesis)
    double wPi = 0, wK = 0, wE = 0, npePi = 0;
    for (const auto &h : pid.getHypotheses())
    {
        const int a = abs(int(h.PDG));
        if (a == 211)      { wPi = h.weight; npePi = h.npe; }
        else if (a == 321) { wK = h.weight; }
        else if (a == 11)  { wE = h.weight; }
    }

    out[0] = pid.getNpe();
    out[1] = mean;
    out[2] = median;
    out[3] = rms;
    out[4] = m2;
    out[5] = wPi;
    out[6] = wPi - wK;
    out[7] = wPi - wE;
    out[8] = npePi;
    return true;
}

// =============================================================================
// Process one sample
// =============================================================================
void ProcessSample(const string &pattern, const string &label, TFile *outFile)
{
    vector<string> files = ExpandGlob(pattern);
    if (files.empty())
    {
        cerr << "No files found for pattern: " << pattern << endl;
        return;
    }

    const size_t NR = RADIATORS.size();
    vector<vector<TH1D *>> hist(NR, vector<TH1D *>(NFEAT));
    vector<vector<TH2D *>> hist2D(NR, vector<TH2D *>(NFEAT));
    vector<long> nPID(NR, 0);
    vector<long> nNoSeg(NR, 0), nNoMom(NR, 0), nNoPhot(NR, 0);

    // --- Create histograms ---
    for (size_t r = 0; r < NR; ++r)
    {
        const auto &rad = RADIATORS[r];
        for (int i = 0; i < NFEAT; ++i)
        {
            const auto &f = FEATURES[i];
            const double xMax = f.isTheta ? f.xMax * rad.thetaMax / 60.0 : f.xMax;

            string hname = f.name + "_" + rad.label + "_" + label;
            hist[r][i] = new TH1D(hname.c_str(), f.title.c_str(), f.nBins, f.xMin, xMax);

            string h2name  = f.name + "_vsP_" + rad.label + "_" + label;
            string h2title = f.name + " vs p (" + rad.label + ", " + label + ");p [GeV];" +
                             f.name + ";PID";
            hist2D[r][i] = new TH2D(h2name.c_str(), h2title.c_str(),
                                    N_PBINS, 0.0, P_MAX, f.nBins, f.xMin, xMax);
        }
    }

    podio::ROOTReader reader;
    reader.openFiles(files);

    long nEvents = min<long>(reader.getEntries("events"), MAX_EVENTS);
        cout << "[" << label << "] files: " << files.size()
            << ", events: " << nEvents << endl;

    // --- Event loop ---
    for (long iev = 0; iev < nEvents; ++iev)
    {
        podio::Frame frame(reader.readEntry("events", iev));

        for (size_t r = 0; r < NR; ++r)
        {
            const auto &rad = RADIATORS[r];
            const auto *raw    = frame.get(rad.collName);
            const auto *rawSeg = frame.get(rad.trackCollName);
            if (!raw || !rawSeg || raw->size() == 0) continue;

            const auto &pids = frame.get<edm4eic::CherenkovParticleIDCollection>(rad.collName);
            const auto &segs = frame.get<edm4eic::TrackSegmentCollection>(rad.trackCollName);

            size_t idx = 0;
            for (const auto &pid : pids)
            {
                const size_t i = idx++;
                if (i >= segs.size()) { ++nNoSeg[r]; continue; }

                const double p = MomentumFromSegment(segs[i]);
                if (p <= 0)     { ++nNoMom[r]; continue; } // Segment has no points
                if (p >= P_MAX) continue;

                double val[NFEAT];
                if (!ComputeFeatures(pid, p, rad.n, val)) { ++nNoPhot[r]; continue; }

                for (int k = 0; k < NFEAT; ++k)
                {
                    hist[r][k]->Fill(val[k]);
                    hist2D[r][k]->Fill(p, val[k]);
                }
                ++nPID[r];
            }
        }
    }

    // --- Summarize and save ---
    outFile->cd();
    for (size_t r = 0; r < NR; ++r)
    {
        cout << "[" << label << "] " << RADIATORS[r].label
             << ": OK=" << nPID[r]
             << "  no photons=" << nNoPhot[r] << endl;

        for (int i = 0; i < NFEAT; ++i)
        {
            hist[r][i]->Write();
            hist2D[r][i]->Write();
        }
    }
}

// =============================================================================
// main
// =============================================================================
void DrichFeatues()
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
    DrichFeatues();
    return 0;
}
#endif