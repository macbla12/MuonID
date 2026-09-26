// DumpTrackerHits_Podio.cxx
//
// Retrieves tracker hits for a given track (CentralCKFTracks)
// and creates a three-dimensional scatter plot (TGraph2D) with the hits (X, Y, Z) for each event.
// The energy deposition of every track hit is also printed to the console.

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include <edm4eic/TrackerHitCollection.h>
#include <edm4eic/TrackCollection.h>
#include <edm4eic/Measurement2DCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>

#include <TFile.h>
#include <TGraph2D.h>

#include <iostream>
#include <string>
#include <vector>
#include <glob.h>

using namespace std;

// -----------------------------------------------------------------------------
// Helper that expands a file pattern (glob).
// -----------------------------------------------------------------------------
vector<string> ExpandGlob(const string &pattern)
{
    vector<string> result;
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i)
        result.push_back(string(glob_result.gl_pathv[i]));
    globfree(&glob_result);
    return result;
}

// -----------------------------------------------------------------------------
// VARIANT 1: Print all hits to the console
// -----------------------------------------------------------------------------
void DumpAllTrackerHits(const podio::Frame &frame, const vector<string> &collNames)
{
    for (const auto &collName : collNames)
    {
        const auto &hits = frame.get<edm4eic::TrackerHitCollection>(collName);

        cout << "  [" << collName << "] number of hits: " << hits.size() << endl;

        for (const auto &hit : hits)
        {
            auto pos = hit.getPosition(); // mm
            double t   = hit.getTime();   // ns
            double edep = hit.getEdep();  // GeV

            cout << "    x=" << pos.x << " y=" << pos.y << " z=" << pos.z
                 << "  t=" << t << " ns"
                 << "  edep=" << edep << " GeV" << endl;
        }
    }
}

// -----------------------------------------------------------------------------
// VARIANT 2: Print hits for the selected track to the console (including edep)
// -----------------------------------------------------------------------------
void DumpHitsForTrack(const podio::Frame &frame, size_t trackIndex)
{
    const auto &tracks = frame.get<edm4eic::TrackCollection>("CentralCKFTracks");

    if (trackIndex >= tracks.size())
    {
        cout << "    (no track with index " << trackIndex << ")" << endl;
        return;
    }

    const auto track = tracks[trackIndex];

    cout << "    Track with " << track.measurements_size() << " measurements" << endl;

    double edepSum = 0.0; // GeV
    int nHits = 0;

    for (const auto &meas : track.getMeasurements())
    {
        for (const auto &hit : meas.getHits())
        {
            auto pos = hit.getPosition();
            double edep = hit.getEdep(); // GeV

            cout << "      hit: x=" << pos.x << " y=" << pos.y << " z=" << pos.z
                 << "  t=" << hit.getTime() << " ns"
                 << "  edep=" << edep * 1e6 << " keV" << endl;

            edepSum += edep;
            ++nHits;
        }
    }

        cout << "    Total edep for track " << trackIndex << " = " << edepSum * 1e6 << " keV"
            << " (from " << nHits << " hits)" << endl;
}

// -----------------------------------------------------------------------------
// SAVE 3D SCATTER PLOT: Create and save a 3D scatter plot of track hits
// -----------------------------------------------------------------------------
void SaveTrackScatterPlot3D(const podio::Frame &frame, size_t trackIndex, unsigned eventNum, TFile *outFile)
{
    if (!outFile || !outFile->IsOpen()) return;

    const auto &tracks = frame.get<edm4eic::TrackCollection>("CentralCKFTracks");

    if (trackIndex >= tracks.size())
    {
        cout << "    (track " << trackIndex << " is missing in event " << eventNum << " -> skipping 3D scatter plot)" << endl;
        return;
    }

    const auto track = tracks[trackIndex];

    // Names and axis labels for the scatter plot
    string graphName = "scatter_track0_event_" + to_string(eventNum);
    string graphTitle = "3D Scatter Plot - Track " + to_string(trackIndex) + " (Event " + to_string(eventNum) + ");X [mm];Y [mm];Z [mm]";

    auto scatter = new TGraph2D();
    scatter->SetName(graphName.c_str());
    scatter->SetTitle(graphTitle.c_str());

    int pointIdx = 0;
    for (const auto &meas : track.getMeasurements())
    {
        for (const auto &hit : meas.getHits())
        {
            auto pos = hit.getPosition();
            scatter->SetPoint(pointIdx++, pos.x, pos.y, pos.z);
        }
    }

    // Style the points in the scatter plot
    scatter->SetMarkerStyle(20); // Solid dots
    scatter->SetMarkerSize(1.0); // Marker size
    scatter->SetMarkerColor(kRed + 1); // Red hits

    outFile->cd();
    scatter->Write();
    delete scatter;
}

// -----------------------------------------------------------------------------
// main - event loop with scatter plot generation
// -----------------------------------------------------------------------------
void DumpTracks()
{
    vector<string> trackerCollections = {
        "SiBarrelVertexRecHits",
        "SiBarrelTrackerRecHits",
        "SiEndcapTrackerRecHits",
        "MPGDBarrelRecHits",
    };

    vector<string> fileList = ExpandGlob("/run/media/epic/Data/Background/Muons/Continuous/reco_*.root"); // <-- replace with your path

    podio::ROOTReader reader;
    reader.openFiles(fileList);

    unsigned nEvents = reader.getEntries("events");
    cout << "Number of events: " << nEvents << endl;

    // ROOT output file with scatter plots
    TFile *outFile = new TFile("tracker_hits.root", "RECREATE");

    unsigned nToPrint = std::min<unsigned>(nEvents, 5); // Preview the first few events

    for (unsigned entry = 0; entry < nToPrint; ++entry)
    {
        podio::Frame frame(reader.readEntry("events", entry));

        cout << "Event " << entry << ":" << endl;

        // Variant 1: All hits in the event
        DumpAllTrackerHits(frame, trackerCollections);

        // Variant 2: Print hits for track 0 to the console (including edep)
        cout << "  Hits for track 0:" << endl;
        DumpHitsForTrack(frame, 0);

        // Create and save a 3D scatter plot for track 0 in this event
        SaveTrackScatterPlot3D(frame, 0, entry, outFile);
    }

    outFile->Close();
    delete outFile;
    cout << "\n3D scatter plots saved to file: tracker_hits.root" << endl;
}

#ifndef __CLING__
int main(int argc, char** argv)
{
    DumpTracks();
    return 0;
}
#endif