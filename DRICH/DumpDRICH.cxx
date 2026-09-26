// DumpDRICH_Podio.cxx
//
// Loop over events and print dRICH contents (only events with dRICH hits/PID),
// useful for the pion/muon classifier:
//   - RawTrackerHit                 -> cellID, charge, timeStamp
//   - MCRecoTrackerHitAssociation   -> sim hit: pozycja, edep, czas, PDG
//   - TrackSegment                  -> punkty toru, pedy, dlugosc
//   - CherenkovParticleID (IRT)     -> npe, n, photon energy, hypotheses, photon angles
//   - edm4hep::ParticleID           -> type, pdg, algType, likelihood
//   - MCParticles                   -> prawda MC (etykieta)
//
// Collection names are detected automatically (search for "DRICH" in the name).

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include <edm4eic/RawTrackerHitCollection.h>
#include <edm4eic/MCRecoTrackerHitAssociationCollection.h>
#include <edm4eic/TrackSegmentCollection.h>
#include <edm4eic/CherenkovParticleIDCollection.h>
#include <edm4hep/MCParticleCollection.h>
#include <edm4hep/SimTrackerHitCollection.h>
#include <edm4hep/ParticleIDCollection.h>

#include <TFile.h>
#include <TH1D.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <glob.h>

using namespace std;

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
// List dRICH collections in the frame
// -----------------------------------------------------------------------------
vector<string> FindDRICHCollections(const podio::Frame &frame, bool verbose)
{
    vector<string> names;
    for (const auto &name : frame.getAvailableCollections())
    {
        if (name.find("DRICH") == string::npos) continue;
        names.push_back(name);

        if (verbose)
        {
            const auto *coll = frame.get(name);
            cout << "    " << name << "  [" << (coll ? string(coll->getTypeName()) : "?")
                 << "]  size=" << (coll ? coll->size() : 0) << endl;
        }
    }
    sort(names.begin(), names.end());
    return names;
}

// -----------------------------------------------------------------------------
// Does the event contain any dRICH hits or PID?
// -----------------------------------------------------------------------------
bool EventHasDRICHHits(const podio::Frame &frame, const vector<string> &names)
{
    for (const auto &name : names)
    {
        const auto *coll = frame.get(name);
        if (!coll) continue;
        const string type = string(coll->getTypeName());

        if ((type == "edm4eic::RawTrackerHitCollection" ||
             type == "edm4eic::CherenkovParticleIDCollection") && coll->size() > 0)
            return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Raw hits
// -----------------------------------------------------------------------------
void DumpRawHits(const podio::Frame &frame, const string &name, size_t maxPrint)
{
    const auto &hits = frame.get<edm4eic::RawTrackerHitCollection>(name);
    cout << "  [" << name << "] RawTrackerHit, count: " << hits.size() << endl;

    size_t n = 0;
    for (const auto &hit : hits)
    {
        if (n++ >= maxPrint) { cout << "    ..." << endl; break; }
        cout << "    cellID=" << hit.getCellID()
             << "  charge=" << hit.getCharge()
             << "  timeStamp=" << hit.getTimeStamp() << endl;
    }
}

// -----------------------------------------------------------------------------
// Raw-hit to simulated-hit associations
// -----------------------------------------------------------------------------
void DumpAssociations(const podio::Frame &frame, const string &name, size_t maxPrint)
{
    const auto &assocs = frame.get<edm4eic::MCRecoTrackerHitAssociationCollection>(name);
    cout << "  [" << name << "] MCRecoTrackerHitAssociation, count: " << assocs.size() << endl;

    size_t n = 0;
    for (const auto &a : assocs)
    {
        if (n++ >= maxPrint) { cout << "    ..." << endl; break; }

        const auto raw = a.getRawHit();
        const auto sim = a.getSimHit();
        const auto pos = sim.getPosition();

        cout << "    raw.cellID=" << raw.getCellID()
             << "  weight=" << a.getWeight()
             << "  | sim: x=" << pos.x << " y=" << pos.y << " z=" << pos.z << " mm"
             << "  t=" << sim.getTime() << " ns"
             << "  edep=" << sim.getEDep() * 1e6 << " keV";

        if (sim.getParticle().isAvailable())
            cout << "  pdg=" << sim.getParticle().getPDG();
        cout << endl;
    }
}

// -----------------------------------------------------------------------------
// TrackSegment
// -----------------------------------------------------------------------------
void DumpTrackSegments(const podio::Frame &frame, const string &name)
{
    const auto &segs = frame.get<edm4eic::TrackSegmentCollection>(name);
    cout << "  [" << name << "] TrackSegment, count: " << segs.size() << endl;

    size_t i = 0;
    for (const auto &seg : segs)
    {
        cout << "    segment " << i++ << ": length=" << seg.getLength()
             << " mm  (+-" << seg.getLengthError() << ")"
             << "  punktow=" << seg.points_size() << endl;

        for (const auto &p : seg.getPoints())
        {
            const auto &pos = p.position;
            const auto &mom = p.momentum;
            double pTot = sqrt(mom.x * mom.x + mom.y * mom.y + mom.z * mom.z);
            cout << "      pt: x=" << pos.x << " y=" << pos.y << " z=" << pos.z << " mm"
                 << "  |p|=" << pTot << " GeV"
                 << "  pathlength=" << p.pathlength << " mm" << endl;
        }
    }
}

// -----------------------------------------------------------------------------
// CherenkovParticleID
// -----------------------------------------------------------------------------
void DumpCherenkovPID(const podio::Frame &frame, const string &name, size_t maxPhotons,
                      TH1D *hNpe, TH1D *hTheta)
{
    const auto &pids = frame.get<edm4eic::CherenkovParticleIDCollection>(name);
    cout << "  [" << name << "] CherenkovParticleID, count: " << pids.size() << endl;

    size_t i = 0;
    for (const auto &pid : pids)
    {
        cout << "    PID " << i++ << ":"
             << "  npe=" << pid.getNpe()
             << "  n(refr.)=" << pid.getRefractiveIndex()
             << "  photon_energy=" << pid.getPhotonEnergy() * 1e9 << " eV" << endl;

        if (hNpe) hNpe->Fill(pid.getNpe());

        cout << "      hypotheses (" << pid.hypotheses_size() << "):" << endl;
        for (const auto &h : pid.getHypotheses())
        {
            cout << "        pdg=" << h.PDG
                 << "  npe=" << h.npe
                 << "  weight=" << h.weight << endl;
        }

        cout << "      fotony (theta, phi) [rad], n=" << pid.thetaPhiPhotons_size() << ":" << endl;
        size_t nPh = 0;
        double thetaSum = 0.0;
        for (const auto &tp : pid.getThetaPhiPhotons())
        {
            thetaSum += tp.a; // a = theta, b = phi
            if (hTheta) hTheta->Fill(tp.a * 1e3); // mrad
            if (nPh < maxPhotons)
                cout << "        theta=" << tp.a << "  phi=" << tp.b << endl;
            else if (nPh == maxPhotons)
                cout << "        ..." << endl;
            ++nPh;
        }
        if (nPh > 0)
            cout << "      <theta> = " << thetaSum / nPh * 1e3 << " mrad" << endl;

        cout << "      associated raw hits: " << pid.rawHitAssociations_size() << endl;
        if (pid.getChargedParticle().isAvailable())
        {
            const auto seg = pid.getChargedParticle();
              cout << "      associated TrackSegment: length=" << seg.getLength()
                  << " mm, points=" << seg.points_size() << endl;
        }
    }
}

// -----------------------------------------------------------------------------
// edm4hep::ParticleID (DRICHParticleIDs, DRICHTruthSeededParticleIDs)
// -----------------------------------------------------------------------------
void DumpParticleIDs(const podio::Frame &frame, const string &name)
{
    const auto &pids = frame.get<edm4hep::ParticleIDCollection>(name);
    cout << "  [" << name << "] ParticleID, count: " << pids.size() << endl;

    for (const auto &pid : pids)
    {
        cout << "    type=" << pid.getType()
             << "  pdg=" << pid.getPDG()
             << "  algType=" << pid.getAlgorithmType()
             << "  likelihood=" << pid.getLikelihood()
             << "  parameters=" << pid.parameters_size() << endl;
    }
}

// -----------------------------------------------------------------------------
// MC truth
// -----------------------------------------------------------------------------
void DumpMCTruth(const podio::Frame &frame, size_t maxPrint)
{
    const auto &parts = frame.get<edm4hep::MCParticleCollection>("MCParticles");
    cout << "  [MCParticles] count: " << parts.size() << endl;

    size_t n = 0;
    for (const auto &p : parts)
    {
        if (p.getGeneratorStatus() != 1) continue;
        if (n++ >= maxPrint) { cout << "    ..." << endl; break; }

        const auto mom = p.getMomentum();
        double pTot = sqrt(mom.x * mom.x + mom.y * mom.y + mom.z * mom.z);
        double theta = pTot > 0 ? acos(mom.z / pTot) : 0.0;

        cout << "    pdg=" << p.getPDG()
             << "  |p|=" << pTot << " GeV"
             << "  theta=" << theta << " rad"
             << "  charge=" << p.getCharge() << endl;
    }
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
void DumpDRICH()
{
    vector<string> fileList = ExpandGlob("/run/media/epic/Data/Background/Muons/Continuous/reco_*.root"); // <-- replace with your path

    podio::ROOTReader reader;
    reader.openFiles(fileList);

    unsigned nEvents = reader.getEntries("events");
    cout << "Event count: " << nEvents << endl;

    TFile *outFile = new TFile("drich_dump.root", "RECREATE");
    auto *hNpe   = new TH1D("h_npe", "dRICH: npe per PID;npe;Count", 100, 0, 100);
    auto *hTheta = new TH1D("h_theta", "dRICH: photon Cherenkov angle;#theta [mrad];Count", 200, 0, 400);

    const unsigned maxToPrint = 5;   // Number of events WITH HITS to print
    unsigned nPrinted = 0;
    unsigned nWithHits = 0;
    unsigned nScanned = 0;

    for (unsigned entry = 0; entry < nEvents && nPrinted < maxToPrint; ++entry)
    {
        ++nScanned;
        podio::Frame frame(reader.readEntry("events", entry));

        auto names = FindDRICHCollections(frame, false);
        if (!EventHasDRICHHits(frame, names)) continue; // Skip empty events

        ++nWithHits;
        cout << "\n================ Event " << entry << " (with hits) ================" << endl;

        if (nPrinted == 0)
        {
            cout << "  dRICH collections in frame:" << endl;
            FindDRICHCollections(frame, true);
        }

        for (const auto &name : names)
        {
            const auto *coll = frame.get(name);
            if (!coll) continue;
            const string type = string(coll->getTypeName());

            if (type == "edm4eic::RawTrackerHitCollection")
                DumpRawHits(frame, name, 10);
            else if (type == "edm4eic::MCRecoTrackerHitAssociationCollection")
                DumpAssociations(frame, name, 10);
            else if (type == "edm4eic::TrackSegmentCollection")
                DumpTrackSegments(frame, name);
            else if (type == "edm4eic::CherenkovParticleIDCollection")
                DumpCherenkovPID(frame, name, 10, hNpe, hTheta);
            else if (type == "edm4hep::ParticleIDCollection")
                DumpParticleIDs(frame, name);
            // Skip LinkCollections
        }

        DumpMCTruth(frame, 5);
        ++nPrinted;
    }

        cout << "\nScanned " << nScanned << " events, found " << nWithHits
            << " with dRICH hits (printed " << nPrinted << ")." << endl;

    outFile->cd();
    hNpe->Write();
    hTheta->Write();
    outFile->Close();
    delete outFile;
    cout << "Histograms saved to: drich_dump.root" << endl;
}

#ifndef __CLING__
int main(int argc, char** argv)
{
    DumpDRICH();
    return 0;
}
#endif