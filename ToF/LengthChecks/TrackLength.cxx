
#include <podio/ROOTReader.h>
#include <podio/Frame.h>
#include <edm4eic/TrackSegmentCollection.h>
#include <edm4hep/MCParticleCollection.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <glob.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "../ToFSim.cxx"   // Twój ToFSim (RK4)

// ---------- KONFIGURACJA ----------
const std::string FILE_PATTERN = "/run/media/epic/Data/Background/Muon/Continuous/reco_aMuonsLowE2.root";
const std::string SEG_COLL     = "CentralTrackSegments";
const std::string MC_COLL      = "MCParticles";
const int    MAX_EVENTS  = 10000;
const bool   DEBUG_PRINT = true;      // wypisz punkty segmentu w 1. evencie
const double MUON_MASS   = 0.10566;   // GeV

// Tryb wyboru punktow (pole pt.system jest = 0, wiec NIE uzywamy go):
//   ALL      - wszystkie punkty segmentu (test modelu bez znajomosci geometrii)
//   GEOMETRY - tylko punkty w oknach R/z odpowiadajacych TOF (dostosuj ponizej!)
enum Mode { ALL, GEOMETRY };
const Mode MODE = GEOMETRY;

// Okna geometryczne TOF [mm] - DOSTOSUJ do swojej geometrii (sprawdz wydruk R,z)
const double BTOF_RMIN = 600, BTOF_RMAX = 720, BTOF_ZMAX = 1200;   // barrel
const double FTOF_ZMIN = 1750, FTOF_ZMAX = 1950, FTOF_RMIN = 100, FTOF_RMAX = 650; // forward/backward (|z|)

const bool FIRST_ONLY_PER_SEGMENT = (MODE == GEOMETRY); // w trybie GEOMETRY tylko pierwszy punkt TOF
// ----------------------------------

static bool inTOF(double x, double y, double z) {
  const double R = std::hypot(x, y), az = std::fabs(z);
  const bool barrel = (R >= BTOF_RMIN && R <= BTOF_RMAX && az <= BTOF_ZMAX);
  const bool endcap = (az >= FTOF_ZMIN && az <= FTOF_ZMAX && R >= FTOF_RMIN && R <= FTOF_RMAX);
  return barrel || endcap;
}

void TrackLength() {
  glob_t g;
  glob(FILE_PATTERN.c_str(), 0, nullptr, &g);
  std::vector<std::string> files;
  for (size_t i = 0; i < g.gl_pathc; i++) files.emplace_back(g.gl_pathv[i]);
  globfree(&g);

  if (files.empty()) { std::cerr << "Brak plikow: " << FILE_PATTERN << std::endl; return; }

  podio::ROOTReader reader;
  reader.openFiles(files);
  const size_t nev = std::min<size_t>(reader.getEntries("events"), MAX_EVENTS);

  TH1F h_diff("h_diff", "L_{reco} - L_{ToFSim};#DeltaL [mm];points", 200, -50, 50);
  TH1F h_rel ("h_rel",  "(L_{reco}-L_{ToFSim})/L_{reco};rel. diff;points", 200, -0.1, 0.1);
  TH2F h_vs  ("h_vs",   "#DeltaL vs L_{reco};L_{reco} [mm];#DeltaL [mm]", 100, 0, 2500, 100, -50, 50);

  long n_ev_noMC = 0, n_pts = 0, n_sel = 0, n_distfail = 0, n_filled = 0;

  for (size_t ev = 0; ev < nev; ev++) {
    podio::Frame frame(reader.readEntry("events", ev));
    const auto& segs = frame.get<edm4eic::TrackSegmentCollection>(SEG_COLL);
    const auto& mcs  = frame.get<edm4hep::MCParticleCollection>(MC_COLL);

    // pojedynczy mion: pierwsza naladowana czastka final-state
    bool found = false;
    double px = 0, py = 0, pz = 0;
    int charge = 0;
    for (const auto p : mcs) {
      if (p.getGeneratorStatus() != 1) continue;
      if (p.getCharge() == 0) continue;
      auto m = p.getMomentum();
      if (m.x*m.x + m.y*m.y + m.z*m.z > 1) continue;   // p < 100 MeV
      px = m.x; py = m.y; pz = m.z;
      charge = (p.getCharge() > 0) ? 1 : -1;
      found = true;
      break;
    }
    if (!found) { n_ev_noMC++; continue; }

    TLorentzVector p4;
    p4.SetXYZM(px, py, pz, MUON_MASS);

    for (const auto seg : segs) {
      for (const auto pt : seg.getPoints()) {
        const double x = pt.position.x, y = pt.position.y, z = pt.position.z;
        n_pts++;

        if (ev < 4 && DEBUG_PRINT){
          std::cout << "pathlength=" << pt.pathlength << " mm, pos=(" << x << "," << y << "," << z
                    << ") R=" << std::hypot(x, y) << " z=" << z << "\n";
          std::cout << "============================" << std::endl; 
        }
        if (MODE == GEOMETRY && !inTOF(x, y, z)) continue;
        n_sel++;

        TVector3 tof(x, y, z);   // mm
        auto sim = ToFSim(p4, charge, tof);
        if (!sim.DistanceCheck) { n_distfail++; continue; }

        const double L_reco = pt.pathlength;
        const double L_sim  = sim.track_length;
        h_diff.Fill(L_reco - L_sim);
        h_vs.Fill(L_reco, L_reco - L_sim);
        if (L_reco > 0) h_rel.Fill((L_reco - L_sim) / L_reco);
        n_filled++;

        if (FIRST_ONLY_PER_SEGMENT) break;
      }
    }
  }

  TFile out("pathlength_compare.root", "RECREATE");
  h_diff.Write();
  h_rel.Write();
  h_vs.Write();
  out.Close();

  std::cout << "events=" << nev << " (bez MC: " << n_ev_noMC << ")"
            << ", punkty=" << n_pts << ", wybrane=" << n_sel
            << ", DistanceCheck fail=" << n_distfail
            << ", wypelnione=" << n_filled << "\n";
  if (n_filled == 0)
    std::cout << "UWAGA: nic nie wypelnione - sprawdz okna geometryczne / DistanceCheck w ToFSim\n";
  std::cout << "mean dL = " << h_diff.GetMean() << " mm, RMS = " << h_diff.GetRMS() << " mm\n";
}